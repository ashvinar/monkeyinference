"""DFlash 2 draft: 5-layer block-diffusion + selector, Splash Q4 weights.

This is not an autoregressive small LM. Query slots are
`[leftover_token, MASK×7]`. Context KV is projected from concatenated Bonsai
hiddens at layers 5/19/33/47/61 (25600 → 5120). Logits use Bonsai's `lm_head`
so the verify path stays on the target distribution. Decoding is still
leftover-greedy: accepted tokens must match the target argmax.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope

from monkeyinference.splash_q4 import (
    BLOCK,
    CONV_GROUP,
    CONV_TAPS,
    HEAD_DIM,
    HIDDEN,
    KV_HEADS,
    LAYERS,
    MASK_TOKEN_ID,
    N_HEADS,
    PROPOSAL_TOKENS,
    SELECTOR_TOP_K,
    TARGET_HIDDEN,
    DraftWeights,
    load_dflash_draft,
)

TARGET_LAYER_IDS = (5, 19, 33, 47, 61)


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    orig = x.dtype
    x32 = x.astype(mx.float32)
    var = mx.mean(x32 * x32, axis=-1, keepdims=True)
    y = x32 * mx.rsqrt(var + eps) * weight.astype(mx.float32)
    return y.astype(orig)


def grouped_conv(
    hidden: mx.array,
    delta: mx.array,
    base: mx.array,
    *,
    block_size: int = BLOCK,
    group_size: int = CONV_GROUP,
    taps: int = CONV_TAPS,
) -> mx.array:
    """Two-tap grouped dynamic conv (vLLM `_grouped_conv`).

    hidden: [T, H], delta: [T, taps, groups], base: [taps, H]
    """
    t, h = int(hidden.shape[-2]), int(hidden.shape[-1])
    num_groups = h // group_size
    blocks = hidden.reshape(t, num_groups, group_size)
    coeff = base.reshape(1, taps, num_groups, group_size) + mx.expand_dims(delta, -1)
    out = coeff[:, 0] * blocks
    pos = mx.arange(t) % block_size
    for tap in range(1, taps):
        pad = mx.zeros((tap, num_groups, group_size), dtype=hidden.dtype)
        shifted = mx.concatenate([pad, blocks[: t - tap]], axis=0)
        gate = (pos >= tap).reshape(t, 1, 1).astype(hidden.dtype)
        out = out + coeff[:, tap] * shifted * gate
    return out.reshape(t, h)


class AuxCapture:
    """Record residual-stream hiddens at DFlash's five target layer ids."""

    def __init__(self, text_model):
        self.ids = TARGET_LAYER_IDS
        self.last: dict[int, mx.array] = {}
        self.tokens: list[mx.array] = []
        self._bound: list[tuple] = []
        for idx in self.ids:
            layer = text_model.layers[idx]
            orig = layer.__call__

            def bind(fn, layer_idx):
                def wrapped(*args, **kwargs):
                    y = fn(*args, **kwargs)
                    self.last[layer_idx] = y
                    return y

                return wrapped

            layer.__call__ = bind(orig, idx)
            self._bound.append((layer, orig))

    def close(self) -> None:
        for layer, orig in self._bound:
            layer.__call__ = orig
        self._bound.clear()

    def record_last_forward(self) -> mx.array:
        missing = [i for i in self.ids if i not in self.last]
        if missing:
            raise RuntimeError(f"aux capture missed layers {missing}")
        cat = mx.concatenate([self.last[i] for i in self.ids], axis=-1)
        # Drop batch dim → [T, 25600]
        if cat.ndim == 3:
            cat = cat.reshape(cat.shape[1], cat.shape[2])
        self.tokens.append(cat)
        return cat

    def context(self) -> mx.array | None:
        if not self.tokens:
            return None
        return mx.concatenate(self.tokens, axis=0)

    def keep_last_n(self, n: int) -> None:
        """After a verify of L tokens, keep the first `n` of that chunk."""
        if not self.tokens:
            return
        last = self.tokens[-1]
        if n <= 0:
            self.tokens.pop()
            return
        if n < int(last.shape[0]):
            self.tokens[-1] = last[:n]


class DFlashDrafter:
    def __init__(self, weights: DraftWeights, *, rope_theta: float = 10_000_000.0):
        self.w = weights
        self.n_heads = N_HEADS
        self.n_kv = KV_HEADS
        self.head_dim = HEAD_DIM
        self.scale = HEAD_DIM**-0.5
        self.rope = initialize_rope(
            HEAD_DIM,
            base=rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=262144,
        )
        self.q_size = N_HEADS * HEAD_DIM
        self.kv_size = KV_HEADS * HEAD_DIM

    @property
    def bytes_on_disk(self) -> int:
        return self.w.bytes_on_disk

    def _conv_pair(self, hidden: mx.array, dynamic: mx.array, base: mx.array):
        # dynamic: [T, 1280] = 2 sides × 2 taps × 320 groups
        t = int(hidden.shape[0])
        coeff = dynamic.reshape(t, 2, CONV_TAPS, HIDDEN // CONV_GROUP)
        prepared = grouped_conv(hidden, coeff[:, 0], base[0])
        return prepared, coeff[:, 1]

    def _split_qkv(self, qkv: mx.array):
        q = qkv[..., : self.q_size]
        k = qkv[..., self.q_size : self.q_size + self.kv_size]
        v = qkv[..., self.q_size + self.kv_size :]
        return q, k, v

    def _heads(self, x: mx.array, n_heads: int) -> mx.array:
        # x: [T, n_heads * d] → [1, n_heads, T, d]
        t = int(x.shape[0])
        return x.reshape(1, t, n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _from_heads(self, x: mx.array) -> mx.array:
        return x.transpose(0, 2, 1, 3).reshape(int(x.shape[2]), -1)

    def commit_context(self, aux: mx.array, cache: list) -> None:
        """Project captured target hiddens and write draft K/V (no query)."""
        if aux.ndim == 3:
            aux = aux.reshape(-1, TARGET_HIDDEN)
        projected = self.w.context_proj(aux)
        hidden = rms_norm(projected, self.w.hidden_norm)
        t = int(hidden.shape[0])
        for layer, c in zip(self.w.layers, cache):
            qkv = layer.qkv(hidden)
            _, k, v = self._split_qkv(qkv)
            k = rms_norm(k.reshape(t, self.n_kv, self.head_dim), layer.k_norm).reshape(
                t, self.kv_size
            )
            keys = self._heads(k, self.n_kv)
            values = self._heads(v, self.n_kv)
            keys = self.rope(keys, offset=c.offset)
            c.update_and_fetch(keys, values)

    def _attn(self, hidden: mx.array, layer, cache) -> mx.array:
        t = int(hidden.shape[0])
        qkv = layer.qkv(hidden)
        q, k, v = self._split_qkv(qkv)
        q = rms_norm(q.reshape(t, self.n_heads, self.head_dim), layer.q_norm).reshape(
            t, self.q_size
        )
        k = rms_norm(k.reshape(t, self.n_kv, self.head_dim), layer.k_norm).reshape(
            t, self.kv_size
        )
        queries = self.rope(self._heads(q, self.n_heads), offset=cache.offset)
        keys = self.rope(self._heads(k, self.n_kv), offset=cache.offset)
        values = self._heads(v, self.n_kv)
        keys, values = cache.update_and_fetch(keys, values)
        # Bidirectional over the query block; full attend to committed context.
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=None
        )
        return layer.o_proj(self._from_heads(out))

    def _mlp(self, hidden: mx.array, layer) -> mx.array:
        gate = layer.gate(hidden)
        up = layer.up(hidden)
        return layer.down(nn.silu(gate) * up)

    def backbone(self, hidden: mx.array, cache: list) -> mx.array:
        """hidden: [T, H] query rows. Mutates `cache`."""
        h = hidden
        for layer, c in zip(self.w.layers, cache):
            residual = h
            n = rms_norm(h, layer.input_norm)
            n, coeff = self._conv_pair(n, layer.attn_dynamic(n), layer.attn_conv)
            a = self._attn(n, layer, c)
            a = grouped_conv(a, coeff, layer.attn_conv[1])
            h = residual + a
            residual = h
            n = rms_norm(h, layer.post_attn_norm)
            n, coeff = self._conv_pair(n, layer.mlp_dynamic(n), layer.mlp_conv)
            m = self._mlp(n, layer)
            m = grouped_conv(m, coeff, layer.mlp_conv[1])
            h = residual + m
        return rms_norm(h, self.w.final_norm)

    def select(self, logits: mx.array, selector_hidden: mx.array, anchor: int, k: int) -> list[int]:
        """Greedy lattice walk over top-16 candidates at MASK slots 1..k."""
        k = min(k, PROPOSAL_TOKENS)
        if k <= 0:
            return []
        slots = logits[1 : 1 + k]
        hidden = selector_hidden[1 : 1 + k]
        path: list[int] = []
        prev = int(anchor)
        for i in range(k):
            row = slots[i]
            top = mx.argsort(-row)[:SELECTOR_TOP_K]
            mx.eval(top)
            cand = [int(x) for x in top.tolist()]
            unary = row[mx.array(cand, dtype=mx.uint32)]
            succ = self.w.successor[mx.array(cand, dtype=mx.uint32)]
            pred = self.w.predecessor[prev]
            h = hidden[i]
            scores = unary + (pred * h) @ succ.T
            mx.eval(scores)
            idx = int(mx.argmax(scores).item())
            tok = cand[idx]
            path.append(tok)
            prev = tok
        return path

    def propose(
        self,
        *,
        aux: mx.array,
        leftover_token: int,
        embed,
        lm_head,
        k: int = PROPOSAL_TOKENS,
    ) -> list[int]:
        cache = [KVCache() for _ in range(LAYERS)]
        self.commit_context(aux, cache)
        ids = [int(leftover_token)] + [MASK_TOKEN_ID] * PROPOSAL_TOKENS
        tokens = mx.array(ids, dtype=mx.uint32)
        hidden = embed(tokens)
        if hidden.ndim == 3:
            hidden = hidden.reshape(hidden.shape[1], hidden.shape[2])
        hidden = self.backbone(hidden, cache)
        logits = lm_head(hidden)
        sel_h = self.w.selector(hidden)
        mx.eval(logits, sel_h)
        return self.select(logits, sel_h, leftover_token, k)


def load_drafter(directory: str | Path | None = None) -> DFlashDrafter:
    return DFlashDrafter(load_dflash_draft(directory))
