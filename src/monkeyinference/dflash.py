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

from monkeyinference.spec import pin_cache, revert_cache
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


class _LayerTap(nn.Module):
    def __init__(self, inner, store: dict, idx: int):
        super().__init__()
        self.inner = inner
        self._store = store
        self._idx = idx
        self.is_linear = getattr(inner, "is_linear", False)

    def __call__(self, *args, **kwargs):
        y = self.inner(*args, **kwargs)
        self._store[self._idx] = y
        return y


class AuxCapture:
    """Record residual-stream hiddens at DFlash's five target layer ids."""

    def __init__(self, text_model):
        self.ids = TARGET_LAYER_IDS
        self.last: dict[int, mx.array] = {}
        self.tokens: list[mx.array] = []
        self._layers = text_model.layers
        self._orig: dict[int, object] = {}
        for idx in self.ids:
            inner = self._layers[idx]
            self._orig[idx] = inner
            self._layers[idx] = _LayerTap(inner, self.last, idx)

    def close(self) -> None:
        for idx, inner in self._orig.items():
            self._layers[idx] = inner
        self._orig.clear()

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

    def __len__(self) -> int:
        return sum(int(t.shape[0]) for t in self.tokens)

    def uncommitted(self, committed: int) -> mx.array | None:
        """Rows of `context()` not yet written into the draft KV cache."""
        ctx = self.context()
        if ctx is None:
            return None
        n = int(ctx.shape[0])
        if committed >= n:
            return None
        return ctx[committed:]


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
        self.ctx_cache: list = []
        self.reset_cache()

    @property
    def bytes_on_disk(self) -> int:
        return self.w.bytes_on_disk

    def reset_cache(self) -> None:
        """Drop draft context KV. Call at the start of each generate."""
        self.ctx_cache = [KVCache() for _ in range(LAYERS)]

    def commit_new(self, aux: mx.array) -> None:
        """Append newly accepted target hiddens onto the persistent draft KV."""
        self.commit_context(aux, self.ctx_cache)

    def _q4(self, lin, x: mx.array) -> mx.array:
        """fp16 Q4 in, fp32 out. Residual magnitudes (~5e4) are not fp16-precise."""
        return lin(x.astype(mx.float16)).astype(mx.float32)

    def _q4_f32(self, lin, x: mx.array) -> mx.array:
        """fp32 accum. context_proj biases ±21 overflow an fp16 matmul output."""
        return lin(x.astype(mx.float32))

    def _conv_pair(self, hidden: mx.array, dynamic: mx.array, base: mx.array):
        # dynamic: [T, 1280] = 2 sides × 2 taps × 320 groups
        t = int(hidden.shape[0])
        coeff = dynamic.reshape(t, 2, CONV_TAPS, HIDDEN // CONV_GROUP)
        prepared = grouped_conv(hidden, coeff[:, 0], base[0].astype(hidden.dtype))
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
        projected = self._q4_f32(self.w.context_proj, aux)
        hidden = rms_norm(projected, self.w.hidden_norm).astype(mx.float16)
        t = int(hidden.shape[0])
        for layer, c in zip(self.w.layers, cache):
            qkv = self._q4(layer.qkv, hidden)
            _, k, v = self._split_qkv(qkv)
            k = rms_norm(k.reshape(t, self.n_kv, self.head_dim), layer.k_norm).reshape(
                t, self.kv_size
            )
            keys = self._heads(k, self.n_kv).astype(mx.float16)
            values = self._heads(v, self.n_kv).astype(mx.float16)
            keys = self.rope(keys, offset=c.offset)
            c.update_and_fetch(keys, values)

    def _attn(self, hidden: mx.array, layer, cache) -> mx.array:
        t = int(hidden.shape[0])
        qkv = self._q4(layer.qkv, hidden)
        q, k, v = self._split_qkv(qkv)
        q = rms_norm(q.reshape(t, self.n_heads, self.head_dim), layer.q_norm).reshape(
            t, self.q_size
        )
        k = rms_norm(k.reshape(t, self.n_kv, self.head_dim), layer.k_norm).reshape(
            t, self.kv_size
        )
        queries = self.rope(self._heads(q, self.n_heads).astype(mx.float16), offset=cache.offset)
        keys = self.rope(self._heads(k, self.n_kv).astype(mx.float16), offset=cache.offset)
        values = self._heads(v, self.n_kv).astype(mx.float16)
        keys, values = cache.update_and_fetch(keys, values)
        out = mx.fast.scaled_dot_product_attention(
            queries.astype(mx.float32),
            keys.astype(mx.float32),
            values.astype(mx.float32),
            scale=self.scale,
            mask=None,
        )
        return self._q4(layer.o_proj, self._from_heads(out))

    def _mlp(self, hidden: mx.array, layer) -> mx.array:
        gate = self._q4(layer.gate, hidden)
        up = self._q4(layer.up, hidden)
        return self._q4(layer.down, nn.silu(gate) * up)

    def backbone(self, hidden: mx.array, cache: list) -> mx.array:
        """hidden: [T, H] query rows. Residual stays float32: |h|~5e4, fp16 ULP is 32."""
        h = hidden.astype(mx.float32)
        for layer, c in zip(self.w.layers, cache):
            residual = h
            n = rms_norm(h, layer.input_norm)
            n, coeff = self._conv_pair(n, self._q4(layer.attn_dynamic, n), layer.attn_conv)
            a = self._attn(n, layer, c)
            a = grouped_conv(a, coeff.astype(a.dtype), layer.attn_conv[1].astype(a.dtype))
            h = residual + a
            residual = h
            n = rms_norm(h, layer.post_attn_norm)
            n, coeff = self._conv_pair(n, self._q4(layer.mlp_dynamic, n), layer.mlp_conv)
            m = self._mlp(n, layer)
            m = grouped_conv(m, coeff.astype(m.dtype), layer.mlp_conv[1].astype(m.dtype))
            h = residual + m
        return rms_norm(h, self.w.final_norm)

    def select(self, logits: mx.array, selector_hidden: mx.array, anchor: int, k: int) -> list[int]:
        """Greedy lattice walk over top-16 candidates at MASK slots 1..k."""
        k = min(k, PROPOSAL_TOKENS)
        if k <= 0:
            return []
        slots = logits[1 : 1 + k]
        hidden = selector_hidden[1 : 1 + k]
        top = mx.argsort(-slots, axis=-1)[:, :SELECTOR_TOP_K]
        mx.eval(top)
        path: list[int] = []
        prev = int(anchor)
        for i in range(k):
            cand = top[i]
            unary = slots[i][cand]
            succ = self.w.successor[cand]
            pred = self.w.predecessor[prev]
            scores = unary.astype(mx.float32) + (pred * hidden[i]).astype(mx.float32) @ succ.astype(
                mx.float32
            ).T
            idx = int(mx.argmax(scores).item())
            tok = int(cand[idx].item())
            path.append(tok)
            prev = tok
        return path

    def propose(
        self,
        *,
        leftover_token: int,
        embed,
        lm_head,
        k: int = PROPOSAL_TOKENS,
    ) -> list[int]:
        """Query leftover+MASK against the persistent context KV. Does not rebuild context."""
        cache = self.ctx_cache
        pin = pin_cache(cache)
        try:
            ids = [int(leftover_token)] + [MASK_TOKEN_ID] * PROPOSAL_TOKENS
            tokens = mx.array(ids, dtype=mx.uint32)
            hidden = embed(tokens)
            if hidden.ndim == 3:
                hidden = hidden.reshape(hidden.shape[1], hidden.shape[2])
            hidden = self.backbone(hidden, cache)
            logits = lm_head(hidden.astype(mx.float16))
            sel_h = self._q4(self.w.selector, hidden)
            mx.eval(sel_h, logits)
            return self.select(
                logits.reshape(int(logits.shape[-2]), int(logits.shape[-1])),
                sel_h,
                leftover_token,
                k,
            )
        finally:
            revert_cache(pin)


_DRAFTER: DFlashDrafter | None = None


def load_drafter(directory: str | Path | None = None) -> DFlashDrafter:
    """Load once. Draft Q4 unpack is ~1.3 GB and should not hit the generate timer twice."""
    global _DRAFTER
    if _DRAFTER is None:
        _DRAFTER = DFlashDrafter(load_dflash_draft(directory))
    else:
        _DRAFTER.reset_cache()
    return _DRAFTER
