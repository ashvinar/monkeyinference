"""Draft models and zero-copy cache pins for speculative decode.

Gated experiment. `generate(..., speculative=False)` is the production path.
On the M4 Air, PLD / early-exit / DFlash are net-negative vs greedy except
PLD on a copyable prompt. See docs/ternary-engine.md.

GDN (`ArraysCache`) and the Metal gated-delta kernel allocate a *new*
`state_out` each step and the layer does `cache[i] = new`. KVCache only
advances `offset` in a preallocated buffer. Holding the previous array
references / offset is therefore a true copy-on-write pin: restore is a
pointer swap, not a 150 MB memcpy.
"""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache


class Drafter(Protocol):
    def propose(self, tokens: list[int], max_tokens: int) -> list[int]: ...


class PromptLookupDrafter:
    """N-gram prompt lookup (Stern et al. / Saxena PLD). Zero extra weights."""

    def __init__(self, ngram: int = 3, max_draft: int = 8):
        self.ngram = ngram
        self.max_draft = max_draft

    def propose(self, tokens: list[int], max_tokens: int) -> list[int]:
        n = self.ngram
        if max_tokens <= 0 or len(tokens) < n + 1:
            return []
        needle = tokens[-n:]
        limit = len(tokens) - n
        for i in range(limit - 1, -1, -1):
            if tokens[i : i + n] == needle:
                end = min(i + n + min(self.max_draft, max_tokens), len(tokens))
                draft = tokens[i + n : end]
                if draft:
                    return list(draft)
        return []


def pin_cache(cache) -> list:
    """Copy-on-write pin: shallow list clone + KV offsets. No device memcpy."""
    pins = []
    for c in cache:
        if isinstance(c, KVCache) or (
            hasattr(c, "keys") and hasattr(c, "offset") and hasattr(c, "trim")
        ):
            pins.append(("kv", c, int(c.offset)))
        elif hasattr(c, "cache"):
            pins.append(("arr", c, c.cache.copy(), c.lengths, c.left_padding))
        else:
            raise TypeError(f"cannot pin cache type {type(c)}")
    return pins


def revert_cache(pins) -> None:
    """Restore pinned references. Does not copy device memory."""
    for item in pins:
        if item[0] == "kv":
            _, c, offset = item
            c.offset = offset
        elif item[0] == "arr":
            _, c, arrays, lengths, pad = item
            c.cache = arrays
            c.lengths = lengths
            c.left_padding = pad
        else:
            raise TypeError(item[0])


# Back-compat names used by the first milestone; pin is the real implementation.
def snapshot_cache(cache) -> list:
    return pin_cache(cache)


def restore_cache(cache, snaps) -> None:
    revert_cache(snaps)


class EarlyExitDrafter:
    """Self-speculation: first N layers + shared `lm_head`. Same 248320 vocab, 0 extra bytes."""

    def __init__(self, model, n_layers: int):
        layers = model.model.layers
        if n_layers < 1 or n_layers > len(layers):
            raise ValueError(f"n_layers={n_layers} out of range 1..{len(layers)}")
        self.model = model
        self.n_layers = int(n_layers)
        self.layers = layers[: self.n_layers]
        self.embed = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.ssm_idx = 0
        self.fa_idx = next((i for i, layer in enumerate(self.layers) if not layer.is_linear), None)

    def make_cache(self):
        return [ArraysCache(size=2) if layer.is_linear else KVCache() for layer in self.layers]

    def logits(self, ids: mx.array, cache) -> mx.array:
        if ids.ndim == 1:
            ids = ids[None]
        hidden = self.embed(ids)
        ssm_mask = create_ssm_mask(hidden, cache[self.ssm_idx])
        fa_mask = (
            create_attention_mask(hidden, cache[self.fa_idx])
            if self.fa_idx is not None
            else None
        )
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=c)
        return self.lm_head(self.norm(hidden))

    def propose_from_state(self, y: mx.array, cache, k: int, eos: set[int]) -> list[int]:
        """Autoregressive draft from leftover token `y`. Mutates `cache`."""
        draft: list[int] = []
        cur = y
        for _ in range(k):
            logits = self.logits(cur, cache)
            mx.eval(logits)
            tok = int(mx.argmax(logits[:, -1, :]).item())
            draft.append(tok)
            cur = mx.array([tok], dtype=mx.uint32)
            if tok in eos:
                break
        return draft
