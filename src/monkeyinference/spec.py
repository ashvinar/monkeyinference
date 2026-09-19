"""Draft models and cache snapshot/restore for speculative decode."""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx
from mlx_lm.models.cache import KVCache


class Drafter(Protocol):
    def propose(self, tokens: list[int], max_tokens: int) -> list[int]: ...


class PromptLookupDrafter:
    """N-gram prompt lookup (Stern et al. / Saxena PLD). Zero extra weights."""

    def __init__(self, ngram: int = 3, max_draft: int = 5):
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


def snapshot_cache(cache) -> list:
    """Deep-copy GDN arrays; record KV offsets. Must eval copies before the next forward."""
    snaps = []
    copies = []
    for c in cache:
        if isinstance(c, KVCache) or (
            hasattr(c, "keys") and hasattr(c, "offset") and hasattr(c, "trim")
        ):
            snaps.append(("kv", int(c.offset)))
        elif hasattr(c, "cache"):
            cloned = []
            for a in c.cache:
                if a is None:
                    cloned.append(None)
                else:
                    cloned.append(a + 0)
                    copies.append(cloned[-1])
            snaps.append(("arr", cloned, c.lengths, c.left_padding))
        else:
            raise TypeError(f"cannot snapshot cache type {type(c)}")
    if copies:
        mx.eval(*copies)
    return snaps


def restore_cache(cache, snaps) -> None:
    for c, s in zip(cache, snaps):
        if s[0] == "kv":
            c.offset = s[1]
        elif s[0] == "arr":
            c.cache = s[1]
            c.lengths = s[2]
            c.left_padding = s[3]
        else:
            raise TypeError(s[0])
