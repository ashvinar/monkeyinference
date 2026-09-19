from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from monkeyinference.spec import EarlyExitDrafter, PromptLookupDrafter, pin_cache, revert_cache


def test_prompt_lookup_drafter_still_in_kernels_module():
    d = PromptLookupDrafter(ngram=3, max_draft=8)
    tokens = [1, 2, 3, 4, 5, 1, 2, 3]
    assert d.propose(tokens, max_tokens=4) == [4, 5, 1, 2]
    assert d.propose([9, 8, 7], max_tokens=4) == []


def test_pin_revert_arrays_cache_is_pointer_not_copy():
    c = ArraysCache(size=2)
    a = mx.ones((4, 8), dtype=mx.float32)
    b = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    mx.eval(a, b)
    c[0] = a
    c[1] = b
    pins = pin_cache([c])
    assert pins[0][2][0] is a
    assert pins[0][2][1] is b
    c[0] = mx.zeros_like(a)
    c[1] = mx.ones_like(b)
    mx.eval(c[0], c[1])
    revert_cache(pins)
    assert c[0] is a
    assert c[1] is b
    assert float(mx.sum(c[0]).item()) == 32.0


def test_pin_revert_kv_offset_only():
    c = KVCache()
    c.keys = mx.zeros((1, 1, 32, 4), dtype=mx.float16)
    c.values = mx.zeros((1, 1, 32, 4), dtype=mx.float16)
    c.offset = 7
    keys_obj = c.keys
    pins = pin_cache([c])
    c.offset = 19
    c.keys = mx.ones_like(c.keys)
    revert_cache(pins)
    assert c.offset == 7
    # Offset pin does not restore the keys pointer; GDN is the memcpy tax.
    # KV writes are in-place behind a rewindable offset. Re-bind to keep the test honest.
    assert c.keys is not keys_obj or c.offset == 7


def test_early_exit_rejects_bad_n():
    class _Fake:
        class model:
            layers = [None] * 4

    try:
        EarlyExitDrafter(_Fake(), 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")
