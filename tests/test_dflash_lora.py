"""LoRA wrap on a tiny Splash-shaped Q4 linear does not need the 27B pack."""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from monkeyinference.dflash_lora import LoRAQ4
from monkeyinference.splash_q4 import Q4Linear, pack_mlx_q4


def _q4(n=128, k=256, seed=0) -> Q4Linear:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    scales = (0.02 + 0.01 * rng.random((n, k // 64))).astype(np.float32)
    biases = (-0.1 + 0.05 * rng.random((n, k // 64))).astype(np.float32)
    w, s, b = pack_mlx_q4(codes, scales, biases)
    return Q4Linear(w, s, b)


def test_lora_zero_b_matches_base():
    lin = _q4()
    x = mx.random.normal((8, 256)).astype(mx.float16)
    base = lin(x)
    lora = LoRAQ4(lin, r=8, scale=2.0)
    mx.eval(lora.lora_b)
    got = lora(x)
    mx.eval(got, base)
    err = float(mx.max(mx.abs(got.astype(mx.float32) - base.astype(mx.float32))).item())
    assert err < 1e-3, err


def test_lora_nonzero_changes_output():
    lin = _q4(seed=1)
    x = mx.random.normal((4, 256)).astype(mx.float16)
    lora = LoRAQ4(lin, r=8, scale=2.0)
    lora.lora_b = mx.ones_like(lora.lora_b) * 0.01
    mx.eval(lora.lora_b)
    y0 = lin(x)
    y1 = lora(x)
    mx.eval(y0, y1)
    err = float(mx.max(mx.abs(y1.astype(mx.float32) - y0.astype(mx.float32))).item())
    assert err > 1e-4, err


if __name__ == "__main__":
    test_lora_zero_b_matches_base()
    print("ok test_lora_zero_b_matches_base")
    test_lora_nonzero_changes_output()
    print("ok test_lora_nonzero_changes_output")
    print("all passed")
