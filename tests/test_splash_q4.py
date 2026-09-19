from __future__ import annotations

import numpy as np

from monkeyinference.splash_q4 import (
    GROUP,
    STORAGE_N,
    pack_mlx_q4,
    pack_splash_q4,
    q4_packed_bytes,
    splash_q4_to_mlx,
    unpack_splash_q4_codes,
)
from monkeyinference.dflash import grouped_conv, rms_norm

import mlx.core as mx


def test_q4_packed_bytes_matches_splash_formula():
    assert q4_packed_bytes(256, 64) == (256 * 64 // 16) * 9
    assert q4_packed_bytes(1280, 5120) == 3686400
    assert q4_packed_bytes(5120, 25600) == 73728000


def test_synthetic_splash_q4_roundtrip_into_mlx():
    rng = np.random.default_rng(0)
    n, k = STORAGE_N, GROUP * 2  # 256 × 128
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    scales = (0.02 + 0.01 * rng.random((n, k // GROUP))).astype(np.float32)
    biases = (-0.1 + 0.05 * rng.random((n, k // GROUP))).astype(np.float32)
    packed = pack_splash_q4(codes, scales, biases)
    assert len(packed) == q4_packed_bytes(n, k)
    got_codes = unpack_splash_q4_codes(packed[: n * k // 2], n, k)
    assert np.array_equal(got_codes, codes)

    wq, s, b = splash_q4_to_mlx(packed, n, k)
    x = rng.standard_normal(k).astype(np.float32)
    # Splash affine: y_n = Σ_g (dot(q,x_g)*s + sum(x_g)*bias)
    ref = np.zeros(n, dtype=np.float32)
    for g in range(k // GROUP):
        sl = slice(g * GROUP, (g + 1) * GROUP)
        dots = codes[:, sl].astype(np.float32) @ x[sl]
        sums = float(x[sl].sum())
        ref += dots * scales[:, g] + sums * biases[:, g]
    y = mx.quantized_matmul(
        mx.array(x[None, :].astype(np.float16)),
        wq,
        s,
        b,
        transpose=True,
        group_size=GROUP,
        bits=4,
    )
    mx.eval(y)
    err = float(np.max(np.abs(np.array(y[0], dtype=np.float32) - ref)))
    # bf16 scale/bias truncation + fp16 qmm
    assert err < 0.15, err


def test_grouped_conv_identity_tap0_zero_delta():
    t, h = 8, 64
    hidden = mx.arange(t * h, dtype=mx.float32).reshape(t, h)
    base = mx.zeros((2, h), dtype=mx.float32)
    base[0, :] = 1.0  # tap 0 is identity, tap 1 is 0
    delta = mx.zeros((t, 2, h // 16), dtype=mx.float32)
    out = grouped_conv(hidden, delta, base, block_size=8, group_size=16, taps=2)
    mx.eval(out)
    assert float(mx.max(mx.abs(out - hidden)).item()) == 0.0


def test_rms_norm_unit_weight():
    x = mx.ones((4, 8), dtype=mx.float32)
    w = mx.ones((8,), dtype=mx.float32)
    y = rms_norm(x, w)
    mx.eval(y)
    # ones / rms(ones)=1
    assert abs(float(y[0, 0].item()) - 1.0) < 1e-5


def test_real_draft_files_unpack_attention_dynamic():
    from pathlib import Path

    from monkeyinference.splash_q4 import PackedFile, MAGIC, q4_packed_bytes, HIDDEN, DYNAMIC

    layer = Path.home() / ".monkey/models/Qwen3.8-27B-Splash-draft/draft/layer-0.bin"
    if not layer.is_file():
        return
    f = PackedFile(layer, MAGIC, 0, 0)
    f.section(HIDDEN * 2, "input-norm")
    f.section(4 * HIDDEN * 2, "attention-convolution")
    dyn = f.section(q4_packed_bytes(DYNAMIC, HIDDEN), "attention-dynamic")
    wq, s, b = splash_q4_to_mlx(dyn, DYNAMIC, HIDDEN)
    x = mx.random.normal((1, HIDDEN)).astype(mx.float16)
    y = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=GROUP, bits=4)
    mx.eval(y)
    assert tuple(y.shape) == (1, DYNAMIC)
    # Finite, non-zero — a silent zero unpack would fail this.
    mag = float(mx.mean(mx.abs(y)).item())
    assert mag > 1e-4, mag
    assert np.isfinite(mag)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")
