from __future__ import annotations

import mlx.core as mx
import numpy as np

from monkeyinference.hadamard import BLOCK, HADAMARD_SCALE, fwht
from monkeyinference.kernels import mlx_affine_qmv, stream_copy, ternary_gemv, ternary_qmm, ternary_qmv_once
from monkeyinference.roofline import pack_ternary
from monkeyinference.spec import PromptLookupDrafter


def test_stream_copy_roundtrip():
    x = mx.arange(1024, dtype=mx.float32)
    y = stream_copy(x)
    mx.eval(y)
    assert mx.allclose(x, y).item()


def test_fwht_matches_mlx_hadamard():
    width = 5120
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((2, width)).astype(np.float16))
    signs = mx.array(rng.choice([-1.0, 1.0], size=(width,)).astype(np.float32))
    got = fwht(x, signs, inverse=False)
    ref = mx.hadamard_transform(
        (x.astype(mx.float32) * signs).reshape(-1, BLOCK), scale=HADAMARD_SCALE
    ).reshape(x.shape).astype(x.dtype)
    mx.eval(got, ref)
    err = float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item())
    assert err < 1e-3, err


def test_fwht_inverse_involution():
    width = 2048
    rng = np.random.default_rng(2)
    x = mx.array(rng.standard_normal((width,)).astype(np.float16))
    signs = mx.array(rng.choice([-1.0, 1.0], size=(width,)).astype(np.float32))
    y = fwht(fwht(x, signs, inverse=False), signs, inverse=True)
    mx.eval(y)
    err = float(mx.max(mx.abs(y.astype(mx.float32) - x.astype(mx.float32))).item())
    assert err < 2e-2, err


def test_ternary_gemv_matches_numpy_and_mlx():
    for n, k in [(128, 512), (512, 1024), (1024, 2048), (17408, 5120)]:
        x, w, scales, biases = pack_ternary(n, k, seed=n + k)
        got = np.array(ternary_gemv(x, w, scales).astype(mx.float32))
        mlx = np.array(mlx_affine_qmv(x, w, scales, biases).astype(mx.float32))
        xx = np.array(x.astype(mx.float32))
        ww = np.array(w)
        sc = np.array(scales.astype(mx.float32))
        codes = np.empty((n, k), np.float32)
        for wi in range(k // 16):
            word = ww[:, wi]
            for lane in range(16):
                codes[:, wi * 16 + lane] = (word >> (2 * lane)) & 3
        recon = ((codes.reshape(n, k // 128, 128) - 1.0) * sc[:, :, None]).reshape(n, k)
        npy = recon @ xx
        npy_err = float(np.max(np.abs(npy - got)))
        mlx_err = float(np.max(np.abs(mlx - got)))
        # Packing must match dequant exactly; GEMV fp16 inputs accumulate a few ulps.
        deq = np.array(
            mx.dequantize(w[:4], scales[:4], biases[:4], group_size=128, bits=2).astype(mx.float32)
        )
        pack_err = float(np.max(np.abs(deq - recon[:4])))
        assert pack_err == 0.0, (pack_err, n, k)
        assert npy_err < 0.25, (npy_err, n, k)
        assert mlx_err < 0.50, (mlx_err, n, k)


def test_ternary_qmm_batch_matches_gemv():
    n, k, m = 256, 512, 3
    x, w, scales, biases = pack_ternary(n, k, seed=9)
    xs = mx.stack([x, x * 0.5, x * -1.0])
    got = ternary_qmm(xs, w, scales)
    refs = [ternary_gemv(xs[i], w, scales) for i in range(m)]
    mx.eval(got, *refs)
    for i in range(m):
        err = float(mx.max(mx.abs(got[i] - refs[i])).item())
        assert err < 1e-4, (i, err)


def test_ternary_qmv_once_matches_gemv():
    n, k, m = 256, 512, 6
    x, w, scales, biases = pack_ternary(n, k, seed=12)
    xs = mx.stack([x * ((i + 1) * 0.3) for i in range(m)])
    got = ternary_qmv_once(xs, w, scales)
    refs = [ternary_gemv(xs[i], w, scales) for i in range(m)]
    mx.eval(got, *refs)
    for i in range(m):
        err = float(mx.max(mx.abs(got[i] - refs[i])).item())
        assert err < 1e-4, (i, err)


def test_mlx_affine_flattens_3d_like_2d():
    n, k, m = 128, 512, 6
    x, w, scales, biases = pack_ternary(n, k, seed=11)
    xs = mx.stack([x * ((i + 1) * 0.25) for i in range(m)])
    y2 = mlx_affine_qmv(xs, w, scales, biases)
    y3 = mlx_affine_qmv(xs[None], w, scales, biases)
    mx.eval(y2, y3)
    err = float(mx.max(mx.abs(y2 - y3[0])).item())
    assert err == 0.0, err


def test_prompt_lookup_drafter():
    d = PromptLookupDrafter(ngram=3, max_draft=4)
    tokens = [1, 2, 3, 4, 5, 1, 2, 3]
    assert d.propose(tokens, max_tokens=4) == [4, 5, 1, 2]
    assert d.propose([9, 8, 7], max_tokens=4) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")
