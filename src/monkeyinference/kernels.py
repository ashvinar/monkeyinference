"""Metal kernels for STREAM bandwidth and ternary affine-2bit GEMV.

Weight layout matches MLX / Prism affine 2-bit g128:
  w:      uint32[N, K/16]   16 codes per word, 2 bits each, K-contiguous
  scales: float16[N, K/128]
  codes {0,1,2} decode to {-s, 0, +s} via (code - 1) * scale
  Prism stores bias = -scale; this kernel never loads bias.

Decode (seq=1) is a GEMV. Prefill keeps mx.quantized_matmul.
"""

from __future__ import annotations

import mlx.core as mx

GROUP = 128
PACK = 16  # 2-bit codes per uint32

_STREAM_COPY_SRC = r"""
    uint i = thread_position_in_grid.x;
    out[i] = inp[i];
"""

_TERNARY_QMV_SRC = r"""
    // MLX qmv_fast layout: 64 threads / TG, 2 simdgroups, 4 rows each.
    const uint tid = thread_position_in_grid.x;
    const uint tg = tid / 64u;
    const uint lid = tid % 64u;
    const uint simd_gid = lid / 32u;
    const uint simd_lid = lid % 32u;
    const uint out_row = tg * 8u + simd_gid * 4u;
    const uint words_per_row = K / 16u;
    const uint groups_per_row = K / 128u;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint w_idx = simd_lid; w_idx < words_per_row; w_idx += 32u) {
        const uint kbase = w_idx * 16u;
        const uint group = w_idx / 8u;
        const float x0 = float(x[kbase + 0]);
        const float x1 = float(x[kbase + 1]);
        const float x2 = float(x[kbase + 2]);
        const float x3 = float(x[kbase + 3]);
        const float x4 = float(x[kbase + 4]);
        const float x5 = float(x[kbase + 5]);
        const float x6 = float(x[kbase + 6]);
        const float x7 = float(x[kbase + 7]);
        const float x8 = float(x[kbase + 8]);
        const float x9 = float(x[kbase + 9]);
        const float x10 = float(x[kbase + 10]);
        const float x11 = float(x[kbase + 11]);
        const float x12 = float(x[kbase + 12]);
        const float x13 = float(x[kbase + 13]);
        const float x14 = float(x[kbase + 14]);
        const float x15 = float(x[kbase + 15]);

        #pragma unroll
        for (uint r = 0; r < 4u; r++) {
            const uint row = out_row + r;
            if (row >= N) {
                continue;
            }
            const uint word = w[row * words_per_row + w_idx];
            const float s = float(scales[row * groups_per_row + group]);
            float local = 0.0f;
            local += (float((word >> 0) & 3u) - 1.0f) * x0;
            local += (float((word >> 2) & 3u) - 1.0f) * x1;
            local += (float((word >> 4) & 3u) - 1.0f) * x2;
            local += (float((word >> 6) & 3u) - 1.0f) * x3;
            local += (float((word >> 8) & 3u) - 1.0f) * x4;
            local += (float((word >> 10) & 3u) - 1.0f) * x5;
            local += (float((word >> 12) & 3u) - 1.0f) * x6;
            local += (float((word >> 14) & 3u) - 1.0f) * x7;
            local += (float((word >> 16) & 3u) - 1.0f) * x8;
            local += (float((word >> 18) & 3u) - 1.0f) * x9;
            local += (float((word >> 20) & 3u) - 1.0f) * x10;
            local += (float((word >> 22) & 3u) - 1.0f) * x11;
            local += (float((word >> 24) & 3u) - 1.0f) * x12;
            local += (float((word >> 26) & 3u) - 1.0f) * x13;
            local += (float((word >> 28) & 3u) - 1.0f) * x14;
            local += (float((word >> 30) & 3u) - 1.0f) * x15;
            const float term = local * s;
            if (r == 0) acc0 += term;
            else if (r == 1) acc1 += term;
            else if (r == 2) acc2 += term;
            else acc3 += term;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (simd_lid == 0) {
        if (out_row + 0 < N) y[out_row + 0] = T(acc0);
        if (out_row + 1 < N) y[out_row + 1] = T(acc1);
        if (out_row + 2 < N) y[out_row + 2] = T(acc2);
        if (out_row + 3 < N) y[out_row + 3] = T(acc3);
    }
"""

# Prefill path: each threadgroup does ROWS output rows for one token in the batch.
# x is [M, K], y is [M, N]. tid.y selects the token.
_TERNARY_QMM_SRC = r"""
    const uint tid = thread_position_in_grid.x;
    const uint lid = tid % 32u;
    const uint row0 = (tid / 32u) * ROWS;
    const uint token = thread_position_in_grid.y;
    const uint words_per_row = K / 16u;
    const uint groups_per_row = K / 128u;
    const device T* xrow = x + token * K;

    float acc[ROWS];
    for (uint r = 0; r < ROWS; r++) {
        acc[r] = 0.0f;
    }

    for (uint w_idx = lid; w_idx < words_per_row; w_idx += 32u) {
        const uint kbase = w_idx * 16u;
        const uint group = w_idx / 8u;
        float xv[16];
        #pragma unroll
        for (int lane = 0; lane < 16; lane++) {
            xv[lane] = float(xrow[kbase + uint(lane)]);
        }

        for (uint r = 0; r < ROWS; r++) {
            const uint row = row0 + r;
            if (row >= N) {
                continue;
            }
            const uint word = w[row * words_per_row + w_idx];
            const float s = float(scales[row * groups_per_row + group]);
            float local = 0.0f;
            #pragma unroll
            for (int lane = 0; lane < 16; lane++) {
                const uint code = (word >> (2 * lane)) & 3u;
                local += (float(code) - 1.0f) * xv[lane];
            }
            acc[r] += local * s;
        }
    }

    for (uint r = 0; r < ROWS; r++) {
        acc[r] = simd_sum(acc[r]);
        const uint row = row0 + r;
        if (lid == 0 && row < N) {
            y[token * N + row] = T(acc[r]);
        }
    }
"""

_stream_copy_kernel = None
_qmv_cache: dict[tuple[int, int, int, type], object] = {}
_qmm_cache: dict[tuple[int, int, int, type], object] = {}

ROWS_DEFAULT = 8


def _stream_copy():
    global _stream_copy_kernel
    if _stream_copy_kernel is None:
        _stream_copy_kernel = mx.fast.metal_kernel(
            name="monkey_stream_copy",
            input_names=["inp"],
            output_names=["out"],
            source=_STREAM_COPY_SRC,
        )
    return _stream_copy_kernel


def stream_copy(inp: mx.array) -> mx.array:
    """Device copy used as a STREAM-like bandwidth probe."""
    n = int(inp.size)
    tg = 256
    grid = ((n + tg - 1) // tg) * tg
    return _stream_copy()(
        inputs=[inp],
        template=[("T", inp.dtype)],
        grid=(grid, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[inp.shape],
        output_dtypes=[inp.dtype],
    )[0]


def _qmv_kernel(n: int, k: int, rows: int, dtype):
    key = (n, k, rows, dtype)
    kernel = _qmv_cache.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="monkey_ternary_qmv",
            input_names=["x", "w", "scales"],
            output_names=["y"],
            header="#include <metal_simdgroup>\n",
            source=_TERNARY_QMV_SRC,
        )
        _qmv_cache[key] = kernel
    return kernel


def _qmm_kernel(n: int, k: int, rows: int, dtype):
    key = (n, k, rows, dtype)
    kernel = _qmm_cache.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="monkey_ternary_qmm",
            input_names=["x", "w", "scales"],
            output_names=["y"],
            header="#include <metal_simdgroup>\n",
            source=_TERNARY_QMM_SRC,
        )
        _qmm_cache[key] = kernel
    return kernel


def ternary_gemv(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    *,
    rows: int = ROWS_DEFAULT,
) -> mx.array:
    """y = ternary_W x  for a single vector x [K] or [1, K] or [..., K] with prod=K."""
    x = mx.reshape(x, (-1,))
    k = int(x.shape[0])
    n = int(weight.shape[0])
    if k % GROUP:
        raise ValueError(f"K={k} is not a multiple of group size {GROUP}")
    if weight.shape[1] * PACK != k:
        raise ValueError(f"weight width {weight.shape[1]} does not pack K={k}")
    n_tg = (n + 7) // 8
    out = _qmv_kernel(n, k, 8, x.dtype)(
        inputs=[x, weight, scales],
        template=[("T", x.dtype), ("N", n), ("K", k), ("ROWS", 8)],
        grid=(n_tg * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )[0]
    return out


def ternary_qmm(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    *,
    rows: int = ROWS_DEFAULT,
) -> mx.array:
    """y = x @ ternary_W.T  for x [M, K]."""
    if x.ndim == 1:
        return ternary_gemv(x, weight, scales, rows=rows)
    orig = x.shape
    x2 = mx.reshape(x, (-1, orig[-1]))
    m, k = int(x2.shape[0]), int(x2.shape[1])
    n = int(weight.shape[0])
    if m == 1:
        y = ternary_gemv(x2, weight, scales, rows=rows)
        return mx.reshape(y, orig[:-1] + (n,))
    n_tg = (n + rows - 1) // rows
    y = _qmm_kernel(n, k, rows, x.dtype)(
        inputs=[x2, weight, scales],
        template=[("T", x.dtype), ("N", n), ("K", k), ("ROWS", rows)],
        grid=(n_tg * 32, m, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]
    return mx.reshape(y, orig[:-1] + (n,))


def mlx_affine_qmv(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """Reference Prism/MLX path: generic affine 2-bit quantized matmul."""
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=GROUP,
        bits=2,
    )
