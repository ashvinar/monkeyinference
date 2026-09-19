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
    // Clone of MLX qmv_fast: 64 threads / TG, 2 simdgroups × 4 rows.
    // 2-bit qdot: pre-shift x by 4^lane, mask-and-accumulate, no per-lane shifts.
    // Prism stores bias = -scale, so y = s * (codes·x - sum(x)) and we never
    // load the 0.42 GB bias tensor. x is already Hadamard-transformed.
    const uint lid = thread_index_in_threadgroup;
    const uint simd_gid = lid / 32u;
    const uint simd_lid = lid % 32u;
    const uint tg = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    const uint out_row = tg * 8u + simd_gid * 4u;
    const uint words_per_row = K / 16u;
    const uint groups_per_row = K / 128u;
    const device T* xrow = x + token * K;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint w_idx = simd_lid; w_idx < words_per_row; w_idx += 32u) {
        const uint kbase = w_idx * 16u;
        const uint group = w_idx / 8u;

        float xt0, xt1, xt2, xt3, xt4, xt5, xt6, xt7;
        float xt8, xt9, xt10, xt11, xt12, xt13, xt14, xt15;
        float xsum = 0.0f;
        {
            const float x0 = float(xrow[kbase + 0]);
            const float x1 = float(xrow[kbase + 1]);
            const float x2 = float(xrow[kbase + 2]);
            const float x3 = float(xrow[kbase + 3]);
            const float x4 = float(xrow[kbase + 4]);
            const float x5 = float(xrow[kbase + 5]);
            const float x6 = float(xrow[kbase + 6]);
            const float x7 = float(xrow[kbase + 7]);
            const float x8 = float(xrow[kbase + 8]);
            const float x9 = float(xrow[kbase + 9]);
            const float x10 = float(xrow[kbase + 10]);
            const float x11 = float(xrow[kbase + 11]);
            const float x12 = float(xrow[kbase + 12]);
            const float x13 = float(xrow[kbase + 13]);
            const float x14 = float(xrow[kbase + 14]);
            const float x15 = float(xrow[kbase + 15]);
            xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
            xt0 = x0;           xt1 = x1 * 0.25f;      xt2 = x2 * 0.0625f;    xt3 = x3 * 0.015625f;
            xt4 = x4;           xt5 = x5 * 0.25f;      xt6 = x6 * 0.0625f;    xt7 = x7 * 0.015625f;
            xt8 = x8;           xt9 = x9 * 0.25f;      xt10 = x10 * 0.0625f;  xt11 = x11 * 0.015625f;
            xt12 = x12;         xt13 = x13 * 0.25f;    xt14 = x14 * 0.0625f;  xt15 = x15 * 0.015625f;
        }

        #pragma unroll
        for (uint r = 0; r < 4u; r++) {
            const uint row = out_row + r;
            if (row >= N) {
                continue;
            }
            const uint word = w[row * words_per_row + w_idx];
            const float s = float(scales[row * groups_per_row + group]);
            const uint b0 = word & 0xffu;
            const uint b1 = (word >> 8u) & 0xffu;
            const uint b2 = (word >> 16u) & 0xffu;
            const uint b3 = (word >> 24u) & 0xffu;
            float accum = 0.0f;
            accum += xt0  * float(b0 & 0x03u);
            accum += xt1  * float(b0 & 0x0cu);
            accum += xt2  * float(b0 & 0x30u);
            accum += xt3  * float(b0 & 0xc0u);
            accum += xt4  * float(b1 & 0x03u);
            accum += xt5  * float(b1 & 0x0cu);
            accum += xt6  * float(b1 & 0x30u);
            accum += xt7  * float(b1 & 0xc0u);
            accum += xt8  * float(b2 & 0x03u);
            accum += xt9  * float(b2 & 0x0cu);
            accum += xt10 * float(b2 & 0x30u);
            accum += xt11 * float(b2 & 0xc0u);
            accum += xt12 * float(b3 & 0x03u);
            accum += xt13 * float(b3 & 0x0cu);
            accum += xt14 * float(b3 & 0x30u);
            accum += xt15 * float(b3 & 0xc0u);
            const float term = s * (accum - xsum);
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
        const uint ybase = token * N;
        if (out_row + 0 < N) y[ybase + out_row + 0] = T(acc0);
        if (out_row + 1 < N) y[ybase + out_row + 1] = T(acc1);
        if (out_row + 2 < N) y[ybase + out_row + 2] = T(acc2);
        if (out_row + 3 < N) y[ybase + out_row + 3] = T(acc3);
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
    x = mx.reshape(x, (1, -1))
    return mx.reshape(ternary_qmm(x, weight, scales, rows=rows), (int(weight.shape[0]),))


def ternary_qmm(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    *,
    rows: int = ROWS_DEFAULT,
) -> mx.array:
    """y = x @ ternary_W.T  for x [M, K].

    Decode (M=1) and small-M verify use the qdot qmv_fast clone (64-thread TGs).
    Larger prefill M still uses the token-parallel qmm kernel.
    """
    orig = x.shape
    x2 = mx.reshape(x, (-1, orig[-1])) if x.ndim != 1 else mx.reshape(x, (1, -1))
    m, k = int(x2.shape[0]), int(x2.shape[1])
    n = int(weight.shape[0])
    if k % GROUP:
        raise ValueError(f"K={k} is not a multiple of group size {GROUP}")
    if weight.shape[1] * PACK != k:
        raise ValueError(f"weight width {weight.shape[1]} does not pack K={k}")
    # qdot qmv_fast clone: M-parallel GEMVs. Prefer this under ~16 tokens;
    # above that MLX qmm (PackedLinear) is the production verify path.
    if m <= 16:
        n_tg = (n + 7) // 8
        y = _qmv_kernel(n, k, 8, x.dtype)(
            inputs=[x2, weight, scales],
            template=[("T", x.dtype), ("N", n), ("K", k), ("ROWS", 8)],
            grid=(n_tg * 64, m, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[x.dtype],
        )[0]
        if x.ndim == 1:
            return mx.reshape(y, (n,))
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
    """Prism/MLX affine 2-bit matmul.

    Always flatten to 2-D `[M, K]`. MLX's Metal dispatch uses
    `M = x.size / K` when `x` is row-contiguous 2-D, and switches from
    per-token `qmv` to a single weight-streaming `qmm` once `M` crosses
    `get_qmv_batch_limit` (about 6–10 on this 10-core M4 for Bonsai
    shapes). A 3-D `[1, K, H]` leftover+draft tensor is the verify path;
    leaving it 3-D used to look like a tiny batch of GEMVs.
    """
    orig = tuple(x.shape)
    k = int(orig[-1])
    x2 = mx.contiguous(mx.reshape(x, (-1, k)))
    y = mx.quantized_matmul(
        x2,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=GROUP,
        bits=2,
    )
    n = int(y.shape[-1])
    return mx.reshape(y, orig[:-1] + (n,))
