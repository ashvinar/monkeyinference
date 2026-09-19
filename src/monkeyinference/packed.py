"""Drop-in replacements for nn.Linear / nn.Embedding on Prism ternary packs."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from monkeyinference.hadamard import fwht
from monkeyinference.kernels import (
    M8,
    VERIFY_ONCE_MAX,
    mlx_affine_qmv,
    ternary_qmm,
    ternary_qmm_m8,
    ternary_qmv_once,
    ternary_trit_qmm,
)
from monkeyinference import parity as parity_mod


class PackedLinear(nn.Module):
    """Ternary affine-2bit linear with optional input Hadamard.

    Decode (M=1) may use the custom qdot GEMV (`use_custom=True`), or
    five-trit if this (N,K) won the real-weight gate. Speculative verify
    (M=2..8) uses the 8-row MMA on 2-bit weights. Prefill (M>16) uses
    MLX quantized matmul.
    """

    def __init__(
        self,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
        signs: mx.array | None,
        block: int,
        *,
        dtype=mx.float16,
        use_custom: bool = True,
    ):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.signs = signs
        self.block = int(block or 0)
        self.dtype = dtype
        self.use_custom = use_custom
        self.trit_weight = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            # Graph-fused with the matmul: 10 KB activation, not a second weight stream.
            # A 1024-point FWHT cannot live inside the 64-thread output-tiled qmv
            # without recomputing H(x) once per 8 output rows.
            x = fwht(x, self.signs, inverse=False)
        rows = int(x.size // x.shape[-1]) if x.size else 0
        # M=1 decode: custom qdot GEMV.
        # 2..8: 8-row MMA (pad if needed). 9..16: weight-once qdot.
        # Prefill M>16: MLX qmm.
        if self.use_custom and rows <= 1 and self.trit_weight is not None:
            y = ternary_trit_qmm(x, self.trit_weight, self.scales).astype(self.dtype)
            if parity_mod.PARITY_REMAINING > 0:
                ref = mlx_affine_qmv(x, self.weight, self.scales, self.biases).astype(
                    mx.float32
                )
                diff = y.astype(mx.float32) - ref
                mx.eval(diff)
                absd = mx.abs(diff)
                parity_mod.record_parity(
                    float(mx.max(absd).item()),
                    float(mx.sqrt(mx.mean(diff * diff)).item()),
                    tuple(int(s) for s in y.shape),
                )
            return y
        if self.use_custom and rows <= 1:
            y = ternary_qmm(x, self.weight, self.scales).astype(self.dtype)
            if parity_mod.PARITY_REMAINING > 0:
                ref = mlx_affine_qmv(x, self.weight, self.scales, self.biases).astype(
                    mx.float32
                )
                diff = y.astype(mx.float32) - ref
                mx.eval(diff)
                absd = mx.abs(diff)
                parity_mod.record_parity(
                    float(mx.max(absd).item()),
                    float(mx.sqrt(mx.mean(diff * diff)).item()),
                    tuple(int(s) for s in y.shape),
                )
            return y
        if self.use_custom and 1 < rows <= M8:
            y = ternary_qmm_m8(x, self.weight, self.scales).astype(self.dtype)
            if parity_mod.PARITY_REMAINING > 0:
                ref = mlx_affine_qmv(x, self.weight, self.scales, self.biases).astype(
                    mx.float32
                )
                diff = y.astype(mx.float32) - ref
                mx.eval(diff)
                absd = mx.abs(diff)
                parity_mod.record_parity(
                    float(mx.max(absd).item()),
                    float(mx.sqrt(mx.mean(diff * diff)).item()),
                    tuple(int(s) for s in y.shape),
                )
            return y
        if 1 < rows <= VERIFY_ONCE_MAX:
            y = ternary_qmv_once(x, self.weight, self.scales).astype(self.dtype)
            if parity_mod.PARITY_REMAINING > 0:
                ref = mlx_affine_qmv(x, self.weight, self.scales, self.biases).astype(
                    mx.float32
                )
                diff = y.astype(mx.float32) - ref
                mx.eval(diff)
                absd = mx.abs(diff)
                parity_mod.record_parity(
                    float(mx.max(absd).item()),
                    float(mx.sqrt(mx.mean(diff * diff)).item()),
                    tuple(int(s) for s in y.shape),
                )
            return y
        y = mlx_affine_qmv(x, self.weight, self.scales, self.biases)
        return y.astype(self.dtype)

    def enable_five_trit(self) -> None:
        """Lossless 5-trit recode of this linear. Keeps 2-bit weights for M>1 MMA."""
        from monkeyinference.trit import pack_five_trit

        self.trit_weight = pack_five_trit(self.weight)
        mx.eval(self.trit_weight)


class PackedEmbedding(nn.Module):
    """Token embedding: dequant lookup + inverse Hadamard."""

    def __init__(
        self,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
        signs: mx.array | None,
        block: int,
        *,
        dtype=mx.float16,
    ):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.signs = signs
        self.block = int(block or 0)
        self.dtype = dtype

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        indices = x.reshape(-1)
        out = (
            mx.dequantize(
                self.weight[indices],
                self.scales[indices],
                self.biases[indices],
                group_size=128,
                bits=2,
            )
            .reshape(*shape, -1)
            .astype(self.dtype)
        )
        if self.block:
            out = fwht(out, self.signs, inverse=True)
        return out

    def as_linear(self, x: mx.array) -> mx.array:
        """Tied-head path (not used by Bonsai; kept for interface completeness)."""
        if self.block:
            x = fwht(x, self.signs, inverse=False)
        return mlx_affine_qmv(x, self.weight, self.scales, self.biases)
