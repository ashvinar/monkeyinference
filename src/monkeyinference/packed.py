"""Drop-in replacements for nn.Linear / nn.Embedding on Prism ternary packs."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from monkeyinference.hadamard import fwht
from monkeyinference.kernels import mlx_affine_qmv, ternary_qmm
from monkeyinference import parity as parity_mod


class PackedLinear(nn.Module):
    """Ternary affine-2bit linear with optional input Hadamard.

    Decode (M=1) may use the custom qdot GEMV (`use_custom=True`).
    Prefill and speculative verify (M>1) always use MLX quantized matmul
    so one weight stream covers every draft token.
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
        use_custom: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.signs = signs
        self.block = int(block or 0)
        self.dtype = dtype
        self.use_custom = use_custom

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            # Graph-fused with the matmul: 10 KB activation, not a second weight stream.
            # A 1024-point FWHT cannot live inside the 64-thread output-tiled qmv
            # without recomputing H(x) once per 8 output rows.
            x = fwht(x, self.signs, inverse=False)
        rows = int(x.size // x.shape[-1]) if x.size else 0
        # Custom qdot GEMV is the decode (M=1) kernel. Verify (M>1) stays on
        # MLX qmm so one target pass streams weights once for all draft tokens.
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
        y = mlx_affine_qmv(x, self.weight, self.scales, self.biases)
        return y.astype(self.dtype)


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
