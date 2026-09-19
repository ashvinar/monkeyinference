"""Activation Hadamard matching Prism's bundled MLX runtime.

Contract (hadamard.json + runtime.py::fwht):
  - normalized Sylvester/Walsh Hadamard, block 1024
  - forward: x := H (x ⊙ signs) / sqrt(block)
  - inverse (embeddings): x := (H x) ⊙ signs / sqrt(block)
"""

from __future__ import annotations

import math

import mlx.core as mx

BLOCK = 1024
HADAMARD_SCALE = 1.0 / math.sqrt(BLOCK)


def fwht(x: mx.array, signs: mx.array | None, *, inverse: bool = False) -> mx.array:
    """Apply the Prism activation Hadamard along the last axis."""
    shape, dtype = x.shape, x.dtype
    width = shape[-1]
    if width % BLOCK:
        raise ValueError(f"Hadamard block {BLOCK} does not divide activation width {width}")
    y = x.astype(mx.float32)
    if signs is not None and not inverse:
        y = y * signs
    y = mx.hadamard_transform(y.reshape(-1, BLOCK), scale=HADAMARD_SCALE).reshape(shape)
    if signs is not None and inverse:
        y = y * signs
    return y.astype(dtype)
