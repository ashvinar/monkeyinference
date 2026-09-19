"""Five-trit packing: 5 ternary codes per byte, g128 group padded to 130.

Histogram on Bonsai affine-2bit is {0,1,2} only (0 of 3.54e9 code-3), so
`3^5 = 243 < 256` is a lossless re-encoding. Each g128 group is stored as
26 bytes (128 codes + 2 pad-ones). Pad is code 1 (exact zero) so dummy
trits do not contribute. Scales stay g128.

Bytes vs 2-bit: 26/32 = 0.8125 (−18.75%). Codes 6.822 GB → 5.543 GB;
stream with scales+signs ≈ 5.975 GB.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from monkeyinference.kernels import (
    BYTES_PER_GROUP,
    GROUP,
    PAD_TRITS_PER_GROUP,
    PACK,
    TRITS_PER_BYTE,
    ternary_trit_qmm,
)

POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint32)


def bytes_per_row(k: int) -> int:
    if k % GROUP:
        raise ValueError(f"K={k} is not a multiple of group size {GROUP}")
    return (k // GROUP) * BYTES_PER_GROUP


def unpack_affine_2bit(weight: np.ndarray) -> np.ndarray:
    """uint32[N, K/16] → uint8[N, K] codes in {0,1,2,3}."""
    n, words = weight.shape
    lanes = np.arange(PACK, dtype=np.uint32)
    codes = (weight.reshape(n, words, 1) >> (2 * lanes)) & 3
    return codes.reshape(n, words * PACK).astype(np.uint8)


def pack_five_trit(weight_u32: mx.array | np.ndarray) -> mx.array:
    """Lossless 2-bit affine → 5-trit uint8. Pad each g128 with two code-1s."""
    w = np.array(weight_u32)
    codes = unpack_affine_2bit(w)
    n, k = codes.shape
    if int(codes.max()) > 2:
        raise ValueError("five-trit pack is not lossless: found code 3")
    ng = k // GROUP
    grouped = codes.reshape(n, ng, GROUP)
    pad = np.ones((n, ng, PAD_TRITS_PER_GROUP), dtype=np.uint8)
    c = np.concatenate([grouped, pad], axis=-1)
    c = c.reshape(n, ng, BYTES_PER_GROUP, TRITS_PER_BYTE)
    packed = np.matmul(c.astype(np.uint16), POW3.astype(np.uint16)).astype(np.uint8)
    return mx.array(packed.reshape(n, ng * BYTES_PER_GROUP))


def unpack_five_trit(packed: mx.array | np.ndarray, k: int) -> np.ndarray:
    """uint8[N, bytes] → uint8[N, K] (drops the two pad trits per group)."""
    p = np.array(packed)
    n, nbytes = p.shape
    ng = k // GROUP
    if nbytes != ng * BYTES_PER_GROUP:
        raise ValueError(f"packed width {nbytes} != {ng} groups × {BYTES_PER_GROUP}")
    b = p.reshape(n, ng, BYTES_PER_GROUP).astype(np.uint16)
    codes = np.empty((n, ng, BYTES_PER_GROUP, TRITS_PER_BYTE), dtype=np.uint8)
    for t in range(TRITS_PER_BYTE):
        codes[:, :, :, t] = (b // POW3[t]) % 3
        # POW3[t] is 3^t; successive div. Vectorized: already (b // 3^t) % 3.
    flat = codes.reshape(n, ng, GROUP + PAD_TRITS_PER_GROUP)
    return flat[:, :, :GROUP].reshape(n, k)


def trit_weight_bytes(n: int, k: int) -> int:
    return n * bytes_per_row(k)


def five_trit_qmm(x: mx.array, trit_w: mx.array, scales: mx.array) -> mx.array:
    """y = x @ five-trit_W.T. Greedy M=1 path; same (code-1)*s contract as qdot."""
    return ternary_trit_qmm(x, trit_w, scales)
