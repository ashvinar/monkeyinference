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


def linear_nk(linear) -> tuple[int, int]:
    """PackedLinear affine-2bit weight is uint32[N, K/16]."""
    n = int(linear.weight.shape[0])
    k = int(linear.weight.shape[1]) * PACK
    return n, k


def iter_packed_linears(model):
    """Yield (path, PackedLinear) for Bonsai text modules."""
    from monkeyinference.packed import PackedLinear

    lm = getattr(model, "lm_head", None)
    if isinstance(lm, PackedLinear):
        yield "lm_head", lm
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return
    names = (
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    for i, layer in enumerate(layers):
        for name in names:
            obj = layer
            ok = True
            for part in name.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and isinstance(obj, PackedLinear):
                yield f"layers.{i}.{name}", obj


# Shapes where five-trit GEMV beat 2-bit qdot by ≥5% on real Bonsai
# weights, interleaved. Calibrated 2026-09-19: **no winners**. Every unique
# (N,K) was 1.31–2.14× slower than qdot. The earlier synthetic 0.60–0.72×
# “wins” were qdot occupancy artifacts, not a format advantage. 2.02× on
# mlp_up is a LOSS (trit slower). Empty frozenset = all 2-bit, which is the
# per-shape max (cannot do worse than 2-bit).
TRIT_WIN_NK: frozenset[tuple[int, int]] = frozenset()
TRIT_WIN_MARGIN = 0.95  # trit must be < 95% of qdot wall
LM_HEAD_N_SKIP = 100_000  # do not pack 248320-row lm_head during calibrate


def calibrate_five_trit_wins(
    model,
    *,
    warmup: int = 4,
    iters: int = 12,
    margin: float = TRIT_WIN_MARGIN,
) -> dict:
    """Interleaved qdot vs five-trit on one real tensor per unique (N, K)."""
    import time

    from monkeyinference.kernels import ternary_qmm, ternary_trit_qmm

    rows = []
    seen: set[tuple[int, int]] = set()
    for path, lin in iter_packed_linears(model):
        nk = linear_nk(lin)
        if nk in seen:
            continue
        seen.add(nk)
        n, k = nk
        if n >= LM_HEAD_N_SKIP:
            rows.append(
                {
                    "path": path,
                    "n": n,
                    "k": k,
                    "skip": "lm_head_too_wide",
                    "win": False,
                }
            )
            continue
        x = mx.random.normal((k,)).astype(mx.float16)
        trit = pack_five_trit(lin.weight)
        mx.eval(x, trit, lin.weight, lin.scales)

        def qdot():
            return ternary_qmm(x, lin.weight, lin.scales)

        def trit_fn():
            return ternary_trit_qmm(x, trit, lin.scales)

        for _ in range(warmup):
            mx.eval(qdot())
            mx.eval(trit_fn())
        q_s, t_s = [], []
        for _ in range(iters):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(qdot())
            mx.synchronize()
            q_s.append(time.perf_counter() - t0)
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(trit_fn())
            mx.synchronize()
            t_s.append(time.perf_counter() - t0)
        q_us = 1e6 * sorted(q_s)[len(q_s) // 2]
        t_us = 1e6 * sorted(t_s)[len(t_s) // 2]
        ratio = t_us / q_us if q_us else None
        win = bool(ratio is not None and ratio < margin)
        rows.append(
            {
                "path": path,
                "n": n,
                "k": k,
                "qdot_us": q_us,
                "trit_us": t_us,
                "trit_over_qdot": ratio,
                "win": win,
            }
        )
        del trit
        mx.clear_cache()
    wins = frozenset((int(r["n"]), int(r["k"])) for r in rows if r.get("win"))
    return {"margin": margin, "wins": sorted(wins), "rows": rows}


def apply_mixed_five_trit(model, wins: frozenset[tuple[int, int]] | None = None) -> dict:
    """Enable five-trit GEMV only on shapes in `wins`. MMA stays 2-bit."""
    if wins is None:
        wins = TRIT_WIN_NK
    enabled = []
    skipped = []
    for path, lin in iter_packed_linears(model):
        nk = linear_nk(lin)
        if nk in wins:
            if lin.trit_weight is None:
                lin.enable_five_trit()
            enabled.append({"path": path, "n": nk[0], "k": nk[1]})
        else:
            lin.trit_weight = None
            skipped.append({"path": path, "n": nk[0], "k": nk[1]})
    return {
        "n_enabled": len(enabled),
        "n_skipped": len(skipped),
        "wins": sorted(wins),
        "enabled": enabled,
        "skipped": skipped,
    }
