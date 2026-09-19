"""Five-trit unpack vs 2-bit qdot. No 27B load.

If census-weighted speedup would put greedy at ≥ ~11.5 tok/s, the next
step is an e2e convert (drop 2-bit, greedy only). Otherwise keep qdot.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx

from monkeyinference.kernels import ternary_gemv, ternary_trit_qmm
from monkeyinference.roofline import pack_ternary
from monkeyinference.trit import pack_five_trit, trit_weight_bytes

OUT = Path("results/five_trit.json")
GREEDY_TPS = 10.22
TARGET_TPS = 11.5

# Bonsai linear census (count of PackedLinear calls per decode token).
SHAPES = [
    ("mlp_up", 17408, 5120, 64),
    ("mlp_down", 5120, 17408, 64),
    ("gdn_qkv", 10240, 5120, 48),
    ("gdn_z", 6144, 5120, 48),
    ("gdn_out", 5120, 6144, 48),
    ("attn_q", 6144, 5120, 16),
    ("attn_k", 1024, 5120, 16),
    ("attn_v", 1024, 5120, 16),
    ("attn_o", 5120, 6144, 16),
    # Full lm_head is 248320×5120; pack_ternary would materialize 5 GB of
    # uint32 codes. Time an 8192-row slice and scale; the kernel grids over N.
    ("lm_head_slice", 8192, 5120, 1),
]
LM_HEAD_N = 248320
LM_HEAD_SLICE_N = 8192


def _bench(fn, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    rows = []
    pred_qdot_ms = 0.0
    pred_trit_ms = 0.0
    for name, n, k, count in SHAPES:
        print(f"=== {name} {n}×{k} ×{count} ===", flush=True)
        x, w, scales, biases = pack_ternary(n, k, seed=n + k)
        trit = pack_five_trit(w)
        mx.eval(trit)
        # Correctness on this shape (lm_head included).
        got = ternary_trit_qmm(x, trit, scales)
        ref = ternary_gemv(x, w, scales)
        mx.eval(got, ref)
        err = float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item())
        qdot_s = _bench(lambda: ternary_gemv(x, w, scales))
        trit_s = _bench(lambda: ternary_trit_qmm(x, trit, scales))
        two_bit_bytes = int(w.nbytes) + int(scales.nbytes)
        trit_bytes = int(trit.nbytes) + int(scales.nbytes)
        row = {
            "name": name,
            "n": n,
            "k": k,
            "count": count,
            "qdot_us": qdot_s * 1e6,
            "trit_us": trit_s * 1e6,
            "trit_over_qdot": trit_s / qdot_s,
            "max_abs_vs_qdot": err,
            "two_bit_bytes": two_bit_bytes,
            "trit_bytes": trit_bytes,
            "byte_ratio": trit_bytes / two_bit_bytes,
            "qdot_gbs": two_bit_bytes / qdot_s / 1e9,
            "trit_gbs": trit_bytes / trit_s / 1e9,
        }
        scale = (LM_HEAD_N / LM_HEAD_SLICE_N) if name == "lm_head_slice" else 1.0
        row["n_scale"] = scale
        row["qdot_us_scaled"] = row["qdot_us"] * scale
        row["trit_us_scaled"] = row["trit_us"] * scale
        rows.append(row)
        pred_qdot_ms += count * qdot_s * 1000 * scale
        pred_trit_ms += count * trit_s * 1000 * scale
        print(
            f"  qdot {row['qdot_us']:.0f} us  trit {row['trit_us']:.0f} us  "
            f"{row['trit_over_qdot']:.2f}×  err {err:.4f}  scale {scale:.1f}",
            flush=True,
        )

    speedup = pred_qdot_ms / pred_trit_ms if pred_trit_ms else None
    pred_greedy = GREEDY_TPS * speedup if speedup else None
    ship = bool(pred_greedy is not None and pred_greedy >= TARGET_TPS and all(r["max_abs_vs_qdot"] < 0.05 for r in rows))
    report = {
        "greedy_tps": GREEDY_TPS,
        "target_tps": TARGET_TPS,
        "census_qdot_ms": pred_qdot_ms,
        "census_trit_ms": pred_trit_ms,
        "census_speedup": speedup,
        "predicted_greedy_tok_s": pred_greedy,
        "ship": ship,
        "read": (
            "SHIP: census-weighted five-trit would put greedy at ≥ 11.5. "
            "Convert at load, drop 2-bit, greedy-only (MMA stays on 2-bit layout)."
            if ship
            else "HOLD: unpack tax eats the 18.75% byte cut, or correctness failed. "
            "Keep 2-bit qdot. Five-trit is not the 11.5–13 lever on this GPU until "
            "the kernel is faster than qdot."
        ),
        "rows": rows,
        "trit_weight_mlp_up_bytes": trit_weight_bytes(17408, 5120),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("census_speedup", "predicted_greedy_tok_s", "ship", "read")}, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
