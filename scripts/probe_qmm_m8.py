"""Gate: ternary_qmm_m8 T=8 / T=1 <= 1.3x on MLP-up 17408x5120.

Pipelined timing (warmup, then N evals under one sync). T=1 is the existing
qdot GEMV; T=8 is the 256-thread dequant-in-TG MMA. LPM should be off.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from monkeyinference.kernels import (
    mlx_affine_qmv,
    ternary_gemv,
    ternary_qmm_m8,
    ternary_qmv_once,
)
from monkeyinference.roofline import pack_ternary

OUT = Path("results/qmm_m8.json")
STREAM_READ_GBS = 92.0  # 512 MiB median; 1 GiB was 85


def _lpm() -> int:
    try:
        out = subprocess.check_output(["pmset", "-g"], text=True)
    except Exception:
        return -1
    for line in out.splitlines():
        if "lowpowermode" in line.lower():
            return int(line.split()[-1])
    return -1


def _sync():
    mx.synchronize()


def _pipelined(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    _sync()
    return (time.perf_counter() - t0) / iters


def bench_shape(n: int, k: int, *, warmup: int = 5, iters: int = 20) -> dict:
    x1, w, scales, biases = pack_ternary(n, k, seed=n + k)
    rng = np.random.default_rng(n + k + 1)
    x8 = mx.array(rng.standard_normal((8, k)).astype(np.float16))
    mx.eval(x1, x8, w, scales, biases)

    us_t1 = _pipelined(lambda: ternary_gemv(x1, w, scales), warmup, iters) * 1e6
    us_m8 = _pipelined(lambda: ternary_qmm_m8(x8, w, scales), warmup, iters) * 1e6
    us_once = _pipelined(lambda: ternary_qmv_once(x8, w, scales), warmup, iters) * 1e6
    us_mlx8 = _pipelined(lambda: mlx_affine_qmv(x8, w, scales, biases), warmup, iters) * 1e6

    y1 = ternary_gemv(x8[0], w, scales)
    y8 = ternary_qmm_m8(x8, w, scales)
    yref = mlx_affine_qmv(x8, w, scales, biases)
    mx.eval(y1, y8, yref)
    err_vs_gemv = float(mx.max(mx.abs(y8[0] - y1)).item())
    err_vs_mlx = float(
        mx.max(mx.abs(y8.astype(mx.float32) - yref.astype(mx.float32))).item()
    )

    weight_bytes = int(w.nbytes) + int(scales.nbytes)
    x8_bytes = 8 * k * 2
    y8_bytes = 8 * n * 2
    bytes_once = weight_bytes + x8_bytes + y8_bytes
    return {
        "n": n,
        "k": k,
        "us_t1_qdot": us_t1,
        "us_t8_m8": us_m8,
        "us_t8_qmv_once": us_once,
        "us_t8_mlx": us_mlx8,
        "t8_over_t1": us_m8 / us_t1 if us_t1 else None,
        "gate_1_3x": (us_m8 / us_t1) <= 1.3 if us_t1 else None,
        "gbs_t1": weight_bytes / (us_t1 * 1e-6) / 1e9,
        "gbs_t8_if_once": bytes_once / (us_m8 * 1e-6) / 1e9,
        "stream_us_if_once": 1e6 * bytes_once / (STREAM_READ_GBS * 1e9),
        "err_vs_gemv_row0": err_vs_gemv,
        "err_vs_mlx": err_vs_mlx,
        "weight_bytes": weight_bytes,
    }


def main() -> None:
    lpm = _lpm()
    rows = []
    for name, n, k in [
        ("mlp_up", 17408, 5120),
        ("mlp_down", 5120, 17408),
        ("gdn_qkv", 10240, 5120),
        ("attn_q", 6144, 5120),
    ]:
        print(f"bench {name} {n}x{k} ...", flush=True)
        rec = bench_shape(n, k)
        rec["name"] = name
        rows.append(rec)
        print(
            f"  T1 {rec['us_t1_qdot']:.0f} us  T8-m8 {rec['us_t8_m8']:.0f} us  "
            f"ratio {rec['t8_over_t1']:.2f}  gate={rec['gate_1_3x']}  "
            f"once {rec['us_t8_qmv_once']:.0f} us  mlx8 {rec['us_t8_mlx']:.0f} us  "
            f"err {rec['err_vs_mlx']:.4f}",
            flush=True,
        )
    report = {
        "lpm": lpm,
        "gate": "T=8 / T=1 <= 1.3 on mlp_up 17408x5120",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
