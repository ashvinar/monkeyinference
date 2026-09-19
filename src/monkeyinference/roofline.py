"""Roofline: measure STREAM bandwidth and ternary GEMV efficiency on this GPU."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from monkeyinference.kernels import GROUP, mlx_affine_qmv, stream_copy, ternary_gemv

APPLE_M4_AIR_SPEC_GBS = 120.0
LANGUAGE_WEIGHT_BYTES = 7.674e9  # Bonsai language tensors, including redundant biases
LANGUAGE_WEIGHT_BYTES_NO_BIAS = 7.254e9  # drop 0.420 GB biases
STARTING_DECODE_TPS = 8.0
STARTING_PREFILL_TPS = 40.0


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return ""


def machine_info() -> dict:
    return {
        "platform": platform.platform(),
        "chip": _sysctl("machdep.cpu.brand_string") or "Apple silicon",
        "mem_bytes": int(_sysctl("hw.memsize") or 0),
        "ncpu": int(_sysctl("hw.ncpu") or 0),
        "published_dram_gbs": APPLE_M4_AIR_SPEC_GBS,
    }


def _sync():
    mx.synchronize()


def bench_stream(nbytes: int = 512 * 1024 * 1024, warmup: int = 5, iters: int = 20) -> dict:
    """Copy `nbytes` of float32 through a Metal kernel; report GB/s."""
    n = nbytes // 4
    inp = mx.ones((n,), dtype=mx.float32)
    mx.eval(inp)
    for _ in range(warmup):
        out = stream_copy(inp)
        mx.eval(out)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = stream_copy(inp)
        mx.eval(out)
    _sync()
    elapsed = time.perf_counter() - t0
    # copy reads nbytes and writes nbytes
    gbs = (2.0 * nbytes * iters) / elapsed / 1e9
    return {
        "bytes": nbytes,
        "iters": iters,
        "elapsed_s": elapsed,
        "gbs": gbs,
        "kind": "metal_copy_read_plus_write",
    }


def pack_ternary(n: int, k: int, seed: int = 0) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, size=(n, k), dtype=np.uint32)
    scales_np = rng.standard_normal((n, k // GROUP)).astype(np.float16)
    shifts = (2 * np.arange(16, dtype=np.uint32)).reshape(1, 1, 16)
    words = np.bitwise_or.reduce(codes.reshape(n, k // 16, 16) << shifts, axis=-1).astype(np.uint32)
    biases_np = (-scales_np.astype(np.float32)).astype(np.float16)
    w = mx.array(words)
    scales = mx.array(scales_np)
    biases = mx.array(biases_np)
    x = mx.array(rng.standard_normal((k,)).astype(np.float16))
    mx.eval(w, scales, biases, x)
    return x, w, scales, biases


def bench_qmv(n: int, k: int, warmup: int = 5, iters: int = 20, custom: bool = True) -> dict:
    x, w, scales, biases = pack_ternary(n, k)
    fn = (lambda: ternary_gemv(x, w, scales)) if custom else (lambda: mlx_affine_qmv(x, w, scales, biases))
    for _ in range(warmup):
        mx.eval(fn())
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    _sync()
    elapsed = time.perf_counter() - t0
    weight_bytes = int(w.nbytes) + int(scales.nbytes)
    if not custom:
        weight_bytes += int(biases.nbytes)
    # GEMV reads weights+scales (+x, write y — negligible vs weights for these shapes)
    gbs = (weight_bytes * iters) / elapsed / 1e9
    tps_if_this_were_whole_model = None
    return {
        "n": n,
        "k": k,
        "custom": custom,
        "elapsed_s": elapsed,
        "iters": iters,
        "weight_bytes": weight_bytes,
        "gbs": gbs,
        "us_per_call": elapsed / iters * 1e6,
    }


def roofline(stream_gbs: float) -> dict:
    """Decode tokens/s if every language byte is read once per token at `stream_gbs`."""
    spec = APPLE_M4_AIR_SPEC_GBS
    measured = stream_gbs
    return {
        "published_dram_gbs": spec,
        "measured_stream_gbs": measured,
        "stream_vs_published": measured / spec,
        "decode_tps_at_published_with_bias": spec * 1e9 / LANGUAGE_WEIGHT_BYTES,
        "decode_tps_at_published_no_bias": spec * 1e9 / LANGUAGE_WEIGHT_BYTES_NO_BIAS,
        "decode_tps_at_measured_with_bias": measured * 1e9 / LANGUAGE_WEIGHT_BYTES,
        "decode_tps_at_measured_no_bias": measured * 1e9 / LANGUAGE_WEIGHT_BYTES_NO_BIAS,
        "starting_decode_tps": STARTING_DECODE_TPS,
        "starting_prefill_tps": STARTING_PREFILL_TPS,
        "starting_fraction_of_published": STARTING_DECODE_TPS / (spec * 1e9 / LANGUAGE_WEIGHT_BYTES),
        "starting_fraction_of_measured": STARTING_DECODE_TPS / (measured * 1e9 / LANGUAGE_WEIGHT_BYTES)
        if measured
        else None,
        "ruling": (
            "Decode at 27B / ~2.13–2.25 bpw is DRAM-bandwidth bound. A perfect "
            "ternary GEMV cannot beat measured_stream_gbs / language_bytes. "
            "Going above that requires speculative decode (multiple accepted "
            "tokens per target weight pass), which is why Splash's 74 tok/s is "
            "not a kernel-only number."
        ),
    }


def run(out: Path | None = None) -> dict:
    report = {"machine": machine_info()}
    report["stream"] = bench_stream()
    # Shapes from Bonsai: hidden 5120, intermediate 17408, vocab 248320
    shapes = [
        ("mlp_up_like", 17408, 5120),
        ("mlp_down_like", 5120, 17408),
        ("attn_q_like", 6144, 5120),  # 24 * 256
        ("lm_head_slice", 8192, 5120),
    ]
    qmv = []
    for name, n, k in shapes:
        custom = bench_qmv(n, k, custom=True)
        ref = bench_qmv(n, k, custom=False)
        custom["name"] = name
        ref["name"] = name
        qmv.append({"custom": custom, "mlx_affine": ref, "speedup": custom["gbs"] / ref["gbs"] if ref["gbs"] else None})
    report["qmv"] = qmv
    report["roofline"] = roofline(report["stream"]["gbs"])
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rep = run(args.out)
    print(json.dumps({
        "stream_gbs": rep["stream"]["gbs"],
        "roofline": rep["roofline"],
        "qmv_speedups": [
            {"name": r["custom"]["name"], "custom_gbs": r["custom"]["gbs"], "mlx_gbs": r["mlx_affine"]["gbs"], "speedup": r["speedup"]}
            for r in rep["qmv"]
        ],
    }, indent=2))
