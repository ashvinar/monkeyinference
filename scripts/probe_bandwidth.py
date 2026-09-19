"""GPU copy / read / write STREAM, plus CPU P-core read and CPU+GPU aggregate.

Decode is read-dominated (7.25 GB weights in, tiny activation out). Copy
STREAM (86.6 GB/s) counts read+write; this script reports them separately.
Does not start a hybrid engine — measurement only.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from monkeyinference.kernels import stream_copy, stream_read_reduce, stream_write

OUT = Path("results/bandwidth.json")
STREAM_BYTES = 512 * 1024 * 1024
N_F32 = STREAM_BYTES // 4
CPU_THREADS = 4  # M4 P-cores
SRC = Path(__file__).resolve().parents[1] / "src" / "monkeyinference" / "stream_cpu.c"
DYLIB = Path("/tmp/monkey_stream_cpu.dylib")


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


def _compile_cpu():
    subprocess.check_call(
        [
            "clang",
            "-O3",
            "-pthread",
            "-shared",
            "-fPIC",
            "-o",
            str(DYLIB),
            str(SRC),
        ]
    )
    lib = ctypes.CDLL(str(DYLIB))
    lib.cpu_touch.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.cpu_read_spawn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cpu_read_spawn.restype = ctypes.c_int
    lib.cpu_read_join.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.cpu_read_join.restype = ctypes.c_uint64
    return lib


class _Worker(ctypes.Structure):
    _fields_ = [
        ("p", ctypes.POINTER(ctypes.c_float)),
        ("n", ctypes.c_size_t),
        ("go", ctypes.POINTER(ctypes.c_int)),
        ("run", ctypes.POINTER(ctypes.c_int)),
        ("bytes", ctypes.c_uint64),
        ("sink", ctypes.c_float),
        ("pad", ctypes.c_int),
    ]


def _cpu_buf(n: int, lib) -> np.ndarray:
    buf = np.empty(n, dtype=np.float32)
    lib.cpu_touch(
        buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(n),
    )
    return buf


def bench_gpu_copy(nbytes=STREAM_BYTES, warmup=5, iters=20) -> dict:
    n = nbytes // 4
    inp = mx.ones((n,), dtype=mx.float32)
    mx.eval(inp)
    for _ in range(warmup):
        mx.eval(stream_copy(inp))
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(stream_copy(inp))
    _sync()
    elapsed = time.perf_counter() - t0
    # copy: read nbytes + write nbytes
    return {
        "kind": "gpu_copy_read_plus_write",
        "bytes_per_iter": 2 * nbytes,
        "iters": iters,
        "elapsed_s": elapsed,
        "gbs": (2.0 * nbytes * iters) / elapsed / 1e9,
    }


def bench_gpu_read(nbytes=STREAM_BYTES, warmup=5, iters=20) -> dict:
    n = nbytes // 4
    inp = mx.ones((n,), dtype=mx.float32)
    mx.eval(inp)
    for _ in range(warmup):
        mx.eval(stream_read_reduce(inp))
    _sync()
    t0 = time.perf_counter()
    sink = None
    for _ in range(iters):
        sink = stream_read_reduce(inp)
        mx.eval(sink)
    _sync()
    elapsed = time.perf_counter() - t0
    # force the reduction to exist
    _ = float(mx.sum(sink).item())
    return {
        "kind": "gpu_read_reduce",
        "bytes_per_iter": nbytes,
        "iters": iters,
        "elapsed_s": elapsed,
        "gbs": (nbytes * iters) / elapsed / 1e9,
        "sink_tg_sum": float(mx.sum(sink).item()),
    }


def bench_gpu_write(nbytes=STREAM_BYTES, warmup=5, iters=20) -> dict:
    n = nbytes // 4
    for _ in range(warmup):
        mx.eval(stream_write(n))
    _sync()
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = stream_write(n)
        mx.eval(out)
    _sync()
    elapsed = time.perf_counter() - t0
    _ = float(out[0].item())
    return {
        "kind": "gpu_write_fill",
        "bytes_per_iter": nbytes,
        "iters": iters,
        "elapsed_s": elapsed,
        "gbs": (nbytes * iters) / elapsed / 1e9,
    }


def _spawn_cpu(lib, buf: np.ndarray, nthreads: int):
    go = ctypes.c_int(0)
    run = ctypes.c_int(1)
    ws = (_Worker * nthreads)()
    ths = (ctypes.c_void_p * nthreads)()
    rc = lib.cpu_read_spawn(
        buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(buf.size),
        ctypes.c_int(nthreads),
        ctypes.byref(go),
        ctypes.byref(run),
        ctypes.cast(ws, ctypes.c_void_p),
        ctypes.cast(ths, ctypes.c_void_p),
    )
    if rc != 0:
        raise RuntimeError("cpu_read_spawn failed")
    return go, run, ws, ths


def bench_cpu_read(lib, nbytes=STREAM_BYTES, seconds=2.0, nthreads=CPU_THREADS) -> dict:
    n = nbytes // 4
    buf = _cpu_buf(n, lib)
    go, run, ws, ths = _spawn_cpu(lib, buf, nthreads)
    t0 = time.perf_counter()
    go.value = 1
    time.sleep(seconds)
    run.value = 0
    total = int(lib.cpu_read_join(ctypes.cast(ws, ctypes.c_void_p), ctypes.cast(ths, ctypes.c_void_p), nthreads))
    elapsed = time.perf_counter() - t0
    return {
        "kind": "cpu_pcore_read",
        "threads": nthreads,
        "bytes": total,
        "elapsed_s": elapsed,
        "gbs": total / elapsed / 1e9,
        "qos": "USER_INTERACTIVE",
    }


def bench_hybrid(lib, nbytes=STREAM_BYTES, gpu_iters=40, nthreads=CPU_THREADS) -> dict:
    """GPU read-reduce and CPU P-core read on disjoint 512 MiB buffers."""
    n = nbytes // 4
    gpu_inp = mx.ones((n,), dtype=mx.float32)
    mx.eval(gpu_inp)
    cpu_buf = _cpu_buf(n, lib)
    for _ in range(3):
        mx.eval(stream_read_reduce(gpu_inp))
    _sync()

    go, run, ws, ths = _spawn_cpu(lib, cpu_buf, nthreads)
    _sync()
    t0 = time.perf_counter()
    go.value = 1
    sink = None
    for _ in range(gpu_iters):
        sink = stream_read_reduce(gpu_inp)
        mx.eval(sink)
    _sync()
    t_gpu = time.perf_counter()
    run.value = 0
    cpu_bytes = int(
        lib.cpu_read_join(ctypes.cast(ws, ctypes.c_void_p), ctypes.cast(ths, ctypes.c_void_p), nthreads)
    )
    t1 = time.perf_counter()
    _ = float(mx.sum(sink).item())
    gpu_elapsed = t_gpu - t0
    wall = t1 - t0
    gpu_bytes = nbytes * gpu_iters
    return {
        "kind": "hybrid_disjoint_read",
        "gpu_iters": gpu_iters,
        "cpu_threads": nthreads,
        "gpu_elapsed_s": gpu_elapsed,
        "wall_s": wall,
        "gpu_bytes": gpu_bytes,
        "cpu_bytes": cpu_bytes,
        "gpu_gbs": gpu_bytes / gpu_elapsed / 1e9,
        "cpu_gbs": cpu_bytes / wall / 1e9,
        "aggregate_gbs": (gpu_bytes + cpu_bytes) / wall / 1e9,
        "note": "CPU bytes counted until join; GPU elapsed is synchronize-to-synchronize.",
    }


def main():
    lpm = _lpm()
    print(f"lowpowermode={lpm}  buffer={STREAM_BYTES/2**20:.0f} MiB")
    lib = _compile_cpu()
    copy = bench_gpu_copy()
    read = bench_gpu_read()
    write = bench_gpu_write()
    cpu = bench_cpu_read(lib)
    hybrid = bench_hybrid(lib)
    live = 7.254e9
    with_bias = 7.674e9
    report = {
        "lowpowermode": lpm,
        "published_dram_gbs": 120.0,
        "buffer_bytes": STREAM_BYTES,
        "gpu_copy": copy,
        "gpu_read": read,
        "gpu_write": write,
        "cpu_read": cpu,
        "hybrid": hybrid,
        "decode_resembles": "gpu_read",
        "roofline_tok_s": {
            "copy_with_bias": copy["gbs"] * 1e9 / with_bias,
            "copy_no_bias": copy["gbs"] * 1e9 / live,
            "read_with_bias": read["gbs"] * 1e9 / with_bias,
            "read_no_bias": read["gbs"] * 1e9 / live,
            "published_with_bias": 120e9 / with_bias,
            "published_no_bias": 120e9 / live,
            "hybrid_read_no_bias": hybrid["aggregate_gbs"] * 1e9 / live,
        },
        "greedy_tps": 10.22,
    }
    r = report["roofline_tok_s"]
    print(
        f"GPU copy  {copy['gbs']:6.1f} GB/s  (read+write, 2x bytes)\n"
        f"GPU read  {read['gbs']:6.1f} GB/s  (reduce, decode-like)\n"
        f"GPU write {write['gbs']:6.1f} GB/s\n"
        f"CPU read  {cpu['gbs']:6.1f} GB/s  ({CPU_THREADS} P-core threads)\n"
        f"Hybrid    GPU {hybrid['gpu_gbs']:.1f} + CPU {hybrid['cpu_gbs']:.1f} "
        f"= {hybrid['aggregate_gbs']:.1f} GB/s aggregate"
    )
    print(
        f"decode ceiling no-bias: copy {r['copy_no_bias']:.1f}  "
        f"read {r['read_no_bias']:.1f}  published {r['published_no_bias']:.1f}  "
        f"hybrid {r['hybrid_read_no_bias']:.1f} tok/s"
    )
    print(f"10.22 / read-no-bias = {10.22 / r['read_no_bias']:.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
