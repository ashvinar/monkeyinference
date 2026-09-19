"""Isolate mlx_lm GDN: same kernel for prefill and verify, sequential in T."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_update

# Bonsai GDN geometry (config.json text_config)
B, HK, HV, DK, DV = 1, 16, 48, 128, 128
LAYERS = 48  # 64 layers, full_attention every 4th → 48 GDN
OUT = Path("results/gdn_path.json")


def _sync():
    mx.synchronize()


def bench_kernel(T: int, warmup=3, iters=8) -> dict:
    q = mx.random.normal((B, T, HK, DK)).astype(mx.float16)
    k = mx.random.normal((B, T, HK, DK)).astype(mx.float16)
    v = mx.random.normal((B, T, HV, DV)).astype(mx.float16)
    g = mx.random.uniform(0.5, 1.0, (B, T, HV)).astype(mx.float32)
    beta = mx.random.uniform(0.0, 1.0, (B, T, HV)).astype(mx.float16)
    state = mx.zeros((B, HV, DV, DK), dtype=mx.float32)
    mx.eval(q, k, v, g, beta, state)

    def run():
        y, s = gated_delta_kernel(q, k, v, g, beta, state, None)
        return y, s

    for _ in range(warmup):
        mx.eval(run())
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(run())
    _sync()
    elapsed = time.perf_counter() - t0
    per = elapsed / iters
    return {
        "T": T,
        "ms": 1000 * per,
        "ms_x48_layers": 1000 * per * LAYERS,
        "ms_per_token_x48": 1000 * per * LAYERS / T,
    }


def bench_update_matches_kernel(T: int) -> dict:
    q = mx.random.normal((B, T, HK, DK)).astype(mx.float16)
    k = mx.random.normal((B, T, HK, DK)).astype(mx.float16)
    v = mx.random.normal((B, T, HV, DV)).astype(mx.float16)
    a = mx.random.normal((B, T, HV)).astype(mx.float16)
    b = mx.random.normal((B, T, HV)).astype(mx.float16)
    A_log = mx.random.uniform(0.0, 2.0, (HV,)).astype(mx.float32)
    dt_bias = mx.ones((HV,), dtype=mx.float32)
    state = mx.zeros((B, HV, DV, DK), dtype=mx.float32)
    y_k, s_k = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True)
    y_o, s_o = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=False)
    mx.eval(y_k, s_k, y_o, s_o)
    return {
        "T": T,
        "y_max_abs": float(mx.max(mx.abs(y_k.astype(mx.float32) - y_o.astype(mx.float32))).item()),
        "state_max_abs": float(mx.max(mx.abs(s_k.astype(mx.float32) - s_o.astype(mx.float32))).item()),
    }


def count_calls_on_model():
    from mlx_lm.models import gated_delta as gd
    from mlx_lm.models.cache import make_prompt_cache

    from monkeyinference.bench import EXPLAIN_PROMPT
    from monkeyinference.load import apply_chat, load_text_model
    from monkeyinference.spec import pin_cache, revert_cache

    loaded = load_text_model(use_custom_kernels=True)
    model = loaded.model
    orig = gd.gated_delta_update
    log = []

    def wrapped(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        log.append(
            {
                "T": int(q.shape[1]),
                "use_kernel": bool(use_kernel),
                "training": bool(model.training),
                "q_shape": [int(x) for x in q.shape],
                "state_none": state is None,
            }
        )
        return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)

    gd.gated_delta_update = wrapped
    # Monkeypatch the already-imported binding in qwen3_5
    import mlx_lm.models.qwen3_5 as q35

    q35.gated_delta_update = wrapped

    prompt = apply_chat(loaded.tokenizer, EXPLAIN_PROMPT, enable_thinking=False)
    tokens = mx.array(loaded.tokenizer.encode(prompt), dtype=mx.uint32)
    cache = make_prompt_cache(model)
    # Prefill all-but-last (same leftover protocol)
    y = tokens
    log.clear()
    t0 = time.perf_counter()
    while y.size > 1:
        n = min(512, int(y.size) - 1)
        mx.eval(model(y[:n][None], cache=cache), [c.state for c in cache])
        y = y[n:]
    mx.synchronize()
    prefill_s = time.perf_counter() - t0
    prefill_calls = list(log)
    leftover = y
    draft = mx.array([8068, 21473, 45543, 369, 264, 14334, 421], dtype=mx.uint32)
    eight = mx.concatenate([leftover, draft])
    pin = pin_cache(cache)
    log.clear()
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(model(leftover[None], cache=cache))
    mx.synchronize()
    t1 = time.perf_counter() - t0
    t1_calls = list(log)
    revert_cache(pin)
    log.clear()
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(model(eight[None], cache=cache))
    mx.synchronize()
    t8 = time.perf_counter() - t0
    t8_calls = list(log)
    gd.gated_delta_update = orig
    q35.gated_delta_update = orig
    return {
        "prefill_tokens": int(tokens.size) - 1,
        "prefill_s": prefill_s,
        "prefill_n_calls": len(prefill_calls),
        "prefill_unique_T": sorted({c["T"] for c in prefill_calls}),
        "prefill_all_kernel": all(c["use_kernel"] for c in prefill_calls),
        "T1_ms": 1000 * t1,
        "T1_n_calls": len(t1_calls),
        "T1_unique_T": sorted({c["T"] for c in t1_calls}),
        "T8_ms": 1000 * t8,
        "T8_n_calls": len(t8_calls),
        "T8_unique_T": sorted({c["T"] for c in t8_calls}),
        "T8_all_kernel": all(c["use_kernel"] for c in t8_calls),
        "same_fn": True,
    }


def main():
    print("isolated gated_delta_kernel (Bonsai shapes, 1 layer then x48)")
    kernel_rows = []
    for T in (1, 2, 3, 4, 8, 16, 64, 512):
        row = bench_kernel(T)
        kernel_rows.append(row)
        print(
            f"  T={T:3d}  {row['ms']:7.2f} ms/layer  "
            f"x48={row['ms_x48_layers']:7.1f} ms  "
            f"per-token-x48={row['ms_per_token_x48']:6.2f} ms"
        )
    print("kernel vs ops (one layer, not 48)")
    parity = [bench_update_matches_kernel(T) for T in (1, 8)]
    for p in parity:
        print(f"  T={p['T']} y_max_abs={p['y_max_abs']:.4g} state_max_abs={p['state_max_abs']:.4g}")
    print("full-model call counts")
    counts = count_calls_on_model()
    print(json.dumps(counts, indent=2))
    payload = {
        "geometry": {
            "B": B,
            "Hk": HK,
            "Hv": HV,
            "Dk": DK,
            "Dv": DV,
            "gdn_layers": LAYERS,
            "state_bytes_per_layer": B * HV * DV * DK * 4,
        },
        "kernel": kernel_rows,
        "kernel_vs_ops": parity,
        "model": counts,
        "conclusion": (
            "Prefill and verify both call mlx_lm.models.gated_delta.gated_delta_update "
            "once per GDN layer with use_kernel=True. The Metal kernel loops "
            "`for (int t = 0; t < T; ++t)` over the leftover+K (or prefill) block "
            "in a single launch. There is no chunkwise-parallel WY path in mlx_lm. "
            "ArraysCache is written once per forward, not once per token."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
