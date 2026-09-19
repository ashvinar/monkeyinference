"""Break-even arithmetic for DFlash + always-8 MMA with GDN commit (replay=0).

Measures, LPM off:
  T=1 qdot vs T=8 MMA target forward (transformer + lm_head, CoW pin)
  leftover+k MMA for k=1..7 (current ABI, not padded to 8 tokens)
  dflash.propose(k) for k=1..7 (block-diffusion: leftover+MASK×7 always)
  isolated GDN T=1..8 (commit-cost proxy; replay assumed 0)

Emits predicted tok/s vs K and min accepts/pass to beat 10.22 greedy.
Draft cost is measured, not estimated. Accepts/pass at K=2,3,7 come from
the leftover-protocol k_sweep (token-identical); other K get min-accepts
only.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_kernel

from monkeyinference.bench import EXPLAIN_PROMPT
from monkeyinference.dflash import AuxCapture, load_drafter
from monkeyinference.load import apply_chat, load_text_model
from monkeyinference.spec import pin_cache, revert_cache

GREEDY_TPS = 10.22
GEMV_MMA_RATIO = 2.25  # MLP-up ternary_qmm_m8 / qdot, for the paper formula
ACCEPTS = {2: 2.2857142857142856, 3: 2.526315789473684, 7: 3.0}
OUT_DEFAULT = Path("results/breakeven.json")

B, HK, HV, DK, DV = 1, 16, 48, 128, 128
GDN_LAYERS = 48


def _median_ms(samples: list[float]) -> float:
    s = sorted(samples)
    return 1000.0 * s[len(s) // 2]


def _mean_ms(samples: list[float]) -> float:
    return 1000.0 * statistics.mean(samples)


def bench_gdn(ts=(1, 2, 3, 4, 5, 6, 7, 8), warmup=3, iters=8) -> list[dict]:
    rows = []
    for T in ts:
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
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(run())
        mx.synchronize()
        per = (time.perf_counter() - t0) / iters
        rows.append(
            {
                "T": T,
                "ms": 1000 * per,
                "ms_x48_layers": 1000 * per * GDN_LAYERS,
                "ms_per_token_x48": 1000 * per * GDN_LAYERS / T,
            }
        )
        print(
            f"  GDN T={T}: {rows[-1]['ms']:.2f} ms/layer, "
            f"{rows[-1]['ms_x48_layers']:.1f} ms ×48",
            flush=True,
        )
    return rows


def time_forward(fn, pin, n=5, warmup=2) -> dict:
    for _ in range(warmup):
        revert_cache(pin)
        mx.eval(fn())
    samples = []
    for _ in range(n):
        revert_cache(pin)
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return {
        "median_ms": _median_ms(samples),
        "mean_ms": _mean_ms(samples),
        "min_ms": 1000.0 * min(samples),
        "samples_ms": [1000.0 * x for x in samples],
    }


def time_propose(dflash, leftover: int, embed, lm_head, k: int, n=5, warmup=2) -> dict:
    for _ in range(warmup):
        ids = dflash.propose(leftover_token=leftover, embed=embed, lm_head=lm_head, k=k)
        assert isinstance(ids, list)
    samples = []
    last = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        last = dflash.propose(leftover_token=leftover, embed=embed, lm_head=lm_head, k=k)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return {
        "k": k,
        "n_ids": len(last),
        "ids": [int(x) for x in last],
        "median_ms": _median_ms(samples),
        "mean_ms": _mean_ms(samples),
        "min_ms": 1000.0 * min(samples),
        "samples_ms": [1000.0 * x for x in samples],
    }


def table_rows(
    t1_ms: float,
    t8_ms: float,
    draft: dict[int, float],
    accepts_map: dict[int, float] | None = None,
) -> list[dict]:
    greedy_ms = 1000.0 / GREEDY_TPS
    verify_ratio = t8_ms / t1_ms if t1_ms else None
    acc_src = ACCEPTS if accepts_map is None else accepts_map
    rows = []
    for k in range(1, 8):
        d_ms = draft[k]
        pass_ms = t8_ms + d_ms  # replay = 0 (GDN prefix commit)
        min_accepts = GREEDY_TPS * pass_ms / 1000.0
        accepts = acc_src.get(k)
        pred = None if accepts is None else accepts / (pass_ms / 1000.0)
        draft_block = d_ms / greedy_ms
        draft_per_token = d_ms / (k * greedy_ms)
        # Paper formula uses GEMV 2.25× plus K×draft_step. Block-diffusion
        # propose is one leftover+MASK×7 pass, so K× overstates draft.
        paper_k_linear = GEMV_MMA_RATIO + k * draft_per_token
        paper_block = (verify_ratio or GEMV_MMA_RATIO) + draft_block
        wins = None if accepts is None else accepts > min_accepts
        rows.append(
            {
                "k": k,
                "draft_ms": d_ms,
                "verify_always8_ms": t8_ms,
                "pass_ms_replay0": pass_ms,
                "accepts_per_pass": accepts,
                "pred_tok_s": pred,
                "min_accepts_to_beat_10_22": min_accepts,
                "wins_on_paper": wins,
                "draft_block_in_greedy_steps": draft_block,
                "draft_per_token_in_greedy_steps": draft_per_token,
                "paper_min_accepts_2_25_plus_K_draft": paper_k_linear,
                "measured_min_accepts_T8_over_T1_plus_draft": paper_block,
                "beats_greedy_margin_tok_s": None
                if pred is None
                else pred - GREEDY_TPS,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--accepts-k7", type=float, default=None)
    p.add_argument(
        "--eval-json",
        type=Path,
        default=Path.home() / ".monkey/dflash-ft/eval.json",
    )
    p.add_argument("--skip-gdn", action="store_true")
    args = p.parse_args(argv)

    accepts = dict(ACCEPTS)
    if args.accepts_k7 is not None:
        accepts[7] = float(args.accepts_k7)
    elif args.eval_json.is_file():
        ev = json.loads(args.eval_json.read_text())
        if "accepts_per_pass" in ev:
            accepts[7] = float(ev["accepts_per_pass"])

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "greedy_tps": GREEDY_TPS,
        "gemv_mma_ratio": GEMV_MMA_RATIO,
        "accepts_from_k_sweep": accepts,
        "replay": 0,
        "adapter_eval_json": str(args.eval_json) if args.eval_json.is_file() else None,
        "note": (
            "DFlash propose always builds leftover+MASK×7 then select() walks k. "
            "Draft backbone is ~flat in K. Verify is always-8 MMA (PackedLinear "
            "already pads M<8). GDN commit assumed (no leftover replay). "
            "K=7 accepts come from leftover-protocol eval when --eval-json is set."
        ),
    }

    if args.skip_gdn:
        report["gdn"] = None
        print("=== skip isolated GDN ===", flush=True)
    else:
        print("=== isolated GDN (commit-cost proxy) ===", flush=True)
        report["gdn"] = bench_gdn()

    print("=== load Bonsai + DFlash ===", flush=True)
    t_load = time.perf_counter()
    loaded = load_text_model(use_custom_kernels=True)
    model = loaded.model
    dflash = load_drafter()
    mx.eval(model.parameters())
    report["load_s"] = time.perf_counter() - t_load
    print(f"  loaded in {report['load_s']:.1f}s", flush=True)

    prompt = apply_chat(loaded.tokenizer, EXPLAIN_PROMPT, enable_thinking=False)
    tokens = mx.array(loaded.tokenizer.encode(prompt), dtype=mx.uint32)
    cache = make_prompt_cache(model)
    aux = AuxCapture(model.model)

    def target_body(ids, cache_):
        hidden = model.model(ids, cache=cache_)
        aux.record_last_forward()
        return hidden

    print("=== prefill explain ===", flush=True)
    y = tokens
    t0 = time.perf_counter()
    while y.size > 1:
        n = min(512, int(y.size) - 1)
        mx.eval(target_body(y[:n][None], cache), [c.state for c in cache])
        y = y[n:]
    leftover = y
    leftover_id = int(leftover.reshape(-1)[-1].item())
    dflash.commit_new(aux.context())
    aux.close()
    report["prefill_s"] = time.perf_counter() - t0
    report["prompt_tokens"] = int(tokens.size)
    report["leftover_id"] = leftover_id
    print(
        f"  prefill {report['prefill_s']:.2f}s, leftover={leftover_id}",
        flush=True,
    )

    pin = pin_cache(cache)
    dummy7 = mx.array([8068, 21473, 45543, 369, 264, 14334, 421], dtype=mx.uint32)
    eight = mx.concatenate([leftover, dummy7])

    def logits(ids):
        hidden = model.model(ids[None], cache=cache)
        return model.lm_head(hidden)

    print("=== T=1 qdot vs T=8 MMA (model + lm_head) ===", flush=True)
    t1 = time_forward(lambda: logits(leftover), pin)
    t8 = time_forward(lambda: logits(eight), pin)
    report["t1_qdot"] = t1
    report["t8_mma"] = t8
    report["t8_over_t1"] = t8["median_ms"] / t1["median_ms"]
    print(
        f"  T=1 {t1['median_ms']:.1f} ms  T=8 {t8['median_ms']:.1f} ms  "
        f"ratio {report['t8_over_t1']:.2f}×",
        flush=True,
    )

    print("=== leftover+k MMA (current ABI) ===", flush=True)
    leftover_k = {}
    for k in range(1, 8):
        ids = mx.concatenate([leftover, dummy7[:k]])
        row = time_forward(lambda ids=ids: logits(ids), pin, n=4, warmup=1)
        leftover_k[k] = row
        print(f"  leftover+{k} (M={k+1}) {row['median_ms']:.1f} ms", flush=True)
    report["leftover_plus_k"] = leftover_k

    print("=== DFlash propose(k) ===", flush=True)
    draft = {}
    draft_rows = []
    for k in range(1, 8):
        row = time_propose(
            dflash,
            leftover_id,
            model.model.embed_tokens,
            model.lm_head,
            k,
        )
        draft[k] = row["median_ms"]
        draft_rows.append(row)
        print(
            f"  k={k} {row['median_ms']:.1f} ms  n_ids={row['n_ids']}",
            flush=True,
        )
    report["draft_propose"] = draft_rows

    greedy_ms = 1000.0 / GREEDY_TPS
    report["greedy_ms"] = greedy_ms
    report["table"] = table_rows(
        t1["median_ms"], t8["median_ms"], draft, accepts_map=accepts
    )

    winners = [r for r in report["table"] if r["wins_on_paper"] is True]
    close = [
        r
        for r in report["table"]
        if r["accepts_per_pass"] is not None
        and r["pred_tok_s"] is not None
        and abs(r["pred_tok_s"] - GREEDY_TPS) / GREEDY_TPS <= 0.08
    ]
    if winners:
        ruling = (
            f"WIN: K={ [r['k'] for r in winners] } beat 10.22 on paper with "
            "always-8 + replay=0. Build always-8 ABI + verify_gdn_commit."
        )
        action = "build_always8_gdn_commit"
    elif close:
        ruling = (
            "CLOSE: predicted tok/s within 8% of 10.22. Chase the acceptance "
            "gap (3.00 vs published 4.1–5.5), not more kernel work."
        )
        action = "chase_acceptance_gap"
    else:
        measured = [r for r in report["table"] if r["pred_tok_s"] is not None]
        best = max(measured, key=lambda r: r["pred_tok_s"]) if measured else None
        ruling = (
            "DEAD: no measured K beats 10.22 on paper with always-8 MMA + "
            f"replay=0. Best predicted {best['pred_tok_s']:.2f} tok/s at K={best['k']} "
            f"(accepts={best['accepts_per_pass']:.2f}, need "
            f"{best['min_accepts_to_beat_10_22']:.2f}). Speculation is formally "
            "dead on this 10-core GPU at the current acceptance and 2.25× verify."
            if best
            else "DEAD: incomplete measurements."
        )
        action = "speculation_dead"

    report["ruling"] = ruling
    report["action"] = action
    if mx.metal.is_available():
        report["peak_memory_gb"] = mx.metal.get_peak_memory() / 1e9
        report["active_memory_gb"] = mx.metal.get_active_memory() / 1e9

    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"action": action, "ruling": ruling, "table": report["table"]}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
