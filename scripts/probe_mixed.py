"""Per-shape five-trit mix: calibrate on real weights, greedy + identity, break-even.

Does not train. Loads Bonsai once. Calibrates unique (N,K) with interleaved
qdot vs five-trit, enables trit only where it wins, then measures greedy
explain tok/s vs leftover (token identity). Remeasures T=1 vs T=8 and the
K table against the new greedy. If still short of beating greedy, runs DFlash
K=7 and histograms first-reject slot in the draft block.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from monkeyinference.bench import EXPLAIN_PROMPT, FRANCE_PROMPT
from monkeyinference.dflash import AuxCapture, load_drafter
from monkeyinference.generate import generate
from monkeyinference.load import apply_chat, load_text_model
from monkeyinference.spec import pin_cache, revert_cache
from monkeyinference.trit import apply_mixed_five_trit, calibrate_five_trit_wins

OUT = Path("results/mixed_trit.json")
ACCEPTS = {2: 2.2857142857142856, 3: 2.526315789473684, 7: 3.0}
OLD_GREEDY = 10.22


def _median_ms(samples: list[float]) -> float:
    s = sorted(samples)
    return 1000.0 * s[len(s) // 2]


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
        "min_ms": 1000.0 * min(samples),
        "samples_ms": [1000.0 * x for x in samples],
    }


def time_propose(dflash, leftover: int, embed, lm_head, k: int, n=4, warmup=1) -> dict:
    for _ in range(warmup):
        dflash.propose(leftover_token=leftover, embed=embed, lm_head=lm_head, k=k)
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
        "median_ms": _median_ms(samples),
        "samples_ms": [1000.0 * x for x in samples],
    }


def reject_histogram(prefixes: list[int], k: int) -> dict:
    """prefixes[i] = number of draft tokens accepted on pass i (0..k)."""
    counts = Counter(int(p) for p in prefixes)
    n = max(len(prefixes), 1)
    return {
        "n_passes": len(prefixes),
        "k": k,
        "accept_n_drafts": {str(i): counts.get(i, 0) for i in range(k + 1)},
        "frac_first_reject_at_slot": {str(i): counts.get(i, 0) / n for i in range(k)},
        "frac_full_k_accepted": counts.get(k, 0) / n,
        "mean_drafts_accepted": (sum(prefixes) / len(prefixes)) if prefixes else 0.0,
        "note": (
            "slot 0 = first draft token rejected (leftover next-token disagreed). "
            "Full K accepted means every draft matched; bonus is extra."
        ),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"old_greedy_tps": OLD_GREEDY}

    print("=== load (2-bit, no mix yet) ===", flush=True)
    loaded = load_text_model(use_custom_kernels=True, mix_five_trit=False)
    model = loaded.model

    print("=== calibrate unique (N,K) on real weights ===", flush=True)
    cal = calibrate_five_trit_wins(model)
    report["calibration"] = cal
    wins = frozenset((int(a), int(b)) for a, b in cal["wins"])
    for row in cal["rows"]:
        tag = "WIN" if row.get("win") else "keep-2bit"
        extra = ""
        if row.get("trit_over_qdot") is not None:
            extra = (
                f" trit/qdot={row['trit_over_qdot']:.2f}"
                f" qdot={row['qdot_us']:.0f}us trit={row['trit_us']:.0f}us"
            )
        print(f"  {tag} {row['n']}x{row['k']} {row.get('path')}{extra}", flush=True)
    print(f"  winning shapes: {sorted(wins)}", flush=True)

    mix = apply_mixed_five_trit(model, wins=wins)
    report["mix"] = {k: mix[k] for k in ("n_enabled", "n_skipped", "wins")}
    print(
        f"  enabled {mix['n_enabled']} linears, skipped {mix['n_skipped']}",
        flush=True,
    )

    print("=== greedy explain (stream) + leftover identity ===", flush=True)
    greedy = generate(loaded, EXPLAIN_PROMPT, max_tokens=48, speculative=False)
    leftover = generate(
        loaded, EXPLAIN_PROMPT, max_tokens=48, speculative=True, draft="none"
    )
    france = generate(loaded, FRANCE_PROMPT, max_tokens=8, speculative=False)
    identity = list(greedy.tokens) == list(leftover.tokens)
    report["greedy"] = greedy.to_dict()
    report["leftover"] = leftover.to_dict()
    report["france"] = {
        "text": france.text,
        "tokens": france.tokens,
        "generation_tps": france.generation_tps,
    }
    report["token_identity_greedy_vs_leftover"] = identity
    print(
        f"  mixed greedy {greedy.generation_tps:.2f} tok/s  leftover "
        f"{leftover.generation_tps:.2f}  identity={identity}  france={france.text!r}",
        flush=True,
    )

    print("=== T=1 vs T=8 after mix ===", flush=True)
    prompt = apply_chat(loaded.tokenizer, EXPLAIN_PROMPT, enable_thinking=False)
    tokens = mx.array(loaded.tokenizer.encode(prompt), dtype=mx.uint32)
    cache = make_prompt_cache(model)
    y = tokens
    while y.size > 1:
        n = min(512, int(y.size) - 1)
        mx.eval(model.model(y[:n][None], cache=cache), [c.state for c in cache])
        y = y[n:]
    leftover_ids = y
    pin = pin_cache(cache)
    dummy7 = mx.array([8068, 21473, 45543, 369, 264, 14334, 421], dtype=mx.uint32)
    eight = mx.concatenate([leftover_ids, dummy7])

    def logits(ids):
        hidden = model.model(ids[None], cache=cache)
        return model.lm_head(hidden)

    t1 = time_forward(lambda: logits(leftover_ids), pin)
    t8 = time_forward(lambda: logits(eight), pin)
    report["t1"] = t1
    report["t8"] = t8
    report["t8_over_t1"] = t8["median_ms"] / t1["median_ms"]
    print(
        f"  T=1 {t1['median_ms']:.1f} ms  T=8 {t8['median_ms']:.1f} ms  "
        f"ratio {report['t8_over_t1']:.2f}x",
        flush=True,
    )

    greedy_tps = float(greedy.generation_tps)
    greedy_ms = 1000.0 / greedy_tps if greedy_tps else 0.0
    report["greedy_ms"] = greedy_ms

    print("=== DFlash propose + K table vs new greedy ===", flush=True)
    aux = AuxCapture(model.model)
    dflash = load_drafter()
    cache2 = make_prompt_cache(model)
    y2 = tokens
    while y2.size > 1:
        n = min(512, int(y2.size) - 1)
        hidden = model.model(y2[:n][None], cache=cache2)
        aux.record_last_forward()
        mx.eval(hidden, [c.state for c in cache2])
        y2 = y2[n:]
    leftover_id = int(y2.reshape(-1)[-1].item())
    dflash.commit_new(aux.context())
    aux.close()
    draft = {}
    draft_rows = []
    for k in range(1, 8):
        row = time_propose(
            dflash, leftover_id, model.model.embed_tokens, model.lm_head, k
        )
        draft[k] = row["median_ms"]
        draft_rows.append(row)
        print(f"  propose k={k} {row['median_ms']:.1f} ms", flush=True)
    report["draft_propose"] = draft_rows

    table = []
    for k in range(1, 8):
        pass_ms = t8["median_ms"] + draft[k]
        min_a = greedy_tps * pass_ms / 1000.0
        acc = ACCEPTS.get(k)
        pred = None if acc is None else acc / (pass_ms / 1000.0)
        table.append(
            {
                "k": k,
                "draft_ms": draft[k],
                "verify_ms": t8["median_ms"],
                "pass_ms_replay0": pass_ms,
                "accepts_per_pass": acc,
                "pred_tok_s": pred,
                "min_accepts_to_beat_greedy": min_a,
                "wins_on_paper": None if acc is None else acc > min_a,
                "beats_greedy_margin_tok_s": None if pred is None else pred - greedy_tps,
            }
        )
    report["table"] = table
    winners = [r for r in table if r["wins_on_paper"] is True]
    measured = [r for r in table if r["pred_tok_s"] is not None]
    best = max(measured, key=lambda r: r["pred_tok_s"]) if measured else None
    if winners:
        action = "spec_wins"
        ruling = (
            f"WIN: K={[r['k'] for r in winners]} beat mixed greedy "
            f"{greedy_tps:.2f} on paper."
        )
    else:
        action = "spec_still_short"
        ruling = (
            f"SHORT: mixed greedy {greedy_tps:.2f} tok/s. Best predicted "
            f"{best['pred_tok_s']:.2f} at K={best['k']} "
            f"(accepts={best['accepts_per_pass']:.2f}, need "
            f"{best['min_accepts_to_beat_greedy']:.2f})."
            if best
            else "SHORT: incomplete."
        )
    report["action"] = action
    report["ruling"] = ruling
    print(ruling, flush=True)

    if action == "spec_still_short":
        print("=== DFlash K=7 reject positions + identity ===", flush=True)
        dflash_run = generate(
            loaded,
            EXPLAIN_PROMPT,
            max_tokens=48,
            speculative=True,
            draft="dflash",
            num_draft=7,
        )
        hist = reject_histogram(dflash_run.accept_prefix, 7)
        ident = list(dflash_run.tokens) == list(leftover.tokens)
        report["dflash_k7"] = dflash_run.to_dict()
        report["reject_histogram"] = hist
        report["token_identity_dflash_vs_leftover"] = ident
        print(
            f"  dflash {dflash_run.generation_tps:.2f} tok/s  accepts/pass="
            f"{dflash_run.accepts_per_pass:.2f}  identity={ident}",
            flush=True,
        )
        print(f"  reject hist {hist['accept_n_drafts']}", flush=True)
        print(f"  first-reject frac {hist['frac_first_reject_at_slot']}", flush=True)

    get_peak = getattr(mx, "get_peak_memory", None)
    if get_peak is None and hasattr(mx, "metal"):
        get_peak = getattr(mx.metal, "get_peak_memory", None)
    if callable(get_peak):
        report["peak_memory_gb"] = get_peak() / 1e9

    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
