"""Fine-tune the existing DFlash Q4 draft onto 2-bit Bonsai.

Retry of the 64-prompt overfit: 1200 diverse teacher sequences, select on
held-out explain accepts, LoRA r=8 / wd=0.1 / 2k steps. Identity-safe.

    export PYTHONPATH=src
    ~/.monkey/mlx-venv/bin/python scripts/finetune_dflash.py --scope
    scripts/run_dflash_ft_detached.sh
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx

from monkeyinference.dflash import load_drafter, reset_drafter
from monkeyinference.dflash_distill import (
    AUX_BUDGET_GB,
    DEFAULT_DIR,
    DISTILL_PROMPTS,
    FrozenHead,
    MIN_FREE_GB,
    MIN_WINDOWS,
    STEP_TIME_ABORT_S,
    WIN_ACCEPTS,
    append_sequence,
    collect_sequence,
    evaluate_compare,
    evaluate_holdout,
    infer_resume_step,
    iter_windows,
    load_dataset,
    train,
)
from monkeyinference.dflash_lora import LORA_RANK, load_adapters, wrap_drafter
from monkeyinference.dflash_prompts import HOLDOUT_PROMPTS, assert_holdout_disjoint
from monkeyinference.load import load_text_model


def _free_gb() -> float:
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 1e9


def scope_report() -> dict:
    q4_params = 1.79699712e9
    # r=8 is ~half of the overfit r=16 run.
    lora = 8.646656e6 * (LORA_RANK / 16)
    n_prompts = len(DISTILL_PROMPTS)
    aux_est = n_prompts * 90 * 25600 * 2 / 1e9
    free = _free_gb()
    return {
        "free_disk_gb": free,
        "full_ft_fp16_master_adam_gb": q4_params * (2 + 4 + 8) / 1e9,
        "full_ft_fits": False,
        "lora_rank": LORA_RANK,
        "lora_params": int(lora),
        "lora_adam_mb": lora * 12 / 1e6,
        "lm_head_fp16_gb": 248320 * 5120 * 2 / 1e9,
        "n_train_prompts": n_prompts,
        "n_holdout_prompts": len(HOLDOUT_PROMPTS),
        "teacher_aux_est_gb": aux_est,
        "min_windows": MIN_WINDOWS,
        "win_accepts": WIN_ACCEPTS,
        "midpoint_min_accepts": 3.00,
        "identity": (
            "Verification is leftover-greedy: accepted drafts must match the "
            "target argmax. A worse adapter costs speed; it cannot change output."
        ),
        "plan": (
            "Bounded retry: 1200 diverse Bonsai teacher sequences, LoRA r=8, "
            "wd=0.1, 2000 steps, select on held-out explain accepts. Kill if "
            "not above 3.00 by midpoint. Win bar 3.70 from greedy 10.75 and "
            "344.2 ms pass. Do not fit train loss."
        ),
    }


def cmd_scope() -> int:
    assert_holdout_disjoint(DISTILL_PROMPTS)
    rep = scope_report()
    print(json.dumps(rep, indent=2))
    print(
        "\nFull fine-tune does not fit "
        f"({rep['full_ft_fp16_master_adam_gb']:.1f} GB > {_free_gb():.1f} GB free)."
    )
    print(
        f"LoRA r={rep['lora_rank']} ~{rep['lora_params']/1e6:.1f}M params, "
        f"Adam ~{rep['lora_adam_mb']:.0f} MB, aux est ~{rep['teacher_aux_est_gb']:.1f} GB, "
        f"{rep['n_train_prompts']} train prompts, win bar {rep['win_accepts']:.2f}."
    )
    print(rep["identity"])
    print(rep["plan"])
    return 0


def cmd_collect(args, loaded=None) -> dict:
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    assert_holdout_disjoint(DISTILL_PROMPTS)
    prompts = DISTILL_PROMPTS[: args.limit] if args.limit else DISTILL_PROMPTS
    meta_path = out / "meta.json"
    done = json.loads(meta_path.read_text()) if meta_path.is_file() else []
    start = len(done)
    aux_dir = out / "aux"
    aux_bytes = 0
    if aux_dir.is_dir():
        aux_bytes = sum(p.stat().st_size for p in aux_dir.glob("*.npy"))
    print(
        f"=== teacher collect {len(prompts)} prompts (resume at {start}), "
        f"free disk {_free_gb():.1f} GB aux_gb={aux_bytes/1e9:.2f} ===",
        flush=True,
    )
    if loaded is None:
        loaded = load_text_model(mix_five_trit=False)
    t0 = time.perf_counter()
    summary = {
        "n_sequences": start,
        "n_windows": sum(1 for m in done for _ in iter_windows(m)),
        "aux_bytes": aux_bytes,
        "directory": str(out),
        "stopped": None,
    }
    for i in range(start, len(prompts)):
        if _free_gb() < MIN_FREE_GB:
            summary["stopped"] = f"free disk {_free_gb():.2f} GB < {MIN_FREE_GB}"
            print("STOP collect:", summary["stopped"], flush=True)
            break
        aux_gb = summary["aux_bytes"] / 1e9
        if i > start and aux_gb > AUX_BUDGET_GB:
            summary["stopped"] = f"aux {aux_gb:.2f} GB > {AUX_BUDGET_GB}"
            print("STOP collect:", summary["stopped"], flush=True)
            break
        user, ntok = prompts[i]
        ntok = min(ntok, args.max_new)
        print(f"  [{i+1}/{len(prompts)}] {user[:70]!r} max_new={ntok}", flush=True)
        row = collect_sequence(loaded, user, ntok)
        summary = append_sequence(row, out, i)
        print(
            f"      gen={row['gen_tokens']} tok in {row['decode_s']:.1f}s "
            f"aux={row['aux'].shape} windows={summary['n_windows']} "
            f"aux_gb={summary['aux_bytes']/1e9:.2f}",
            flush=True,
        )
        del row
        gc.collect()
        mx.clear_cache()
    summary["collect_s"] = time.perf_counter() - t0
    summary["free_disk_gb"] = _free_gb()
    if summary["n_windows"] < MIN_WINDOWS:
        print(
            f"WARN: windows={summary['n_windows']} < {MIN_WINDOWS} "
            "(10× 3032). Dataset still too small if collect stopped early.",
            flush=True,
        )
    print(json.dumps({k: summary[k] for k in summary if k != "directory"}, indent=2), flush=True)
    return summary


def cmd_train(args, loaded=None) -> dict:
    out = Path(args.dir)
    rows = load_dataset(out)
    n_win = sum(1 for row in rows for _ in iter_windows(row))
    steps = args.steps
    if steps is None:
        steps = 2000
    start_step = 0
    adapter_file = out / "adapters.safetensors"
    if args.resume:
        start_step = infer_resume_step(out)
        if start_step >= steps:
            print(
                f"  resume: already at step {start_step}/{steps}; skipping train",
                flush=True,
            )
            return {
                "n_windows": n_win,
                "steps": steps,
                "start_step": start_step,
                "skipped": "already_complete",
                "aborted": None,
            }
    if n_win < MIN_WINDOWS:
        print(
            f"WARN: training on {n_win} windows < {MIN_WINDOWS}. "
            "This retry is supposed to be 10× the overfit set.",
            flush=True,
        )
    print(
        f"=== load draft + wrap LoRA r={args.rank}, {len(rows)} sequences, "
        f"{n_win} windows ===",
        flush=True,
    )
    if loaded is None:
        loaded = load_text_model(mix_five_trit=False)
    embed = loaded.model.model.embed_tokens
    head = FrozenHead(loaded.model.lm_head)
    print(f"  dequant lm_head {tuple(head.weight.shape)}", flush=True)
    # Keep the 27B body: checkpoint eval needs it for held-out accepts.

    reset_drafter()
    drafter = load_drafter(adapter_path=False)
    adapters = wrap_drafter(drafter, r=args.rank)
    print(f"  trainable {adapters.n_trainable/1e6:.2f}M params wd={args.weight_decay}", flush=True)

    if args.resume:
        if start_step > 0 and adapter_file.is_file():
            load_adapters(adapters, adapter_file)
            print(
                f"  resume from step {start_step}/{steps} using {adapter_file}",
                flush=True,
            )
        elif start_step > 0:
            print(
                "  --resume requested but adapters.safetensors is missing; "
                "starting at 0",
                flush=True,
            )
            start_step = 0
        else:
            print("  --resume: no checkpoint; starting at 0", flush=True)
    print(
        f"  windows={n_win} steps={steps} start_step={start_step} "
        f"lr={args.lr} abort if first step>{STEP_TIME_ABORT_S}s "
        f"win_bar={WIN_ACCEPTS:.2f}",
        flush=True,
    )

    def holdout_eval(_step: int):
        return evaluate_holdout(loaded, HOLDOUT_PROMPTS)

    log = train(
        drafter,
        adapters,
        embed,
        head,
        rows,
        steps=steps,
        lr=args.lr,
        out=out,
        start_step=start_step,
        weight_decay=args.weight_decay,
        holdout_eval=holdout_eval,
    )
    log["free_disk_gb"] = _free_gb()
    (out / "train.json").write_text(json.dumps(log, indent=2, default=str))
    slim = {k: log[k] for k in log if k not in ("losses_every_25", "ckpt_eval")}
    print(json.dumps(slim, indent=2, default=str), flush=True)
    return log


def cmd_eval(args, loaded=None) -> dict:
    print("=== eval explain leftover-identity + DFlash K=7 (stock vs LoRA) ===", flush=True)
    reset_drafter()
    if loaded is None:
        loaded = load_text_model(mix_five_trit=False)
    adapters = Path(args.dir) / "adapters.best.safetensors"
    if not adapters.is_file():
        adapters = Path(args.dir) / "adapters.safetensors"
    ev = evaluate_compare(
        loaded, adapters if adapters.is_file() else None, num_draft=7
    )
    (Path(args.dir) / "eval.json").write_text(json.dumps(ev, indent=2, default=str))
    (Path(args.dir) / "compare.json").write_text(json.dumps(ev, indent=2, default=str))
    paper = ev["paper_k7"]
    stock = ev.get("stock") or {}
    print(
        f"  greedy {ev['greedy_tps']:.2f} tok/s  leftover {ev['leftover_tps']:.2f}  "
        f"stock accepts={stock.get('accepts_per_pass')}  "
        f"adapter accepts={ev.get('accepts_per_pass')}  "
        f"identity={ev['identity_vs_leftover']} france={ev['france']!r}",
        flush=True,
    )
    print(
        f"  paper K=7: pred {paper['pred_tok_s']:.2f} tok/s "
        f"(need {WIN_ACCEPTS:.2f} accepts vs measured greedy 10.75, "
        f"have {paper['accepts_per_pass']:.2f}, "
        f"wins={float(paper['accepts_per_pass']) > WIN_ACCEPTS})",
        flush=True,
    )
    return ev


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--scope", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Cap distill prompts (0 = all)")
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--rank", type=int, default=LORA_RANK)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue collect/train from meta.json + adapters",
    )
    args = p.parse_args()
    if args.scope:
        return cmd_scope()
    print(json.dumps(scope_report(), indent=2), flush=True)
    if _free_gb() < MIN_FREE_GB:
        print(
            f"STOP: less than {MIN_FREE_GB} GB free. Not deleting anything. Need more disk."
        )
        return 2
    log: dict = {"scope": scope_report()}
    loaded = None
    n_target = args.limit if args.limit else len(DISTILL_PROMPTS)
    meta_exists = (args.dir / "meta.json").is_file()
    n_have = len(json.loads((args.dir / "meta.json").read_text())) if meta_exists else 0
    do_collect = args.collect_only or (
        not args.train_only and not args.eval_only and n_have < n_target
    )
    if args.collect_only or do_collect:
        loaded = load_text_model(mix_five_trit=False)
        log["collect"] = cmd_collect(args, loaded)
        if args.collect_only:
            Path(args.dir, "run.json").write_text(json.dumps(log, indent=2, default=str))
            return 0
        if log["collect"]["n_windows"] < MIN_WINDOWS and log["collect"].get("stopped"):
            print(
                "STOP: collect ended under the 10× window floor. Not training a small set.",
            )
            Path(args.dir, "run.json").write_text(json.dumps(log, indent=2, default=str))
            return 4
    elif not args.eval_only:
        print(f"reusing dataset at {args.dir}", flush=True)
    if not args.eval_only:
        log["train"] = cmd_train(args, loaded)
        if log["train"].get("aborted"):
            print("STOP:", log["train"]["aborted"])
            Path(args.dir, "run.json").write_text(json.dumps(log, indent=2, default=str))
            return 3
        if args.train_only:
            Path(args.dir, "run.json").write_text(json.dumps(log, indent=2, default=str))
            return 0
    log["eval"] = cmd_eval(args, loaded)
    Path(args.dir, "run.json").write_text(json.dumps(log, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
