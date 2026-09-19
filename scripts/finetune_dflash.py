"""Fine-tune the existing DFlash Q4 draft onto 2-bit Bonsai.

Full dequant + Adam of 1.80B params does not fit (≈25 GB). This script
runs LoRA rank-16 on the frozen Q4 draft (~8.6M params, ~0.1 GB Adam,
~0.7 GB teacher-aux cache). Verification still enforces token identity.

    export PYTHONPATH=src
    ~/.monkey/mlx-venv/bin/python scripts/finetune_dflash.py --scope
    ~/.monkey/mlx-venv/bin/python scripts/finetune_dflash.py --resume
    scripts/run_dflash_ft_detached.sh --watch <pid>
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
    DEFAULT_DIR,
    DISTILL_PROMPTS,
    FrozenHead,
    MIN_ACCEPTS_TO_BEAT_GREEDY,
    STEP_TIME_ABORT_S,
    collect_sequence,
    evaluate_explain,
    infer_resume_step,
    iter_windows,
    load_dataset,
    predicted_k7,
    save_dataset,
    train,
)
from monkeyinference.dflash_lora import LORA_RANK, load_adapters, wrap_drafter
from monkeyinference.load import load_text_model


def _free_gb() -> float:
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 1e9


def scope_report() -> dict:
    q4_params = 1.79699712e9
    lora = 8.646656e6
    aux_64x96 = 64 * 96 * 25600 * 2 / 1e9
    free = _free_gb()
    return {
        "free_disk_gb": free,
        "full_ft_fp16_master_adam_gb": q4_params * (2 + 4 + 8) / 1e9,
        "full_ft_fits": False,
        "lora_rank": LORA_RANK,
        "lora_params": int(lora),
        "lora_adam_mb": lora * 12 / 1e6,
        "lm_head_fp16_gb": 248320 * 5120 * 2 / 1e9,
        "teacher_aux_64seq_gb": aux_64x96,
        "identity": (
            "Verification is leftover-greedy: accepted drafts must match the "
            "target argmax. A worse adapter costs speed; it cannot change output."
        ),
        "plan": (
            "LoRA on frozen Q4 DFlash. Teacher = 2-bit Bonsai greedy + aux "
            "hiddens. Hold out the explain prompt."
        ),
    }


def cmd_scope() -> int:
    rep = scope_report()
    print(json.dumps(rep, indent=2))
    print(
        "\nFull fine-tune does not fit "
        f"({rep['full_ft_fp16_master_adam_gb']:.1f} GB > {_free_gb():.1f} GB free)."
    )
    print(
        f"LoRA r={rep['lora_rank']} does fit: {rep['lora_params']/1e6:.1f}M params, "
        f"Adam ~{rep['lora_adam_mb']:.0f} MB, aux cache ~{rep['teacher_aux_64seq_gb']:.2f} GB, "
        f"dequant lm_head ~{rep['lm_head_fp16_gb']:.2f} GB RAM after the 27B body is dropped."
    )
    print(rep["identity"])
    return 0


def cmd_collect(args) -> dict:
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = DISTILL_PROMPTS[: args.limit] if args.limit else DISTILL_PROMPTS
    print(f"=== teacher collect {len(prompts)} prompts, free disk {_free_gb():.1f} GB ===", flush=True)
    loaded = load_text_model(mix_five_trit=False)
    rows = []
    t0 = time.perf_counter()
    for i, (user, ntok) in enumerate(prompts):
        ntok = min(ntok, args.max_new)
        print(f"  [{i+1}/{len(prompts)}] {user[:60]!r} max_new={ntok}", flush=True)
        row = collect_sequence(loaded, user, ntok)
        print(
            f"      gen={row['gen_tokens']} tok in {row['decode_s']:.1f}s "
            f"aux={row['aux'].shape}",
            flush=True,
        )
        rows.append(row)
    summary = save_dataset(rows, out)
    summary["collect_s"] = time.perf_counter() - t0
    summary["free_disk_gb"] = _free_gb()
    print(json.dumps({k: summary[k] for k in summary if k != "directory"}, indent=2), flush=True)
    del loaded, rows
    gc.collect()
    mx.clear_cache()
    return summary


def cmd_train(args) -> dict:
    out = Path(args.dir)
    rows = load_dataset(out)
    n_win = sum(1 for row in rows for _ in iter_windows(row))
    steps = args.steps
    if steps is None:
        steps = min(4000, max(400, n_win * 2))
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
    print(
        f"=== load draft + wrap LoRA r={LORA_RANK}, {len(rows)} sequences ===",
        flush=True,
    )
    # Need Bonsai embed + lm_head. Load text model, steal those, drop body.
    loaded = load_text_model(mix_five_trit=False)
    embed = loaded.model.model.embed_tokens
    head = FrozenHead(loaded.model.lm_head)
    print(f"  dequant lm_head {tuple(head.weight.shape)}", flush=True)
    # Keep embed+head; drop the 64-layer body.
    loaded.model.model.layers = []
    gc.collect()
    mx.clear_cache()

    reset_drafter()
    drafter = load_drafter(adapter_path=False)
    adapters = wrap_drafter(drafter)
    print(f"  trainable {adapters.n_trainable/1e6:.2f}M params", flush=True)

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
        f"abort if first step>{STEP_TIME_ABORT_S}s",
        flush=True,
    )
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
    )
    log["free_disk_gb"] = _free_gb()
    (out / "train.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: log[k] for k in log if k != "losses_every_25"}, indent=2), flush=True)
    reset_drafter()
    del loaded, drafter, adapters, head, embed, rows
    gc.collect()
    mx.clear_cache()
    return log


def cmd_eval(args) -> dict:
    print("=== eval explain leftover-identity + DFlash K=7 ===", flush=True)
    reset_drafter()
    loaded = load_text_model(mix_five_trit=False)
    adapters = Path(args.dir) / "adapters.safetensors"
    if adapters.is_file():
        load_drafter(adapter_path=adapters)
    else:
        load_drafter(adapter_path=False)
        print("  no adapters file; evaluating the stock Q4 draft", flush=True)
    ev = evaluate_explain(loaded, num_draft=7)
    greedy = None
    from monkeyinference.generate import generate
    from monkeyinference.bench import EXPLAIN_PROMPT

    greedy = generate(loaded, EXPLAIN_PROMPT, max_tokens=48, speculative=False)
    ev["greedy_tps"] = greedy.generation_tps
    ev["beats_greedy"] = bool(
        ev["identity_vs_leftover"] and ev["dflash_tps"] > greedy.generation_tps
    )
    ev["paper_k7"] = predicted_k7(ev["accepts_per_pass"])
    ev["need_accepts_vs_10_22"] = MIN_ACCEPTS_TO_BEAT_GREEDY
    (Path(args.dir) / "eval.json").write_text(json.dumps(ev, indent=2, default=str))
    paper = ev["paper_k7"]
    print(
        f"  greedy {greedy.generation_tps:.2f} tok/s  dflash {ev['dflash_tps']:.2f} "
        f"accepts/pass={ev['accepts_per_pass']:.2f} identity={ev['identity_vs_leftover']} "
        f"france={ev['france']!r}",
        flush=True,
    )
    print(
        f"  paper K=7 replay=0: pred {paper['pred_tok_s']:.2f} tok/s "
        f"(need {paper['min_accepts_to_beat_10_22']:.2f} accepts, "
        f"have {paper['accepts_per_pass']:.2f}, "
        f"wins={paper['wins_on_paper']})",
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
    p.add_argument("--limit", type=int, default=0, help="Cap distill prompts (0 = all 64)")
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from adapters.safetensors + train_state.json / run.log",
    )
    args = p.parse_args()
    if args.scope:
        return cmd_scope()
    print(json.dumps(scope_report(), indent=2), flush=True)
    if _free_gb() < 3.0:
        print("STOP: less than 3 GB free. Not deleting anything. Need more disk.")
        return 2
    log: dict = {"scope": scope_report()}
    do_collect = args.collect_only or (
        not args.train_only and not args.eval_only and not (args.dir / "meta.json").is_file()
    )
    if args.collect_only or do_collect:
        log["collect"] = cmd_collect(args)
        if args.collect_only:
            Path(args.dir, "run.json").write_text(json.dumps(log, indent=2, default=str))
            return 0
    elif not args.eval_only:
        print(f"reusing dataset at {args.dir}", flush=True)
    if not args.eval_only:
        log["train"] = cmd_train(args)
        if log["train"].get("aborted"):
            print("STOP:", log["train"]["aborted"])
            Path(args.dir, "run.json").write_text(json.dumps(log, indent=2, default=str))
            return 3
        if args.train_only:
            Path(args.dir, "run.json").write_text(json.dumps(log, indent=2, default=str))
            return 0
    log["eval"] = cmd_eval(args)
    Path(args.dir, "run.json").write_text(json.dumps(log, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
