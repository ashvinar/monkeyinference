"""Distill the DFlash draft onto 2-bit Bonsai argmax + hidden states.

Verification still enforces token identity: a worse draft can only cost
speed. The teacher is Bonsai itself. Hold out the explain-prompt eval.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from monkeyinference.bench import EXPLAIN_PROMPT, FRANCE_PROMPT
from monkeyinference.dflash import (
    AuxCapture,
    DFlashDrafter,
    MASK_TOKEN_ID,
    PROPOSAL_TOKENS,
    load_drafter,
    reset_drafter,
)
from monkeyinference.dflash_lora import save_adapters
from monkeyinference.dflash_prompts import distill_prompts
from monkeyinference.generate import generate
from monkeyinference.hadamard import fwht
from monkeyinference.load import apply_chat, load_text_model

DEFAULT_DIR = Path.home() / ".monkey/dflash-ft2"
BLOCK = 8  # leftover + 7 MASK, matches propose()
LOSS_GAMMA = 4.0  # paper: γ=4 for block size 8
STEP_TIME_ABORT_S = 8.0  # 10k × 8s is a many-hour run; stop and report
CKPT_EVERY = 100
MIN_WINDOWS = 30320  # 10× the overfit run's 3032
AUX_BUDGET_GB = 6.5
MIN_FREE_GB = 3.5

# Held out of the distill set. Success is measured on EXPLAIN_PROMPT.
EVAL_PROMPTS = (EXPLAIN_PROMPT, FRANCE_PROMPT)
DISTILL_PROMPTS: list[tuple[str, int]] = distill_prompts(n=1200)


class FrozenHead:
    """Dequantized Bonsai lm_head so CE can VJP into draft hiddens."""

    def __init__(self, linear):
        self.block = int(linear.block or 0)
        self.signs = linear.signs
        self.weight = mx.dequantize(
            linear.weight,
            linear.scales,
            linear.biases,
            group_size=128,
            bits=2,
        )
        mx.eval(self.weight)

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            x = fwht(x, self.signs, inverse=False)
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        y = flat @ self.weight.astype(mx.float32).T
        return y.reshape(*x.shape[:-1], -1)


def _eos(tokenizer) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(int(i) for i in ids)
    eos = getattr(tokenizer, "eos_token_id", None)
    return {int(eos)} if eos is not None else set()


def collect_sequence(loaded, user: str, max_tokens: int) -> dict:
    """Greedy Bonsai continuation + DFlash aux hiddens. Teacher only."""
    model = loaded.model
    tokenizer = loaded.tokenizer
    prompt = apply_chat(tokenizer, user, enable_thinking=False)
    ids = [int(i) for i in tokenizer.encode(prompt)]
    prompt_len = len(ids)
    eos = _eos(tokenizer)
    aux = AuxCapture(model.model)
    cache = make_prompt_cache(model)

    def body(tok_arr: mx.array):
        hidden = model.model(tok_arr[None], cache=cache)
        aux.record_last_forward()
        return hidden

    tokens = mx.array(ids, dtype=mx.uint32)
    y = tokens
    while y.size > 1:
        n = min(512, int(y.size) - 1)
        mx.eval(body(y[:n]), [c.state for c in cache])
        y = y[n:]
    generated: list[int] = []
    leftover = y
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        hidden = body(leftover)
        logits = model.lm_head(hidden)
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        else:
            logits = logits[-1, :]
        tok_arr = mx.argmax(logits, axis=-1)
        mx.eval(tok_arr, [c.state for c in cache])
        tok = int(tok_arr.reshape(-1)[-1].item())
        generated.append(tok)
        leftover = mx.array([tok], dtype=mx.uint32)
        if tok in eos:
            break
    if leftover.size:
        mx.eval(body(leftover), [c.state for c in cache])
    wall = time.perf_counter() - t0
    all_ids = ids + generated
    ctx = aux.context()
    aux.close()
    if ctx is None or int(ctx.shape[0]) != len(all_ids):
        raise RuntimeError(
            f"aux rows {None if ctx is None else ctx.shape[0]} != tokens {len(all_ids)}"
        )
    return {
        "prompt": user,
        "prompt_len": prompt_len,
        "tokens": all_ids,
        "text": tokenizer.decode(generated),
        "aux": np.array(ctx.astype(mx.float16)),
        "gen_tokens": len(generated),
        "decode_s": wall,
    }


def save_dataset(rows: list[dict], directory: Path) -> dict:
    directory = Path(directory)
    summary = {"n_sequences": 0, "n_windows": 0, "aux_bytes": 0, "directory": str(directory)}
    for i, row in enumerate(rows):
        summary = append_sequence(row, directory, i)
    return summary


def append_sequence(row: dict, directory: Path, index: int) -> dict:
    """Write one teacher sequence. Does not keep aux in RAM after the npy lands."""
    directory = Path(directory)
    aux_dir = directory / "aux"
    aux_dir.mkdir(parents=True, exist_ok=True)
    path = aux_dir / f"{index:04d}.npy"
    np.save(path, row["aux"])
    rec = {
        "prompt": row["prompt"],
        "prompt_len": row["prompt_len"],
        "tokens": row["tokens"],
        "text": row["text"],
        "gen_tokens": row["gen_tokens"],
        "aux": path.name,
        "n_aux": int(row["aux"].shape[0]),
    }
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else []
    if index < len(meta):
        meta[index] = rec
    elif index == len(meta):
        meta.append(rec)
    else:
        raise RuntimeError(f"append index {index} skips meta len {len(meta)}")
    meta_path.write_text(json.dumps(meta, indent=2))
    bytes_aux = sum((aux_dir / m["aux"]).stat().st_size for m in meta)
    n_win = sum(1 for m in meta for _ in iter_windows(m))
    return {
        "n_sequences": len(meta),
        "n_windows": n_win,
        "aux_bytes": bytes_aux,
        "directory": str(directory),
    }


def load_dataset(directory: Path) -> list[dict]:
    """Meta only. Aux is memmapped per window so 6 GB datasets fit next to 27B."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    return [{**m, "aux_path": directory / "aux" / m["aux"]} for m in meta]


def iter_windows(row: dict, *, block_tail: int = PROPOSAL_TOKENS):
    """Assistant-token leftover windows only (not user-prompt internals)."""
    tokens = row["tokens"]
    prompt_len = int(row["prompt_len"])
    t0 = max(prompt_len - 1, 0)
    last = len(tokens) - block_tail - 1
    for t in range(t0, last + 1):
        yield t


def _aux_prefix(row: dict, t: int) -> mx.array:
    if "aux" in row and not isinstance(row["aux"], (str, Path)):
        return mx.array(row["aux"][: t + 1])
    mm = np.load(row["aux_path"], mmap_mode="r")
    return mx.array(np.ascontiguousarray(mm[: t + 1]))


def block_loss(
    drafter: DFlashDrafter,
    embed,
    head: FrozenHead,
    aux: mx.array,
    tokens: list[int],
    t: int,
    *,
    gamma: float = LOSS_GAMMA,
) -> mx.array:
    leftover = int(tokens[t])
    labels = mx.array(tokens[t + 1 : t + 1 + PROPOSAL_TOKENS], dtype=mx.uint32)
    ctx = aux[: t + 1]
    drafter.reset_cache()
    drafter.commit_new(ctx)
    ids = mx.array([leftover] + [MASK_TOKEN_ID] * PROPOSAL_TOKENS, dtype=mx.uint32)
    hidden = embed(ids)
    if hidden.ndim == 3:
        hidden = hidden.reshape(hidden.shape[1], hidden.shape[2])
    h = drafter.backbone(hidden, drafter.ctx_cache)
    logits = head(h)
    pred = logits[1 : 1 + PROPOSAL_TOKENS]
    ce = nn.losses.cross_entropy(pred, labels)
    k = mx.arange(PROPOSAL_TOKENS).astype(mx.float32)
    w = mx.exp(-k / gamma)
    return (ce * w).sum() / w.sum()


def _lr_at(step: int, steps: int, base: float, warmup: int) -> float:
    if steps <= 1:
        return base
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(steps - warmup, 1)
    return 0.1 * base + 0.9 * base * 0.5 * (1.0 + math.cos(math.pi * p))


# Mix-pass T=8 + K=7 draft (docs/ternary-engine.md). Historical 10.22 bar is 3.52.
STOCK_T8_MS = 298.0
STOCK_DRAFT_K7_MS = 46.2
GREEDY_TPS = 10.22
MIN_ACCEPTS_TO_BEAT_GREEDY = GREEDY_TPS * (STOCK_T8_MS + STOCK_DRAFT_K7_MS) / 1000.0
# Clean-compare greedy 10.75 tok/s = 93.0 ms/token. Pass = 298 + 46.2 = 344.2 ms
# = 3.70 greedy-token-equivalents. LoRA draft at ~73 ms would raise this further.
MEASURED_GREEDY_TPS = 10.75
WIN_ACCEPTS = MEASURED_GREEDY_TPS * (STOCK_T8_MS + STOCK_DRAFT_K7_MS) / 1000.0
MIDPOINT_MIN_ACCEPTS = 3.00
STOCK_ACCEPTS = 3.00


def _write_state(out: Path, *, step: int, steps: int, loss: float) -> None:
    payload = {
        "step": int(step),
        "steps": int(steps),
        "loss": float(loss),
        "updated_unix": time.time(),
    }
    (out / "train_state.json").write_text(json.dumps(payload, indent=2))


def read_train_state(out: Path) -> dict | None:
    path = Path(out) / "train_state.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def infer_resume_step(out: Path) -> int:
    """Next train() start_step from on-disk state.

    Prefer train_state.json (`step` is the next index to run). Else parse
    run.log: adapters are saved when step % 200 == 199, so a logged step S
    resumes at (S // 200) * 200 unless that save just landed (S % 200 == 199),
    in which case resume at S + 1. Missing files → 0.
    """
    out = Path(out)
    state = read_train_state(out)
    if state is not None and "step" in state:
        return max(0, int(state["step"]))
    log_path = out / "run.log"
    last = None
    if log_path.is_file():
        import re

        pat = re.compile(r"step (\d+)/(\d+)")
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pat.search(line)
            if m:
                last = int(m.group(1))
    if last is None:
        return 0
    if last % CKPT_EVERY == CKPT_EVERY - 1:
        return last + 1
    return (last // CKPT_EVERY) * CKPT_EVERY


def predicted_k7(
    accepts: float,
    *,
    t8_ms: float = STOCK_T8_MS,
    draft_ms: float = STOCK_DRAFT_K7_MS,
    greedy_tps: float = GREEDY_TPS,
) -> dict:
    """Replay=0 paper row for K=7. Draft/verify ms default to the mix-pass table."""
    pass_ms = float(t8_ms) + float(draft_ms)
    min_accepts = greedy_tps * pass_ms / 1000.0
    pred = float(accepts) / (pass_ms / 1000.0)
    return {
        "k": 7,
        "draft_ms": float(draft_ms),
        "verify_always8_ms": float(t8_ms),
        "pass_ms_replay0": pass_ms,
        "accepts_per_pass": float(accepts),
        "pred_tok_s": pred,
        "min_accepts_to_beat_10_22": min_accepts,
        "wins_on_paper": float(accepts) > min_accepts,
        "greedy_tps": greedy_tps,
        "beats_greedy_margin_tok_s": pred - greedy_tps,
    }


def train(
    drafter: DFlashDrafter,
    adapters: nn.Module,
    embed,
    head: FrozenHead,
    rows: list[dict],
    *,
    steps: int,
    lr: float = 3e-5,
    out: Path,
    start_step: int = 0,
    timed_step_abort_s: float = STEP_TIME_ABORT_S,
    weight_decay: float = 0.1,
    ckpt_every: int = CKPT_EVERY,
    holdout_eval=None,
    midpoint_min: float = MIDPOINT_MIN_ACCEPTS,
    win_accepts: float = WIN_ACCEPTS,
) -> dict:
    windows: list[tuple[int, int]] = []
    for i, row in enumerate(rows):
        for t in iter_windows(row):
            windows.append((i, t))
    if not windows:
        raise RuntimeError("no training windows")
    rng = np.random.default_rng(0)
    for _ in range(max(start_step, 0)):
        rng.integers(0, len(windows))
    opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    warmup = max(20, steps // 25)

    def loss_for(idx_t: tuple[int, int]):
        i, t = idx_t
        row = rows[i]
        aux = _aux_prefix(row, t)
        return block_loss(drafter, embed, head, aux, row["tokens"], t)

    loss_and_grad = nn.value_and_grad(adapters, lambda _m, wt: loss_for(wt))

    losses: list[float] = []
    t_steps: list[float] = []
    ckpt_eval: list[dict] = []
    best_explain = -1.0
    best_step = -1
    log = {
        "n_windows": len(windows),
        "n_trainable": int(adapters.n_trainable),
        "steps": steps,
        "start_step": int(start_step),
        "lr": lr,
        "weight_decay": weight_decay,
        "ckpt_every": ckpt_every,
        "win_accepts": win_accepts,
        "midpoint_min": midpoint_min,
        "aborted": None,
        "killed": None,
        "best_explain_accepts": None,
        "best_step": None,
    }

    def run_holdout(step: int, *, tag: str) -> dict | None:
        nonlocal best_explain, best_step
        if holdout_eval is None:
            return None
        print(f"  === holdout eval step={step} ({tag}) ===", flush=True)
        ev = holdout_eval(step)
        ev = {**ev, "step": step, "tag": tag}
        ckpt_eval.append(ev)
        with (out / "ckpt_eval.jsonl").open("a") as fh:
            fh.write(json.dumps(ev, default=str) + "\n")
        explain = ev.get("explain_accepts")
        mean = ev.get("mean_accepts")
        print(
            f"  holdout step {step}: explain={explain} mean={mean} "
            f"(stock={STOCK_ACCEPTS} win={win_accepts:.2f})",
            flush=True,
        )
        if explain is not None and float(explain) > best_explain:
            best_explain = float(explain)
            best_step = step
            save_adapters(adapters, out / "adapters.best.safetensors")
            (out / "best.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "explain_accepts": best_explain,
                        "mean_accepts": mean,
                        "tag": tag,
                    },
                    indent=2,
                )
            )
            print(f"  new best explain accepts={best_explain:.3f} at step {step}", flush=True)
        return ev

    if start_step == 0:
        save_adapters(adapters, out / "adapters.best.safetensors")
        run_holdout(0, tag="step0_stock")

    midpoint = steps // 2
    midpoint_checked = start_step > midpoint

    for step in range(start_step, steps):
        i, t = windows[int(rng.integers(0, len(windows)))]
        opt.learning_rate = _lr_at(step, steps, lr, warmup)
        mx.synchronize()
        t0 = time.perf_counter()
        loss, grads = loss_and_grad(adapters, (i, t))
        grads, norm = optim.clip_grad_norm(grads, 1.0)
        mx.eval(loss, grads)
        opt.update(adapters, grads)
        mx.eval(adapters.parameters())
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        t_steps.append(elapsed)
        lv = float(loss.item())
        losses.append(lv)
        nrm = float(norm.item()) if hasattr(norm, "item") else float(norm)
        if step == start_step and elapsed > timed_step_abort_s:
            log["aborted"] = (
                f"first step {elapsed:.2f}s > {timed_step_abort_s}s abort; "
                f"projected {steps * elapsed / 3600:.1f} hours"
            )
            save_adapters(adapters, out / "adapters.abort.safetensors")
            _write_state(out, step=step, steps=steps, loss=lv)
            log["first_step_s"] = elapsed
            log["first_loss"] = lv
            return log
        if step % 25 == 0 or step == steps - 1:
            print(
                f"  step {step}/{steps} loss={lv:.4f} {elapsed:.2f}s "
                f"lr={opt.learning_rate:.2e} clip_norm={nrm:.2f}",
                flush=True,
            )
        at_ckpt = step % ckpt_every == ckpt_every - 1 or step == steps - 1
        if at_ckpt:
            mx.clear_cache()
            save_adapters(adapters, out / "adapters.safetensors")
            _write_state(out, step=step + 1, steps=steps, loss=lv)
            ev = run_holdout(step + 1, tag="ckpt")
            if ev is not None and not midpoint_checked and (step + 1) >= midpoint:
                midpoint_checked = True
                explain = ev.get("explain_accepts")
                best = best_explain if best_explain >= 0 else explain
                if best is None or float(best) <= midpoint_min:
                    log["killed"] = (
                        f"midpoint step {step + 1}: best explain accepts="
                        f"{best} not above {midpoint_min}. Stop."
                    )
                    print("STOP:", log["killed"], flush=True)
                    break

    if t_steps:
        save_adapters(adapters, out / "adapters.safetensors")
        _write_state(out, step=log.get("killed") and (step + 1) or steps, steps=steps, loss=losses[-1])
        log["first_step_s"] = t_steps[0]
        log["median_step_s"] = float(sorted(t_steps)[len(t_steps) // 2])
        log["mean_loss_last50"] = float(np.mean(losses[-50:]))
        log["first_loss"] = losses[0]
        log["losses_every_25"] = losses[::25] + ([losses[-1]] if losses else [])
        log["wall_s"] = float(sum(t_steps))
    log["ckpt_eval"] = ckpt_eval
    log["best_explain_accepts"] = None if best_explain < 0 else best_explain
    log["best_step"] = None if best_step < 0 else best_step
    log["win"] = bool(best_explain > win_accepts)
    # Restore the selected checkpoint for the caller.
    best_path = out / "adapters.best.safetensors"
    if best_path.is_file():
        from monkeyinference.dflash_lora import load_adapters

        load_adapters(adapters, best_path)
        save_adapters(adapters, out / "adapters.safetensors")
        print(
            f"  restored best adapters from step {best_step} "
            f"explain={best_explain}",
            flush=True,
        )
    return log


def evaluate_holdout(loaded, prompts: list[tuple[str, int]], *, num_draft: int = 7) -> dict:
    """Accepts/pass on a disjoint set. Weighted by verify_passes."""
    rows = []
    weighted = 0.0
    passes = 0
    explain_acc = None
    explain_draft_ms = None
    for user, ntok in prompts:
        spec = generate(
            loaded,
            user,
            max_tokens=ntok,
            speculative=True,
            draft="dflash",
            num_draft=num_draft,
        )
        acc = spec.accepts_per_pass
        rows.append(
            {
                "prompt": user[:96],
                "accepts_per_pass": acc,
                "verify_passes": spec.verify_passes,
                "tps": spec.generation_tps,
                "draft_s": spec.draft_s,
                "verify_s": spec.verify_s,
            }
        )
        weighted += acc * spec.verify_passes
        passes += spec.verify_passes
        if user == EXPLAIN_PROMPT:
            explain_acc = acc
            if spec.verify_passes:
                explain_draft_ms = 1000.0 * spec.draft_s / spec.verify_passes
    return {
        "mean_accepts": (weighted / passes) if passes else 0.0,
        "explain_accepts": explain_acc,
        "verify_passes": passes,
        "explain_draft_ms_per_pass": explain_draft_ms,
        "rows": rows,
    }


def _reject_hist(prefix) -> dict:
    hist: dict[int, int] = {}
    for n in prefix:
        hist[int(n)] = hist.get(int(n), 0) + 1
    return hist


def _dflash_row(run) -> dict:
    return {
        "tps": run.generation_tps,
        "accepts_per_pass": run.accepts_per_pass,
        "accept_prefix": list(run.accept_prefix),
        "reject_hist": _reject_hist(run.accept_prefix),
        "text": run.text,
        "verify_s": run.verify_s,
        "draft_s": run.draft_s,
        "replay_s": run.replay_s,
        "peak_memory_gb": run.peak_memory_gb,
        "proposed_draft": run.proposed_draft,
        "accepted_draft": run.accepted_draft,
        "verify_passes": run.verify_passes,
        "paper_k7": predicted_k7(run.accepts_per_pass),
    }


def evaluate_explain(loaded, *, num_draft: int = 7) -> dict:
    leftover = generate(
        loaded, EXPLAIN_PROMPT, max_tokens=48, speculative=True, draft="none"
    )
    spec = generate(
        loaded,
        EXPLAIN_PROMPT,
        max_tokens=48,
        speculative=True,
        draft="dflash",
        num_draft=num_draft,
    )
    france = generate(loaded, FRANCE_PROMPT, max_tokens=8, speculative=False)
    identity = list(spec.tokens) == list(leftover.tokens)
    hist = _reject_hist(spec.accept_prefix)
    return {
        "leftover_tps": leftover.generation_tps,
        "dflash_tps": spec.generation_tps,
        "accepts_per_pass": spec.accepts_per_pass,
        "accept_prefix": spec.accept_prefix,
        "reject_hist": hist,
        "identity_vs_leftover": identity,
        "france": france.text,
        "text": spec.text,
        "verify_s": spec.verify_s,
        "draft_s": spec.draft_s,
        "replay_s": spec.replay_s,
        "peak_memory_gb": spec.peak_memory_gb,
        "proposed_draft": spec.proposed_draft,
        "accepted_draft": spec.accepted_draft,
        "verify_passes": spec.verify_passes,
    }


def evaluate_compare(loaded, adapter_path: Path | None, *, num_draft: int = 7) -> dict:
    """Greedy first, then leftover, stock DFlash, optional LoRA adapter.

    One 27B load. Identity is leftover-greedy. Tok/s vs 10.22 is greedy
    in this same process so thermal/LPM is shared across rows.
    """
    print("  greedy", flush=True)
    greedy = generate(loaded, EXPLAIN_PROMPT, max_tokens=48, speculative=False)
    print(f"    {greedy.generation_tps:.2f} tok/s", flush=True)
    print("  leftover", flush=True)
    leftover = generate(
        loaded, EXPLAIN_PROMPT, max_tokens=48, speculative=True, draft="none"
    )
    print(f"    {leftover.generation_tps:.2f} tok/s", flush=True)
    print("  france", flush=True)
    france = generate(loaded, FRANCE_PROMPT, max_tokens=8, speculative=False)
    print(f"    {france.text!r}", flush=True)

    print("  stock DFlash K=7", flush=True)
    reset_drafter()
    load_drafter(adapter_path=False)
    stock = generate(
        loaded,
        EXPLAIN_PROMPT,
        max_tokens=48,
        speculative=True,
        draft="dflash",
        num_draft=num_draft,
    )
    print(
        f"    {stock.generation_tps:.2f} tok/s accepts/pass={stock.accepts_per_pass:.2f}",
        flush=True,
    )

    l_tok = list(leftover.tokens)
    g_tok = list(greedy.tokens)
    s_tok = list(stock.tokens)
    ev = {
        "greedy_tps": greedy.generation_tps,
        "leftover_tps": leftover.generation_tps,
        "france": france.text,
        "greedy_identity_vs_leftover": g_tok == l_tok,
        "stock": _dflash_row(stock),
        "stock_identity_vs_leftover": s_tok == l_tok,
        "adapter": None,
        "adapter_identity_vs_leftover": None,
        "adapter_identity_vs_stock": None,
        "need_accepts_vs_10_22": MIN_ACCEPTS_TO_BEAT_GREEDY,
        "baseline_greedy_tps": GREEDY_TPS,
    }

    path = Path(adapter_path) if adapter_path else None
    if path is not None and path.is_file():
        print(f"  LoRA adapter {path}", flush=True)
        reset_drafter()
        load_drafter(adapter_path=path)
        ft = generate(
            loaded,
            EXPLAIN_PROMPT,
            max_tokens=48,
            speculative=True,
            draft="dflash",
            num_draft=num_draft,
        )
        print(
            f"    {ft.generation_tps:.2f} tok/s accepts/pass={ft.accepts_per_pass:.2f}",
            flush=True,
        )
        f_tok = list(ft.tokens)
        ev["adapter"] = _dflash_row(ft)
        ev["adapter_identity_vs_leftover"] = f_tok == l_tok
        ev["adapter_identity_vs_stock"] = f_tok == s_tok
        ev["dflash_tps"] = ft.generation_tps
        ev["accepts_per_pass"] = ft.accepts_per_pass
        ev["identity_vs_leftover"] = f_tok == l_tok
        ev["text"] = ft.text
        ev["accept_prefix"] = list(ft.accept_prefix)
        ev["reject_hist"] = _reject_hist(ft.accept_prefix)
        ev["paper_k7"] = predicted_k7(ft.accepts_per_pass)
        ev["beats_greedy"] = bool(
            f_tok == l_tok and ft.generation_tps > greedy.generation_tps
        )
        ev["beats_10_22"] = bool(f_tok == l_tok and ft.generation_tps > GREEDY_TPS)
    else:
        ev["dflash_tps"] = stock.generation_tps
        ev["accepts_per_pass"] = stock.accepts_per_pass
        ev["identity_vs_leftover"] = s_tok == l_tok
        ev["text"] = stock.text
        ev["accept_prefix"] = list(stock.accept_prefix)
        ev["reject_hist"] = _reject_hist(stock.accept_prefix)
        ev["paper_k7"] = predicted_k7(stock.accepts_per_pass)
        ev["beats_greedy"] = bool(
            s_tok == l_tok and stock.generation_tps > greedy.generation_tps
        )
        ev["beats_10_22"] = bool(s_tok == l_tok and stock.generation_tps > GREEDY_TPS)
    return ev
