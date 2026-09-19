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
)
from monkeyinference.dflash_lora import save_adapters
from monkeyinference.generate import generate
from monkeyinference.hadamard import fwht
from monkeyinference.load import apply_chat, load_text_model

DEFAULT_DIR = Path.home() / ".monkey/dflash-ft"
BLOCK = 8  # leftover + 7 MASK, matches propose()
LOSS_GAMMA = 4.0  # paper: γ=4 for block size 8
STEP_TIME_ABORT_S = 8.0  # 10k × 8s is a many-hour run; stop and report

# Held out of the distill set. Success is measured on EXPLAIN_PROMPT.
EVAL_PROMPTS = (EXPLAIN_PROMPT, FRANCE_PROMPT)

DISTILL_PROMPTS: list[tuple[str, int]] = [
    # (user prompt, max new tokens). Chat / explanation / code.
    ("What is speculative decoding, and why does it help on a memory-bound GPU?", 64),
    ("Explain residual connections in a transformer in two short paragraphs.", 64),
    ("Give a practical checklist for reviewing a Python pull request.", 64),
    ("How does a Hadamard transform differ from a DFT? Keep it concrete.", 48),
    ("Write a Python function that merges two sorted lists without allocating extra lists beyond the output.", 80),
    ("Explain gated delta networks as if I have used Mamba but not GDN.", 64),
    ("What should I pack for a three-day hiking trip in the Cascades in September?", 48),
    ("Debug this: a Metal kernel compiles but returns zeros. Where do you look first?", 64),
    ("Summarize copy-on-write vs memcpy for a recurrent state cache.", 48),
    ("Write a bash loop that prints the ten largest files under a directory.", 48),
    ("Why might 2-bit quantization change a language model's next-token distribution?", 64),
    ("Explain DRAM bandwidth vs compute roofline for a 27B GEMV.", 64),
    ("Draft a polite email asking a teammate to review a performance regression.", 48),
    ("Implement binary search in Python over a list of integers, with tests in comments.", 80),
    ("What is block diffusion, in the sense used by speculative draft models?", 64),
    ("How do I keep a MacBook Air cool while running a long GPU job?", 40),
    ("Explain token identity as a correctness gate for speculative decoding.", 48),
    ("Write a regex that matches IPv4 addresses and mention a false-positive.", 48),
    ("Compare LoRA and full fine-tuning when the base weights are 4-bit.", 64),
    ("What's a good way to structure a living design doc for a systems project?", 48),
    ("Explain why fp16 residuals can break a draft model when activations are ~1e4.", 64),
    ("Write a Python dataclass for a generate() result with tok/s fields.", 64),
    ("How does prompt-lookup decoding differ from a trained drafter?", 48),
    ("Give three reasons a Q4 GEMM can beat a 2-bit GEMV on small-M batches.", 64),
    ("Walk through git rebase vs merge for a long-lived feature branch.", 48),
    ("Write a SQL query that counts orders per day for the last 14 days.", 40),
    ("What is a threadgroup in Metal, and why does 256 threads matter for MMA?", 64),
    ("Explain the difference between accepts/pass and tokens per second.", 48),
    ("Implement FizzBuzz in Rust, then in Python, both under 20 lines.", 64),
    ("How would you teach a new hire to read a STREAM benchmark?", 48),
    ("Describe a chat-app feature that summarizes a long thread in one paragraph.", 48),
    ("Why freeze a quantized backbone and train adapters instead of dequantizing?", 64),
    ("Write a Makefile target that runs unit tests and a short bench.", 40),
    ("Explain KV cache pins versus rebuilding the cache after a rejected draft.", 64),
    ("What's the difference between greedy decoding and sampling at temp 1.0?", 48),
    ("Give a short architecture review of a 5-layer draft sitting under a 64-layer target.", 64),
    ("Write a Python generator that yields sliding windows of length 8.", 48),
    ("How do I interpret a front-loaded rejection histogram in speculative decoding?", 64),
    ("Explain unified memory on Apple silicon in one paragraph for a CUDA person.", 48),
    ("Draft a README section: hardware assumptions and honest performance numbers.", 64),
    ("What does it mean for a pack to be lossless-ternary but slower to unpack?", 48),
    ("Write a function that computes cross-entropy of logits vs integer targets in numpy.", 64),
    ("How should I choose K for a block-diffusion drafter that always runs 8 query rows?", 48),
    ("Explain why leftover-greedy is the right identity baseline for speculation.", 48),
    ("A user asks: is my laptop slow or is the model just big? Answer carefully.", 48),
    ("Write a pytest for a function that packs 5 ternary codes per byte.", 64),
    ("What is an occupancy artifact in a GEMV microbench?", 48),
    ("Describe a good on-call handoff note after a failed training run.", 40),
    ("Compare attention and GDN layers in Qwen3.5-style models at a high level.", 64),
    ("Write a Python CLI with argparse for collect/train/eval subcommands.", 64),
    ("Why might a draft be right on token 1 and collapse after?", 48),
    ("Explain cosine LR with warmup without formulas first, then give the formula.", 48),
    ("How do you keep secrets out of a git repo that has a results/ folder?", 40),
    ("Implement a ring buffer of token ids in Python with a fixed capacity.", 48),
    ("What should a performance claim include besides tok/s?", 48),
    ("Give an example of a chat turn that asks for a refactor, then one that asks for an explanation.", 48),
    ("Why is AdamW state so large compared to the trainable parameter count?", 48),
    ("Write a short comment-only walkthrough of rms_norm in float32.", 40),
    ("How would you estimate wall-clock for 5000 GPU training steps from one timed step?", 40),
    ("Explain the difference between a seed main branch and a feature branch on GitHub.", 40),
    ("A teammate inverted a speedup ratio. Write the correction without being snide.", 40),
    ("What is the smallest experiment that would tell you a drafter adapted to a quantized target?", 48),
    ("Write a JSON schema for a break-even table row (k, draft_ms, pred_tok_s).", 40),
    ("Explain why you would not delete someone else's GGUF to free training disk.", 40),
]


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
    aux_dir = directory / "aux"
    aux_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    bytes_aux = 0
    for i, row in enumerate(rows):
        path = aux_dir / f"{i:04d}.npy"
        np.save(path, row["aux"])
        bytes_aux += path.stat().st_size
        meta.append(
            {
                "prompt": row["prompt"],
                "prompt_len": row["prompt_len"],
                "tokens": row["tokens"],
                "text": row["text"],
                "gen_tokens": row["gen_tokens"],
                "aux": str(path.name),
                "n_aux": int(row["aux"].shape[0]),
            }
        )
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    n_win = sum(
        max(0, len(m["tokens"]) - 7 - max(m["prompt_len"] - 1, 0)) for m in meta
    )
    return {
        "n_sequences": len(meta),
        "n_windows": n_win,
        "aux_bytes": bytes_aux,
        "directory": str(directory),
    }


def load_dataset(directory: Path) -> list[dict]:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    rows = []
    for m in meta:
        aux = np.load(directory / "aux" / m["aux"])
        rows.append({**m, "aux": aux})
    return rows


def iter_windows(row: dict, *, block_tail: int = PROPOSAL_TOKENS):
    tokens = row["tokens"]
    prompt_len = int(row["prompt_len"])
    t0 = max(prompt_len - 1, 0)
    last = len(tokens) - block_tail - 1
    for t in range(t0, last + 1):
        yield t


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


CKPT_EVERY = 200  # adapters land on step % CKPT_EVERY == CKPT_EVERY-1
# Mix-pass verify + K=7 draft from docs/ternary-engine.md (2026-09-19).
STOCK_T8_MS = 298.0
STOCK_DRAFT_K7_MS = 46.2
GREEDY_TPS = 10.22
MIN_ACCEPTS_TO_BEAT_GREEDY = GREEDY_TPS * (STOCK_T8_MS + STOCK_DRAFT_K7_MS) / 1000.0


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
    lr: float = 1e-4,
    out: Path,
    start_step: int = 0,
    timed_step_abort_s: float = STEP_TIME_ABORT_S,
) -> dict:
    windows: list[tuple[int, int]] = []
    for i, row in enumerate(rows):
        for t in iter_windows(row):
            windows.append((i, t))
    if not windows:
        raise RuntimeError("no training windows")
    rng = np.random.default_rng(0)
    # Advance the window RNG to match a resumed step so we do not
    # repeat the same prefix of samples.
    for _ in range(max(start_step, 0)):
        rng.integers(0, len(windows))
    opt = optim.AdamW(learning_rate=lr)
    warmup = max(20, steps // 25)

    def loss_for(idx_t: tuple[int, int]):
        i, t = idx_t
        row = rows[i]
        aux = mx.array(row["aux"])
        return block_loss(drafter, embed, head, aux, row["tokens"], t)

    loss_and_grad = nn.value_and_grad(adapters, lambda _m, wt: loss_for(wt))

    losses: list[float] = []
    t_steps: list[float] = []
    log = {
        "n_windows": len(windows),
        "n_trainable": int(adapters.n_trainable),
        "steps": steps,
        "start_step": int(start_step),
        "aborted": None,
    }

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
        if step % CKPT_EVERY == CKPT_EVERY - 1 or step == steps - 1:
            mx.clear_cache()
            save_adapters(adapters, out / "adapters.safetensors")
            _write_state(out, step=step + 1, steps=steps, loss=lv)

    save_adapters(adapters, out / "adapters.safetensors")
    _write_state(out, step=steps, steps=steps, loss=losses[-1] if losses else 0.0)
    log["first_step_s"] = t_steps[0]
    log["median_step_s"] = float(sorted(t_steps)[len(t_steps) // 2])
    log["mean_loss_last50"] = float(np.mean(losses[-50:]))
    log["first_loss"] = losses[0]
    log["losses_every_25"] = losses[::25] + ([losses[-1]] if losses else [])
    log["wall_s"] = float(sum(t_steps))
    return log


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
    hist = {}
    for n in spec.accept_prefix:
        hist[int(n)] = hist.get(int(n), 0) + 1
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
