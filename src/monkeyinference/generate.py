"""Generation: greedy decode via mlx_lm, plus verified speculative decode."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from monkeyinference.load import LoadedModel, apply_chat
from monkeyinference.parity import parity_report, reset_parity
from monkeyinference.spec import (
    EarlyExitDrafter,
    PromptLookupDrafter,
    pin_cache,
    revert_cache,
)

DRAFT_KINDS = ("none", "pld", "early")


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    wall_s: float
    finish_reason: str
    accepted_draft: int = 0
    proposed_draft: int = 0
    verify_passes: int = 0
    draft_kind: str = "none"
    early_layers: int | None = None
    tokens: list[int] = field(default_factory=list)
    parity: list[dict] = field(default_factory=list)
    peak_memory_gb: float | None = None
    draft_s: float = 0.0
    verify_s: float = 0.0
    replay_s: float = 0.0

    @property
    def accepts_per_pass(self) -> float:
        """Mean committed tokens (accepted draft + bonus) per target verify/greedy step."""
        if self.verify_passes <= 0:
            return 0.0
        return self.generation_tokens / self.verify_passes

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["accepts_per_pass"] = self.accepts_per_pass
        return payload


def _eos_set(tokenizer) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(int(i) for i in ids)
    eos = getattr(tokenizer, "eos_token_id", None)
    return {int(eos)} if eos is not None else set()


def _prefill(forward, tokens: mx.array, cache, step: int) -> mx.array:
    y = tokens
    while y.size > 1:
        n = min(step, int(y.size) - 1)
        out = forward(y[:n][None], cache)
        mx.eval(out, [c.state for c in cache])
        y = y[n:]
        mx.clear_cache()
    return y


def _default_num_draft(draft: str) -> int:
    # PLD is free; K=5 was 10/10 on the copy prompt. Larger K over-proposes and
    # pays a reject+replay. Early-exit keeps K small (lm_head every draft token).
    if draft == "pld":
        return 5
    if draft == "early":
        return 4
    return 0


def generate(
    loaded: LoadedModel,
    user: str,
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    speculative: bool = False,
    draft: str = "pld",
    num_draft: int | None = None,
    ngram: int = 3,
    early_layers: int = 4,
    parity_layers: int = 0,
    prefill_step: int = 512,
) -> GenerateResult:
    model = loaded.model
    tokenizer = loaded.tokenizer
    prompt = apply_chat(tokenizer, user, enable_thinking=enable_thinking)
    draft = draft or "pld"
    if draft not in DRAFT_KINDS:
        raise ValueError(f"unknown draft {draft!r}")
    if num_draft is None:
        num_draft = _default_num_draft(draft)

    if parity_layers:
        reset_parity(parity_layers)
    reset = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
    if callable(reset):
        reset()

    t0 = time.perf_counter()
    if speculative:
        result = _speculative(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            draft_kind=draft,
            num_draft=num_draft,
            ngram=ngram,
            early_layers=early_layers,
            prefill_step=prefill_step,
        )
    else:
        if temperature != 0.0:
            raise ValueError("only greedy (temperature=0) is wired; sampling is a later milestone")
        result = _greedy_mlx(model, tokenizer, prompt, max_tokens=max_tokens)
    result.wall_s = time.perf_counter() - t0
    result.parity = parity_report()
    peak_fn = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
    if callable(peak_fn):
        result.peak_memory_gb = peak_fn() / 1e9
    return result


def _greedy_mlx(model, tokenizer, prompt: str, *, max_tokens: int) -> GenerateResult:
    sampler = make_sampler(temp=0.0)
    text = []
    tokens: list[int] = []
    last = None
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
    ):
        text.append(response.text)
        tokens.append(int(response.token))
        last = response
    if last is None:
        return GenerateResult(
            text="",
            prompt_tokens=0,
            generation_tokens=0,
            prompt_tps=0.0,
            generation_tps=0.0,
            wall_s=0.0,
            finish_reason="stop",
            draft_kind="none",
        )
    return GenerateResult(
        text="".join(text),
        prompt_tokens=int(last.prompt_tokens),
        generation_tokens=int(last.generation_tokens),
        prompt_tps=float(last.prompt_tps),
        generation_tps=float(last.generation_tps),
        wall_s=0.0,
        finish_reason=last.finish_reason or "length",
        verify_passes=int(last.generation_tokens),
        draft_kind="none",
        tokens=tokens,
        peak_memory_gb=float(last.peak_memory) if last.peak_memory is not None else None,
    )


def _speculative(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    draft_kind: str,
    num_draft: int,
    ngram: int,
    early_layers: int,
    prefill_step: int,
) -> GenerateResult:
    tokens = mx.array(tokenizer.encode(prompt), dtype=mx.uint32)
    eos = _eos_set(tokenizer)
    target_cache = make_prompt_cache(model)

    early: EarlyExitDrafter | None = None
    draft_cache = None
    if draft_kind == "early":
        early = EarlyExitDrafter(model, early_layers)
        draft_cache = early.make_cache()
    elif draft_kind not in ("pld", "none"):
        raise ValueError(f"unknown draft {draft_kind!r}")

    def target_forward(ids, cache):
        return model(ids, cache=cache)

    t_pre = time.perf_counter()
    y = _prefill(target_forward, tokens, target_cache, prefill_step)
    if early is not None:
        _prefill(lambda ids, cache: early.logits(ids, cache), tokens, draft_cache, prefill_step)
    mx.eval(y)
    prefill_s = time.perf_counter() - t_pre
    prompt_tokens = int(tokens.size)

    pld = PromptLookupDrafter(ngram=ngram, max_draft=num_draft)
    generated: list[int] = []
    accepted_draft = 0
    proposed_draft = 0
    verify_passes = 0
    draft_s = 0.0
    verify_s = 0.0
    replay_s = 0.0
    context_ids = tokens.tolist()
    finish = "length"
    t_decode = time.perf_counter()

    while len(generated) < max_tokens:
        remaining = max_tokens - len(generated)
        k = min(num_draft, max(remaining - 1, 0))
        draft_ids: list[int] = []
        draft_pin = None
        if k > 0 and draft_kind == "pld":
            t_d = time.perf_counter()
            draft_ids = pld.propose(context_ids, max_tokens=k)
            draft_s += time.perf_counter() - t_d
        elif k > 0 and early is not None:
            t_d = time.perf_counter()
            draft_pin = pin_cache(draft_cache)
            draft_ids = early.propose_from_state(y, draft_cache, k, eos)
            draft_s += time.perf_counter() - t_d

        proposed_draft += len(draft_ids)
        if not draft_ids:
            leftover = y
            t_v = time.perf_counter()
            logits = model(leftover[None], cache=target_cache)
            mx.eval(logits)
            verify_s += time.perf_counter() - t_v
            tok = int(mx.argmax(logits[:, -1, :]).item())
            verify_passes += 1
            generated.append(tok)
            context_ids.append(tok)
            if early is not None:
                # Target consumed leftover; draft is still sitting on it.
                mx.eval(early.logits(leftover, draft_cache))
            y = mx.array([tok], dtype=mx.uint32)
            if tok in eos:
                finish = "stop"
                break
            continue

        # One target forward over leftover + all draft tokens.
        # PackedLinear flattens [1, K, H] → [K, H] so MLX sees a 2-D matmul.
        target_pin = pin_cache(target_cache)
        y_run = mx.concatenate([y, mx.array(draft_ids, dtype=mx.uint32)])
        t_v = time.perf_counter()
        logits = model(y_run[None], cache=target_cache)
        mx.eval(logits)
        verify_s += time.perf_counter() - t_v
        verify_passes += 1
        pred = [int(v) for v in mx.argmax(logits, axis=-1).reshape(-1).tolist()]
        n_accept = 0
        for i, dtok in enumerate(draft_ids):
            if pred[i] != int(dtok):
                break
            n_accept += 1
        accepted_draft += n_accept
        bonus = int(pred[n_accept])

        if n_accept < len(draft_ids):
            t_r = time.perf_counter()
            revert_cache(target_pin)
            replay = (
                mx.concatenate([y, mx.array(draft_ids[:n_accept], dtype=mx.uint32)])
                if n_accept
                else y
            )
            mx.eval(model(replay[None], cache=target_cache))
            replay_s += time.perf_counter() - t_r

        if early is not None and draft_pin is not None:
            if n_accept < len(draft_ids):
                t_r = time.perf_counter()
                revert_cache(draft_pin)
                replay = (
                    mx.concatenate([y, mx.array(draft_ids[:n_accept], dtype=mx.uint32)])
                    if n_accept
                    else y
                )
                mx.eval(early.logits(replay, draft_cache))
                replay_s += time.perf_counter() - t_r
            else:
                mx.eval(early.logits(mx.array([draft_ids[-1]], dtype=mx.uint32), draft_cache))

        stop = False
        for dtok in draft_ids[:n_accept]:
            generated.append(int(dtok))
            context_ids.append(int(dtok))
            if int(dtok) in eos or len(generated) >= max_tokens:
                finish = "stop" if int(dtok) in eos else finish
                stop = True
                break
        if stop:
            break
        generated.append(bonus)
        context_ids.append(bonus)
        y = mx.array([bonus], dtype=mx.uint32)
        if bonus in eos:
            finish = "stop"
            break

    decode_s = time.perf_counter() - t_decode
    text = tokenizer.decode(generated, skip_special_tokens=True)
    n_gen = len(generated)
    return GenerateResult(
        text=text,
        prompt_tokens=prompt_tokens,
        generation_tokens=n_gen,
        prompt_tps=prompt_tokens / prefill_s if prefill_s else 0.0,
        generation_tps=n_gen / decode_s if decode_s and n_gen else 0.0,
        wall_s=0.0,
        finish_reason=finish,
        accepted_draft=accepted_draft,
        proposed_draft=proposed_draft,
        verify_passes=verify_passes,
        draft_kind=draft_kind,
        early_layers=early.n_layers if early is not None else None,
        tokens=generated,
        draft_s=draft_s,
        verify_s=verify_s,
        replay_s=replay_s,
    )
