"""Generation: greedy decode via mlx_lm, plus verified speculative decode."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from monkeyinference.load import LoadedModel, apply_chat
from monkeyinference.parity import parity_report, reset_parity
from monkeyinference.spec import Drafter, PromptLookupDrafter, restore_cache, snapshot_cache


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
    parity: list[dict] = field(default_factory=list)
    peak_memory_gb: float | None = None


def _eos_set(tokenizer) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(int(i) for i in ids)
    eos = getattr(tokenizer, "eos_token_id", None)
    return {int(eos)} if eos is not None else set()


def generate(
    loaded: LoadedModel,
    user: str,
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    speculative: bool = False,
    num_draft: int = 5,
    ngram: int = 3,
    parity_layers: int = 0,
    prefill_step: int = 512,
) -> GenerateResult:
    model = loaded.model
    tokenizer = loaded.tokenizer
    prompt = apply_chat(tokenizer, user, enable_thinking=enable_thinking)

    if parity_layers:
        reset_parity(parity_layers)
    if hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()

    t0 = time.perf_counter()
    if speculative:
        result = _speculative(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            num_draft=num_draft,
            ngram=ngram,
            prefill_step=prefill_step,
        )
    else:
        if temperature != 0.0:
            raise ValueError("only greedy (temperature=0) is wired; sampling is a later milestone")
        result = _greedy_mlx(model, tokenizer, prompt, max_tokens=max_tokens)
    result.wall_s = time.perf_counter() - t0
    result.parity = parity_report()
    if hasattr(mx.metal, "get_peak_memory"):
        result.peak_memory_gb = mx.metal.get_peak_memory() / 1e9
    return result


def _greedy_mlx(model, tokenizer, prompt: str, *, max_tokens: int) -> GenerateResult:
    sampler = make_sampler(temp=0.0)
    text = []
    last = None
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
    ):
        text.append(response.text)
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
        )
    return GenerateResult(
        text="".join(text),
        prompt_tokens=int(last.prompt_tokens),
        generation_tokens=int(last.generation_tokens),
        prompt_tps=float(last.prompt_tps),
        generation_tps=float(last.generation_tps),
        wall_s=0.0,
        finish_reason=last.finish_reason or "length",
        peak_memory_gb=float(last.peak_memory) if last.peak_memory is not None else None,
    )


def _speculative(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    num_draft: int,
    ngram: int,
    prefill_step: int,
) -> GenerateResult:
    tokens = mx.array(tokenizer.encode(prompt), dtype=mx.uint32)
    eos = _eos_set(tokenizer)
    cache = make_prompt_cache(model)
    t_pre = time.perf_counter()
    y = tokens
    while y.size > 1:
        n = min(prefill_step, int(y.size) - 1)
        model(y[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        y = y[n:]
        mx.clear_cache()
    mx.eval(y)
    prefill_s = time.perf_counter() - t_pre
    prompt_tokens = int(tokens.size)

    drafter: Drafter = PromptLookupDrafter(ngram=ngram, max_draft=num_draft)
    generated: list[int] = []
    accepted_draft = 0
    proposed_draft = 0
    context_ids = tokens.tolist()
    finish = "length"
    t_decode = time.perf_counter()

    def greedy_from_logits(logits: mx.array) -> int:
        return int(mx.argmax(logits, axis=-1).item())

    while len(generated) < max_tokens:
        remaining = max_tokens - len(generated)
        draft = drafter.propose(
            context_ids, max_tokens=min(num_draft, max(remaining - 1, 0))
        )
        proposed_draft += len(draft)
        if not draft:
            logits = model(y[None], cache=cache)
            mx.eval(logits)
            tok = greedy_from_logits(logits[:, -1, :])
            generated.append(tok)
            context_ids.append(tok)
            y = mx.array([tok], dtype=mx.uint32)
            if tok in eos:
                finish = "stop"
                break
            continue

        snap = snapshot_cache(cache)
        y_run = mx.concatenate([y, mx.array(draft, dtype=mx.uint32)])
        logits = model(y_run[None], cache=cache)
        mx.eval(logits)
        pred_list = mx.argmax(logits, axis=-1).reshape(-1).tolist()
        n_accept = 0
        for i, dtok in enumerate(draft):
            if int(pred_list[i]) != int(dtok):
                break
            n_accept += 1
        accepted_draft += n_accept
        bonus = int(pred_list[n_accept])
        if n_accept < len(draft):
            restore_cache(cache, snap)
            replay = [int(y.reshape(-1)[0].item())] + draft[:n_accept]
            model(mx.array(replay, dtype=mx.uint32)[None], cache=cache)
            mx.eval([c.state for c in cache])
        stop = False
        for dtok in draft[:n_accept]:
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
    )
