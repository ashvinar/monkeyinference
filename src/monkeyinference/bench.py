"""End-to-end bench: speed, numerical parity, coherence on real prompts."""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx

from monkeyinference.generate import generate
from monkeyinference.load import load_text_model

STARTING_DECODE_TPS = 8.0
STARTING_PREFILL_TPS = 40.0


def _coherence(name: str, text: str) -> dict:
    lowered = text.lower()
    if name == "france":
        return {
            "mentions_paris": "paris" in lowered,
            "looks_garbage": (not text.strip()) or text.count("�") > 3,
            "text": text,
        }
    return {
        "mentions_speculative_ideas": any(
            w in lowered for w in ("speculative", "draft", "token", "verify", "predict")
        ),
        "looks_garbage": (not text.strip()) or text.count("�") > 3 or len(set(text.replace(" ", ""))) < 8,
        "text": text,
    }


def run_bench(pack: str | Path | None = None, *, parity_layers: int = 8, quick: bool = False) -> dict:
    loaded = load_text_model(pack, use_custom_kernels=False)
    jobs = [
        {
            "name": "warmup",
            "user": "Reply with the single word ready.",
            "max_tokens": 8,
            "speculative": False,
            "parity": 0,
        },
        {
            "name": "short_decode",
            "user": "Name the capital of France. Reply with only the city name.",
            "max_tokens": 32,
            "speculative": False,
            "parity": parity_layers,
        },
        {
            "name": "explain",
            "user": "Explain speculative decoding in two short sentences.",
            "max_tokens": 80,
            "speculative": False,
            "parity": 0,
        },
        {
            "name": "long_prefill",
            "user": (
                "Read the following notes, then write two sentences explaining speculative decoding.\n\n"
                + (
                    "Speculative decoding proposes several draft tokens and verifies them "
                    "in parallel against the target model. "
                )
                * 40
            ),
            "max_tokens": 64,
            "speculative": False,
            "parity": 0,
        },
        {
            "name": "spec_pld",
            "user": (
                "Copy the next sentence exactly, then stop.\n"
                "Speculative decoding proposes several draft tokens and verifies them in parallel."
            ),
            "max_tokens": 40,
            "speculative": True,
            "parity": 0,
        },
    ]
    if quick:
        jobs = [j for j in jobs if j["name"] not in {"long_prefill"}]
    for job in jobs:
        result = generate(
            loaded,
            job["user"],
            max_tokens=job["max_tokens"],
            speculative=job["speculative"],
            parity_layers=job["parity"],
        )
        runs[job["name"]] = result.__dict__.copy()

    france = _coherence("france", runs["short_decode"]["text"])
    explain = _coherence("spec", runs["explain"]["text"])
    parity_hits = runs["short_decode"].get("parity") or []
    max_abs = max((h["max_abs"] for h in parity_hits), default=None)
    # fp16 GEMV vs affine reference: a few ulps is expected; 5e-2 would mean a real bug
    parity_ok = (max_abs is not None and max_abs < 5e-2) if parity_hits else None

    decode_tps = runs["explain"]["generation_tps"]
    prefill_tps = (runs.get("long_prefill") or runs["explain"])["prompt_tps"]
    coherence_ok = (
        france.get("mentions_paris")
        and not france.get("looks_garbage")
        and explain.get("mentions_speculative_ideas")
        and not explain.get("looks_garbage")
    )
    report = {
        "ok": True,
        "pack": str(loaded.pack),
        "use_custom_kernels": loaded.use_custom_kernels,
        "language_gb": loaded.language_bytes / 1e9,
        "skipped_vision_gb": loaded.skipped_vision_bytes / 1e9,
        "runs": runs,
        "coherence": {"france": france, "explain": explain},
        "coherence_ok": bool(coherence_ok),
        "parity_max_abs": max_abs,
        "parity_ok": parity_ok,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "starting_decode_tps": STARTING_DECODE_TPS,
        "starting_prefill_tps": STARTING_PREFILL_TPS,
        "decode_vs_start": decode_tps / STARTING_DECODE_TPS if STARTING_DECODE_TPS else None,
        "spec_pld": {
            "accepted": runs["spec_pld"]["accepted_draft"],
            "proposed": runs["spec_pld"]["proposed_draft"],
            "generation_tps": runs["spec_pld"]["generation_tps"],
            "text": runs["spec_pld"]["text"],
        },
    }
    del loaded
    gc.collect()
    mx.clear_cache()
    return report
