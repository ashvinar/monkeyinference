"""End-to-end bench: speed, numerical parity, coherence, token identity."""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx

from monkeyinference.generate import generate
from monkeyinference.load import load_text_model

STARTING_DECODE_TPS = 8.0
STARTING_PREFILL_TPS = 40.0

COPY_PROMPT = (
    "Copy the next sentence exactly, then stop.\n"
    "Speculative decoding proposes several draft tokens and verifies them in parallel."
)
FRANCE_PROMPT = "Name the capital of France. Reply with only the city name."
EXPLAIN_PROMPT = "Explain speculative decoding in two short sentences."
EARLY_LAYERS_SWEEP = (2, 4, 6, 8)


def _coherence(name: str, text: str) -> dict:
    lowered = text.lower()
    if name == "france":
        return {
            "mentions_paris": "paris" in lowered,
            "looks_garbage": (not text.strip()) or text.count("�") > 3,
            "text": text,
        }
    if name == "copy":
        return {
            "copied_sentence": "speculative decoding proposes several draft tokens" in lowered,
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


def _payload(result) -> dict:
    return result.to_dict()


def _tokens_match(a: list[int] | None, b: list[int] | None) -> bool:
    if not a or not b:
        return False
    return list(a) == list(b)


def run_bench(
    pack: str | Path | None = None,
    *,
    parity_layers: int = 8,
    quick: bool = False,
) -> dict:
    loaded = load_text_model(pack, use_custom_kernels=True)
    jobs = [
        {
            "name": "warmup",
            "user": "Reply with the single word ready.",
            "max_tokens": 8,
            "speculative": False,
            "draft": "none",
            "parity": 0,
        },
        {
            "name": "short_decode",
            "user": FRANCE_PROMPT,
            "max_tokens": 32,
            "speculative": False,
            "draft": "none",
            "parity": parity_layers,
        },
        {
            "name": "paris_leftover",
            "user": FRANCE_PROMPT,
            "max_tokens": 32,
            "speculative": True,
            "draft": "none",
            "parity": 0,
        },
        {
            "name": "explain",
            "user": EXPLAIN_PROMPT,
            "max_tokens": 80,
            "speculative": False,
            "draft": "none",
            "parity": 0,
        },
        {
            "name": "copy_greedy",
            "user": COPY_PROMPT,
            "max_tokens": 40,
            "speculative": True,
            "draft": "none",
            "parity": 0,
        },
        {
            "name": "copy_pld",
            "user": COPY_PROMPT,
            "max_tokens": 40,
            "speculative": True,
            "draft": "pld",
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
            "draft": "none",
            "parity": 0,
        },
        {
            "name": "explain_leftover",
            "user": EXPLAIN_PROMPT,
            "max_tokens": 48,
            "speculative": True,
            "draft": "none",
            "parity": 0,
        },
    ]
    early_ns = (4,)
    for n in early_ns:
        jobs.append(
            {
                "name": f"early_n{n}",
                "user": EXPLAIN_PROMPT,
                "max_tokens": 48,
                "speculative": True,
                "draft": "early",
                "early_layers": n,
                "parity": 0,
            }
        )
    if quick:
        jobs = [j for j in jobs if j["name"] not in {"long_prefill"}]
    runs = {}
    for job in jobs:
        result = generate(
            loaded,
            job["user"],
            max_tokens=job["max_tokens"],
            speculative=job["speculative"],
            draft=job.get("draft", "pld"),
            early_layers=int(job.get("early_layers") or 4),
            parity_layers=job["parity"],
        )
        runs[job["name"]] = _payload(result)

    france = _coherence("france", runs["short_decode"]["text"])
    explain = _coherence("spec", runs["explain"]["text"])
    copy_coh = _coherence("copy", runs["copy_pld"]["text"])
    parity_hits = runs["short_decode"].get("parity") or []
    max_abs = max((h["max_abs"] for h in parity_hits), default=None)
    parity_ok = (max_abs is not None and max_abs < 5e-2) if parity_hits else None

    copy_greedy_tps = runs["copy_greedy"]["generation_tps"]
    copy_pld_tps = runs["copy_pld"]["generation_tps"]
    copy_speedup = (copy_pld_tps / copy_greedy_tps) if copy_greedy_tps else None
    copy_identity = _tokens_match(runs["copy_greedy"]["tokens"], runs["copy_pld"]["tokens"])
    paris_identity = _tokens_match(
        runs["short_decode"]["tokens"], runs["paris_leftover"]["tokens"]
    )

    early_rows = {}
    early_best = None
    leftover_tokens = runs.get("explain_leftover", {}).get("tokens")
    for n in early_ns:
        row = runs.get(f"early_n{n}")
        if not row:
            continue
        rec = {
            "early_layers": n,
            "generation_tps": row["generation_tps"],
            "accepts_per_pass": row["accepts_per_pass"],
            "accepted_draft": row["accepted_draft"],
            "proposed_draft": row["proposed_draft"],
            "verify_passes": row["verify_passes"],
            "generation_tokens": row["generation_tokens"],
            "draft_s": row.get("draft_s"),
            "verify_s": row.get("verify_s"),
            "replay_s": row.get("replay_s"),
            "token_identity": _tokens_match(leftover_tokens, row.get("tokens")),
            "text": row["text"],
        }
        early_rows[n] = rec
        if early_best is None or rec["accepts_per_pass"] > early_best["accepts_per_pass"]:
            early_best = rec
    early_plateau = None
    if early_best is not None:
        early_plateau = early_best["accepts_per_pass"] < 2.0

    decode_tps = runs["explain"]["generation_tps"]
    prefill_tps = (runs.get("long_prefill") or runs["explain"])["prompt_tps"]
    coherence_ok = (
        france.get("mentions_paris")
        and not france.get("looks_garbage")
        and explain.get("mentions_speculative_ideas")
        and not explain.get("looks_garbage")
        and copy_coh.get("copied_sentence")
        and not copy_coh.get("looks_garbage")
    )
    pld_beats_greedy = bool(copy_speedup and copy_speedup > 1.15)
    identity_ok = bool(copy_identity and paris_identity)
    early_identity_ok = all(r["token_identity"] for r in early_rows.values()) if early_rows else True
    report = {
        "ok": True,
        "pack": str(loaded.pack),
        "use_custom_kernels": loaded.use_custom_kernels,
        "language_gb": loaded.language_bytes / 1e9,
        "skipped_vision_gb": loaded.skipped_vision_bytes / 1e9,
        "runs": runs,
        "coherence": {"france": france, "explain": explain, "copy": copy_coh},
        "coherence_ok": bool(coherence_ok),
        "parity_max_abs": max_abs,
        "parity_ok": parity_ok,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "starting_decode_tps": STARTING_DECODE_TPS,
        "starting_prefill_tps": STARTING_PREFILL_TPS,
        "decode_vs_start": decode_tps / STARTING_DECODE_TPS if STARTING_DECODE_TPS else None,
        "copy_prompt": {
            "greedy_tps": copy_greedy_tps,
            "pld_tps": copy_pld_tps,
            "speedup": copy_speedup,
            "pld_beats_greedy": pld_beats_greedy,
            "accepted": runs["copy_pld"]["accepted_draft"],
            "proposed": runs["copy_pld"]["proposed_draft"],
            "accepts_per_pass": runs["copy_pld"]["accepts_per_pass"],
            "token_identity": copy_identity,
            "greedy_text": runs["copy_greedy"]["text"],
            "pld_text": runs["copy_pld"]["text"],
        },
        "paris_token_identity": paris_identity,
        "early_exit": {
            "sweep": early_rows,
            "best": early_best,
            "plateau_below_2x": early_plateau,
            "token_identity_ok": early_identity_ok,
        },
        "identity_ok": bool(identity_ok and early_identity_ok),
        "gates_ok": bool(
            coherence_ok and identity_ok and early_identity_ok and pld_beats_greedy
        ),
    }
    del loaded
    gc.collect()
    mx.clear_cache()
    return report


def run_dflash_bench(pack: str | Path | None = None) -> dict:
    """Explain-prompt accepts/pass for the Splash DFlash 2 draft vs leftover greedy."""
    loaded = load_text_model(pack, use_custom_kernels=True)
    leftover = generate(
        loaded,
        EXPLAIN_PROMPT,
        max_tokens=48,
        speculative=True,
        draft="none",
        parity_layers=0,
    )
    dflash = generate(
        loaded,
        EXPLAIN_PROMPT,
        max_tokens=48,
        speculative=True,
        draft="dflash",
        parity_layers=0,
    )
    identity = _tokens_match(leftover.tokens, dflash.tokens)
    coh = _coherence("spec", dflash.text)
    from pathlib import Path as _P

    draft_dir = _P.home() / ".monkey/models/Qwen3.8-27B-Splash-draft/draft"
    on_disk = sum(p.stat().st_size for p in draft_dir.glob("*.bin")) if draft_dir.is_dir() else 0
    report = {
        "ok": True,
        "prompt": EXPLAIN_PROMPT,
        "draft_dir": str(draft_dir),
        "draft_bytes": on_disk,
        "draft_gb": on_disk / 1e9,
        "leftover": _payload(leftover),
        "dflash": _payload(dflash),
        "accepts_per_pass": dflash.accepts_per_pass,
        "accepted_draft": dflash.accepted_draft,
        "proposed_draft": dflash.proposed_draft,
        "verify_passes": dflash.verify_passes,
        "token_identity": identity,
        "identity_ok": identity,
        "coherence": coh,
        "beats_early_exit": dflash.accepts_per_pass > 1.2,
        "usable": dflash.accepts_per_pass >= 2.0 and identity,
    }
    del loaded
    gc.collect()
    mx.clear_cache()
    return report
