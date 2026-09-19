"""Load the Bonsai MLX pack as a text-only Qwen3.5 model.

Skips the 0.92 GB FP16 vision tower. Replaces Linear/Embedding modules listed
in config.json with PackedLinear / PackedEmbedding so the Hadamard contract
is applied. Ordinary mlx_lm loaders skip that transform and emit garbage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tokenizer_utils import load as load_mlx_tokenizer

from monkeyinference.packed import PackedEmbedding, PackedLinear

DEFAULT_PACK = Path.home() / ".monkey/models/Ternary-Bonsai-2-27B-mlx-2bit"

LANGUAGE_PREFIX = "language_model."


@dataclass
class LoadedModel:
    model: TextModel
    tokenizer: TokenizerWrapper
    config: dict
    pack: Path
    use_custom_kernels: bool
    language_bytes: int
    skipped_vision_bytes: int
    mix_five_trit: dict | None = None


def _metal_stats() -> dict:
    stats = {}
    for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
        fn = getattr(mx.metal, name, None)
        if callable(fn):
            stats[name.replace("get_", "") + "_bytes"] = int(fn())
    return stats


def load_text_model(
    pack: str | Path | None = None,
    *,
    use_custom_kernels: bool = True,
    load_tokenizer: bool = True,
    mix_five_trit: bool = True,
    trit_wins: frozenset | None = None,
) -> LoadedModel:
    pack = Path(pack or DEFAULT_PACK)
    config = json.loads((pack / "config.json").read_text())
    if config.get("model_type") != "prism_hadamard_qwen35":
        raise ValueError(f"unsupported model_type {config.get('model_type')}")
    text_cfg = config["text_config"]
    model = TextModel(TextModelArgs.from_dict(text_cfg))

    weights = mx.load(str(pack / "model.safetensors"))
    vision_keys = [k for k in weights if k.startswith("vision_tower")]
    skipped = 0
    for k in vision_keys:
        skipped += int(weights[k].nbytes)
        del weights[k]

    lang = {}
    for k, v in weights.items():
        if k.startswith(LANGUAGE_PREFIX):
            lang[k[len(LANGUAGE_PREFIX) :]] = v
        else:
            lang[k] = v
    del weights

    seen: set[str] = set()
    for record in config["modules"]:
        path = record["path"]
        if path in seen:
            raise ValueError(f"duplicate packed module {path}")
        seen.add(path)
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        arrays = [lang[path + "." + s] for s in ("weight", "scales", "biases")]
        signs = lang.get(path + ".signs")
        block = int(record.get("block") or 0)
        if record.get("embedding"):
            module = PackedEmbedding(*arrays, signs, block, dtype=mx.float16)
        else:
            module = PackedLinear(
                *arrays, signs, block, dtype=mx.float16, use_custom=use_custom_kernels
            )
        setattr(parent, parts[-1], module)

    # Remaining tensors (norms, GDN A_log / dt_bias / conv1d, in_proj_a/b, ...).
    model.load_weights(list(lang.items()), strict=False)
    model.eval()
    mx.eval(model.parameters())

    mix_report = None
    if use_custom_kernels and mix_five_trit:
        from monkeyinference.trit import apply_mixed_five_trit

        mix_report = apply_mixed_five_trit(model, wins=trit_wins)

    tokenizer = None
    if load_tokenizer:
        tokenizer = _load_tokenizer(pack)

    language_bytes = sum(int(v.nbytes) for v in lang.values())
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        pack=pack,
        use_custom_kernels=use_custom_kernels,
        language_bytes=language_bytes,
        skipped_vision_bytes=skipped,
        mix_five_trit=mix_report,
    )


def _load_tokenizer(pack: Path) -> TokenizerWrapper:
    try:
        return load_mlx_tokenizer(pack, eos_token_ids=None)
    except Exception:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(pack), trust_remote_code=True)
        return TokenizerWrapper(tok)


def apply_chat(
    tokenizer,
    user: str,
    *,
    system: str = "You are a helpful assistant.",
    enable_thinking: bool = False,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = tokenizer
    inner = getattr(tokenizer, "_tokenizer", None) or getattr(tokenizer, "tokenizer", tokenizer)
    try:
        return inner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return inner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
