# Monkeyinference

Standalone Metal/MLX engine for [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) on Apple silicon. Not a Splash fork.

Target machine for numbers in this repo: MacBook Air M4, 24 GB, 10 GPU cores, **120 GB/s** published DRAM / **86.6 GB/s** measured STREAM.

## Why this exists

Bonsai's geometry matches Qwen3.8-27B, but Splash only loads `splash-packed-q4` packages with a trained DFlash 2 draft. Prism's bundled MLX runtime already ran the 2-bit Hadamard pack at about **8 tok/s decode / 40 tok/s prefill** on this Air. That is ~half of a STREAM roofline. Beating the roofline requires speculative decode; closing the rest of the STREAM gap is a qdot kernel, not a vibe.

## What it does

- Text-only load of the MLX pack (skips the 0.92 GB FP16 vision tower)
- Custom Metal ternary GEMV (`qmv_fast` qdot, `(code-1)*scale`, no bias traffic) plus the Prism activation Hadamard
- Numerical parity against `mx.quantized_matmul` (the Prism path)
- Greedy decode, leftover-greedy, prompt-lookup speculative decode, and early-exit self-speculation with copy-on-write GDN pins
- Coherence + token-identity gates wired into `monkeyinference bench`

On this Air, greedy explain is **9.74 tok/s**. Same-prompt PLD copy is **11.70 tok/s vs 8.19 leftover greedy** (10/10 accept, token-identical). Early-exit self-speculation plateaus at **1.07 accepts/pass** — not a substitute for a trained draft.

## Run

Use the existing MLX venv so we do not duplicate ~GB of packages:

```bash
export PYTHONPATH=src
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli profile --out results/roofline.json
~/.monkey/mlx-venv/bin/python tests/test_kernels.py
~/.monkey/mlx-venv/bin/python tests/test_spec.py
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench --out results/bench.json
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate --speculative --draft pld
```

`--no-custom` forces MLX affine matmul. `--draft early --early-layers 4` is self-speculation (slow until accepts/pass rises).

Weights are expected at `~/.monkey/models/Ternary-Bonsai-2-27B-mlx-2bit/` (already on this machine). Nothing is downloaded.

## Design and measured numbers

Living document: [docs/ternary-engine.md](docs/ternary-engine.md) (also the project-store copy Ashvin tracks).

## Disk

This repo is source only. Do not download a draft model without checking free space. Prompt-lookup speculation needs no extra weights. Early-exit reuses Bonsai's own first N layers.
