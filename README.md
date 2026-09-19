# Monkeyinference

Standalone Metal/MLX engine for [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) on Apple silicon. Not a Splash fork.

Target machine for numbers in this repo: MacBook Air M4, 24 GB, 10 GPU cores, **120 GB/s** published DRAM.

## Why this exists

Bonsai's geometry matches Qwen3.8-27B, but Splash only loads `splash-packed-q4` packages with a trained DFlash 2 draft. Prism's bundled MLX runtime already runs the 2-bit Hadamard pack at about **8 tok/s decode / 40 tok/s prefill** on this Air. That is ~half of a STREAM roofline. Beating the roofline requires speculative decode, not a faster vibe.

## What it does

- Text-only load of the MLX pack (skips the 0.92 GB FP16 vision tower)
- Custom Metal ternary GEMV (`(code-1)*scale`, no bias traffic) plus the Prism activation Hadamard
- Numerical parity against `mx.quantized_matmul` (the Prism path)
- Greedy decode and prompt-lookup speculative decode with GDN cache snapshot/restore
- Coherence prompts wired into `monkeyinference bench`

## Run

Use the existing MLX venv so we do not duplicate ~GB of packages:

```bash
export PYTHONPATH=src
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli profile --out results/roofline.json
~/.monkey/mlx-venv/bin/python -m pytest tests/test_kernels.py
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench --out results/bench.json
```

Weights are expected at `~/.monkey/models/Ternary-Bonsai-2-27B-mlx-2bit/` (already on this machine). Nothing is downloaded.

## Design and measured numbers

Living document: [docs/ternary-engine.md](docs/ternary-engine.md) (also the project-store copy Ashvin tracks).

## Disk

This repo is source only. Do not download a draft model without checking free space. Prompt-lookup speculation needs no extra weights.
