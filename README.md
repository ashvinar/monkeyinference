# Monkeyinference

Standalone Metal/MLX engine for [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) on Apple silicon. Not a Splash fork.

Target machine for numbers in this repo: MacBook Air M4, 24 GB, 10 GPU cores, **120 GB/s** published DRAM. GPU STREAM copy **87–94 GB/s**; GPU read-only **85–94 GB/s** (decode-like, same band — not 100–110). CPU+GPU disjoint read **103 GB/s** aggregate (not a hybrid plan).

## Why this exists

Bonsai's geometry matches Qwen3.8-27B, but Splash only loads `splash-packed-q4` packages with a trained DFlash 2 draft. Prism's bundled MLX runtime already ran the 2-bit Hadamard pack at about **8 tok/s decode / 40 tok/s prefill** on this Air. That is ~half of a STREAM roofline. Beating the roofline requires speculative decode; closing the rest of the STREAM gap is a qdot kernel, not a vibe.

## What it does

- Text-only load of the MLX pack (skips the 0.92 GB FP16 vision tower)
- Custom Metal ternary GEMV (`qmv_fast` qdot, `(code-1)*scale`, no bias traffic) plus the Prism activation Hadamard
- Numerical parity against `mx.quantized_matmul` (the Prism path)
- Greedy decode, leftover-greedy, prompt-lookup speculative decode, early-exit self-speculation, and Splash DFlash 2 leftover-verify
- Coherence + token-identity gates wired into `monkeyinference bench`

On this Air, greedy explain is **10.22 tok/s** (Low Power Mode off; 9.74 is the same band), 80–90% of a read-like STREAM ceiling of **11.3–12.7 tok/s**. Same-prompt PLD copy is **11.70 tok/s vs 8.19 leftover greedy** (10/10, token-identical). Early-exit plateaus at **1.07 accepts/pass**. Splash DFlash 2 transfers at **3.00 accepts/pass** (K=7) and **7.32 tok/s** at default K=2, token-identical, and does not beat greedy: verify was a matvec handed 8 rows. `ternary_qmm_m8` is the 8-row MMA (flat on that kernel, **2.25× vs greedy qdot**, 1.3× gate missed). Codes are genuinely ternary (0/3.54e9 code-3). Training a new draft is not required to clear 2× accepts/pass.

## Run

Use the existing MLX venv so we do not duplicate ~GB of packages:

```bash
export PYTHONPATH=src
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli profile --out results/roofline.json
~/.monkey/mlx-venv/bin/python tests/test_kernels.py
~/.monkey/mlx-venv/bin/python tests/test_spec.py
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench --out results/bench.json
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate --speculative --draft pld
~/.monkey/mlx-venv/bin/python tests/test_splash_q4.py
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench --dflash --out results/dflash.json
```

`--no-custom` forces MLX affine matmul. `--draft early --early-layers 4` is self-speculation (plateaued; do not tune N further). `--draft dflash` uses the Splash DFlash 2 Q4 pack at `~/.monkey/models/Qwen3.8-27B-Splash-draft/draft/` (default `--num-draft 2`; `--num-draft 7` is the 3.00 accepts/pass setting).

Bonsai weights are expected at `~/.monkey/models/Ternary-Bonsai-2-27B-mlx-2bit/` (already on this machine). The DFlash draft was fetched as `draft/` only (1.266 GB); the 17.4 GB Splash package and the 3.85 GB BF16 DFlash2 repo were not downloaded.

## Design and measured numbers

Living document: [docs/ternary-engine.md](docs/ternary-engine.md) (also the project-store copy Ashvin tracks).

## Disk

This repo is source only. Prompt-lookup speculation needs no extra weights. Early-exit reuses Bonsai's own first N layers. The authorized DFlash draft is **+1.266 GB** at `~/.monkey/models/Qwen3.8-27B-Splash-draft/` (not in git). Do not fetch `target/`, `vision/`, or `incoai/Qwen3.8-27B-DFlash2`. Free disk after that add is ~16–18 GB.
