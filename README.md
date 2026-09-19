# Monkeyinference

Standalone Metal/MLX engine for [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) on Apple silicon. Not a Splash fork.

**Default path:** greedy 2-bit qdot decode. Speculation is implemented and **off by default**. On this hardware it is net-negative.

The measurement record — what was tried, what killed each idea, and whether anything beats ~11 tok/s — is [docs/ternary-engine.md](docs/ternary-engine.md). Read that to come back to this project.

## Hardware these numbers assume

MacBook Air M4 (Mac16,12), 24 GB, **10 GPU cores**, published DRAM **120 GB/s**. Measured GPU STREAM copy **87–94 GB/s** (historical **86.6**); read-only **85–94 GB/s** (same band, not 100–110). Low Power Mode **must be off** or greedy drops to ~2.8 tok/s.

A different Mac (M4 Pro / M5) will be faster because the bus is faster, not because this engine is incomplete.

## Honest performance

| | tok/s |
| --- | ---: |
| Starting point (Prism MLX) | **8.0** |
| This engine, greedy explain | **10.21–10.22** |
| Measured STREAM ceiling | **11.3–12.7** |
| PLD on a copyable prompt | 11.70 (not general chat) |
| DFlash 2 default (K=2) | 7.32 (loses to greedy) |
| DFlash 2 K=7 predicted, no replay | 8.71–9.39 (still loses) |
| DFlash 2 K=7 + LoRA r=16 | 3.68 tok/s, **2.09** accepts (worse than stock 3.00) |

Greedy is 80–90% of the measured STREAM ceiling for a 7.254 GB/token weight stream. There is no remaining software path on this Air that materially beats ~11 tok/s. The binding constraint is memory bandwidth. Details and the ruled-out list are in the doc.

France → `Paris`. Speculative paths that were measured are token-identical to leftover greedy.

## Install

Use the existing MLX venv. Do not create a second one.

```bash
cd ~/projects/monkeyinference
export PYTHONPATH=src
# already present on this machine:
#   ~/.monkey/mlx-venv          mlx 0.32.0, mlx-lm 0.31.3
#   ~/.monkey/models/Ternary-Bonsai-2-27B-mlx-2bit/
~/.monkey/mlx-venv/bin/python tests/test_kernels.py
```

From a clean checkout, the same venv works if you `pip install -e .` into it. `pyproject.toml` lists `mlx>=0.32.0` and `mlx-lm>=0.31.3`.

## Weights

| Path | What | Needed to run? |
| --- | --- | --- |
| `~/.monkey/models/Ternary-Bonsai-2-27B-mlx-2bit/` | Prism affine-2bit MLX pack (~8.0 GB on disk; 7.67 GB language + 0.92 GB unused vision) | **Yes** for this engine |
| `~/.monkey/models/Qwen3.8-27B-Splash-draft/draft/` | Splash DFlash 2 Q4 draft, **1.266 GB** | Only for `--draft dflash` |
| `~/.monkey/models/Ternary-Bonsai-2-27B-PQ2_0.gguf` | llama.cpp PQ2_0, 6.7 GB | **No** — not used here |
| `~/.monkey/dflash-ft/` | Overfit LoRA r=16 cache (~322 MB) | **No** — 2.09 accepts; not loaded |
| `~/.monkey/dflash-ft2/` | Bounded LoRA retry (r=8, 1200 prompts) | **No** until it beats 3.70 |

Do not fetch the 17.4 GB Splash package or `incoai/Qwen3.8-27B-DFlash2`.

## Run

Greedy generate (production path):

```bash
export PYTHONPATH=src
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate \
  --prompt "Explain speculative decoding in two short sentences." \
  --max-tokens 48
```

Coherence + token-identity bench (includes a PLD copy-prompt check; default generate is still greedy):

```bash
~/.monkey/mlx-venv/bin/python tests/test_kernels.py
~/.monkey/mlx-venv/bin/python tests/test_spec.py
~/.monkey/mlx-venv/bin/python tests/test_splash_q4.py
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench --out results/bench.json
```

STREAM + GEMV roofline (no 27B load):

```bash
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli profile --out results/roofline.json
```

Speculation is **available and off**. On this Air every measured draft loses to greedy except PLD on a copyable sentence. Re-run the negative results with:

```bash
# prompt-lookup (copy prompts only; 11.70 vs 8.19 leftover greedy)
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate --speculative --draft pld

# early-exit (plateau 1.07 accepts/pass)
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate --speculative --draft early --early-layers 4

# Splash DFlash 2 (needs the 1.27 GB draft/; default K=2 is 7.32 tok/s)
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate --speculative --draft dflash
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench --dflash --out results/dflash.json
```

`--no-custom` forces Prism MLX matmul. `--mix-five-trit` applies the per-shape five-trit gate (win table is empty; stays 2-bit).

Evidence probes: `scripts/probe_mixed.py`, `scripts/probe_breakeven.py`, `scripts/probe_five_trit.py`. See `scripts/README.md`.

## What is gated, not deleted

`apply_mixed_five_trit`, `ternary_trit_qmm`, `ternary_qmm_m8`, PLD, early-exit, DFlash, and the LoRA wrap stay in the tree. They are the evidence. Production `load_text_model` does not enable five-trit. Production `generate` does not speculate. Production `--draft dflash` loads stock Q4, not `~/.monkey/dflash-ft/adapters.safetensors`.

## License

Apache-2.0.
