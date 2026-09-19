# Ternary engine for Bonsai 2 27B on this M4 Air

Living design-and-results doc. Engine repo: [github.com/ashvinar/monkeyinference](https://github.com/ashvinar/monkeyinference) · branch `cursor/ternary-metal-engine-09ea`. Standalone Metal/MLX runtime, not a Splash fork.

**Headline (2026-09-18, later the same day):** Decode is still DRAM-bandwidth bound, but the engine is no longer stuck at the Prism 8 tok/s starting point. A qdot clone of MLX `qmv_fast` (64-thread TGs, biases dropped, Hadamard graph-fused on the 10 KB activation) takes greedy explain to **9.74 tok/s** (86% of the 11.3 measured-STREAM ceiling). Prompt-lookup speculation on the *same* copy prompt is **11.70 tok/s vs 8.19 leftover-greedy**, **1.43×**, **10/10 drafts accepted**, token-identical to greedy. That used to be 7.7 vs 8.4 on mismatched prompts because each verify memcpy'd ~150 MB of GDN state and re-streamed weights once per draft token. Copy-on-write pins plus a weight-once qdot over leftover+K drafts killed that tax.

Self-speculation did **not** get there. Drafting from the first N∈{2,4,6,8} of 64 layers plus the existing `lm_head` (vocab 248320, zero extra bytes) plateaus at **1.07 accepted tokens per verify pass**. That is a modelling limit, not an implementation tax. Taking a trained / downloaded draft to Ashvin is the next disk question; this agent will not spend the 17 GB free.

## Machine

| | |
| --- | ---: |
| Computer | MacBook Air (Mac16,12) |
| Chip | Apple M4, 10 CPU (4P+6E), 10 GPU, Metal 4 |
| Unified memory | 24 GB |
| Published DRAM | **120 GB/s** (Apple M4 Air spec, LPDDR5X) |
| Measured STREAM copy | **86.6 GB/s** (Metal `out[i]=inp[i]`, 512 MiB, read+write) |
| STREAM / published | 72% |
| OS | macOS 27.0 |
| Free disk | **~17 GB**. **No extra weights downloaded.** |

## Roofline

Language tensors in the MLX pack: **7.674 GB** (6.822 GB 2-bit codes + 0.420 GB scales + 0.420 GB redundant biases + 0.012 GB Hadamard signs). Vision tower 0.921 GB is skipped. The custom kernel does not load biases, so the decode stream is **7.254 GB** of codes+scales+signs.

Each greedy decode token streams the language weights once (GEMV), plus a negligible activation/KV term at short context.

| Assumption | tok/s = GB/s ÷ bytes |
| --- | ---: |
| Published 120 GB/s, with biases (7.674 GB) | **15.6** |
| Published 120 GB/s, drop biases (7.254 GB) | 16.5 |
| Measured STREAM 86.6 GB/s, with biases | **11.3** |
| Measured STREAM 86.6 GB/s, drop biases | **11.9** |
| Starting Prism MLX decode | **8.0** |
| This engine greedy explain (qdot, no bias load) | **9.74** |
| 9.74 / 11.3 measured (with-bias ceiling) | **86%** |
| 9.74 / 11.9 measured (no-bias ceiling) | **82%** |
| PLD copy-prompt (same prompt as leftover greedy) | **11.70** |

Profiling **does not contradict** the two original rulings:

1. **Decode is memory-bandwidth bound.** 9.74 tok/s is most of what this Air can stream. The qdot pass recovered the ~1.4 tok/s that was sitting in `affine_qmv_fast` occupancy / bias traffic, not a 2–3×. Microbenches on this 10-core GPU still swing ±30% with thermal; e2e numbers above are the ones that matter.
2. **Beating the ceiling requires speculative decode.** PLD on a copyable prompt crosses 11.7 because it commits 2.8 tokens per target pass. That is *not* a general chat number. Self-speculation off Bonsai's own early layers does not provide a cheap general draft (see below).

Prism's own laptop table (PQ2_0 GGUF, no speculation): M4 Pro 18 tok/s, M5 Pro 28 tok/s streaming ~204 GB/s. This Air landing at ~10 tok/s greedy on 87 GB/s STREAM is still in family.

## Architecture decisions

| Decision | Why | Override |
| --- | --- | --- |
| Standalone Python+MLX repo, not a Splash fork | Splash factory has no AR fallback and only loads `splash-packed-q4` + DFlash. Bonsai is affine-2bit Hadamard safetensors. | Rewrite in Swift/Metal if Python dispatch shows up in traces now that GEMV is STREAM-bound. |
| Text-only load | Vision tower is 0.92 GB FP16 and unused for these prompts. Peak Metal ~8.3 GB. | Keep the tower behind an explicit `--vision` flag later. |
| Keep Prism activation Hadamard (block 1024, `H(x⊙s)/√1024`) | Weights are stored in that basis. Skipping it is silent garbage. | None. |
| Default matmul = custom qdot GEMV (`--no-custom` for MLX) | Clone of MLX `qmv_fast`: 64-thread TGs, 2 simdgroups × 4 rows, pre-shifted x, mask-and-accumulate, no bias load (`y = s·(codes·x − Σx)` because Prism stores `bias = −scale`). Hadamard stays `mx.hadamard_transform` immediately before the GEMV (10 KB; inlining a 1024-point FWHT into an output-tiled TG would recompute H(x) per 8 output rows). E2e greedy 8.38 → **9.74**. Layer-level vs MLX `max_abs` 0.013. | `--no-custom`. |
| Spec verify (M=2..16) = weight-once qdot | MLX stays on `qmv` until M≈12 on these shapes, so leftover+K drafts used to re-stream 7.6 GB per draft token. The once-kernel keeps M in the inner loop; grid is over output-row groups only. Prefill M>16 stays on `mx.quantized_matmul`. | None for spec. |
| GDN cache pin is copy-on-write, not memcpy | `gated_delta` already writes a new `state_out`; `GatedDeltaNet` does `cache[i] = new`. Holding the previous array refs / KV offsets is a real CoW pin (~151 MB × 48 layers is **not** copied). Partial reject reverts pointers and replays leftover+accepted (GDN is recurrent; no per-timestep states). | If Apple adds trimmable GDN cache, switch. |
| First draft = prompt-lookup n-gram (PLD), K=5 | Zero extra weights. K=8+ over-proposes on the copy sentence and pays reject+replay. | Tune K per prompt class. |
| Self-spec draft = first N layers + shared `lm_head` | Vocab 248320 matches by construction; 0 new bytes. N∈{2,4,6,8} all plateau at ~1.07 accepts/pass. | Do **not** download or train a draft without asking Ashvin. |
| Correctness gates in the bench loop | Packing vs `dequantize`; greedy `Paris` + speculative-decoding explanation; **argmax spec tokens == leftover-greedy tokens** on copy and explain. | Token-level logit KL vs Prism if a claim depends on 0.1 tok/s. |

## Benchmark progression

Greedy, thinking off, temp 0. Same prompts as the Splash-investigation table unless noted.

| Engine | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | Peak | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prism MLX (prior) explain | 32 | 59 | 30.0 | **8.16** | 9.26 GB | coherent spec-decode explanation |
| Prism MLX (prior) long | 717 | 60 | 40.2 | 7.81 | 12.09 GB | coherent |
| llama.cpp Metal PQ2_0 tg128 | — | 128 | — | 6.71 ± 1.40 | — | forced tokens |
| monkeyinference greedy explain (milestone 1, MLX affine) | 32 | 59 | 33.1 | **8.38** | 8.25 GB | same idea, coherent |
| monkeyinference greedy long (milestone 1) | 717 | 60 | **41.6** | 8.22 | 10.42 GB | same two-sentence explanation |
| monkeyinference PLD copy (milestone 1, GDN memcpy) | 46 | 14 | 32.0 | 7.72 | 8.33 GB | exact copy; 10/10 accept; **slower than greedy** |
| **greedy explain (qdot, this pass)** | 32 | 59 | 34.1 | **9.74** | **8.24 GB** | same Prism explanation, token-stable |
| **greedy long (qdot)** | 717 | 60 | **40.9** | 9.33 | 10.41 GB | same two-sentence idea |
| **leftover greedy copy (fair baseline)** | 46 | 14 | 30.1 | **8.19** | 8.38 GB | exact copy |
| **PLD copy (CoW + weight-once qdot)** | 46 | 14 | 29.9 | **11.70** | 8.38 GB | exact copy; **10/10**; **1.43× vs leftover greedy**; token-identical |

Starting point for this project: **8 tok/s decode, ~40 tok/s prefill**. Current greedy: **9.74 / 40.9**. Current same-prompt PLD: **11.70 vs 8.19**. 2-token `Paris` rows still read ~19 tok/s (kernel launch + EOS). Ignore them.

The 7.7 vs 8.4 row from this morning compared a 14-token copy to a 59-token explain. That was the wrong picture. The implementation tax was real (memcpy + per-token weight stream); it is no longer the thing holding PLD below greedy.

### Self-speculation (early-exit)

Same leftover protocol, full-model verify, first N layers + shared `lm_head`, no new tensors. Explain prompt, 48 gen tokens, argmax. Token-identical to leftover greedy in every N.

| N | Accepts/pass | Accepted/proposed | Decode tok/s | Notes |
| ---: | ---: | ---: | ---: | --- |
| 2 | 1.04 | 2 / 174 | 1.40 | first two GDN layers only |
| **4** | **1.07** | 3 / 170 | 1.93 | includes first full-attn (layer 3) |
| 6 | 1.07 | 3 / 170 | 1.28 | |
| 8 | 1.07 | 3 / 170 | 1.14 | |

**Plateau below 2× accepted tokens per pass.** Intermediate activations plus the final `lm_head` are not a usable draft for this pack. A trained early-exit head or a separate 248320-vocab draft would need disk/training that this agent is not allowed to spend.

### Microbench (no 27B load)

STREAM 86.6 GB/s. GEMV numbers on this Air swing with thermal; treat as ±30% and prefer the e2e table. After the qdot pass, MLP-up is tied-to-ahead of MLX, attn-q ahead, lm_head slice well ahead (wide N loves 64-thread TGs). E2e greedy +1.4 tok/s is the claim, not a single-shape GB/s.

MLX itself will not switch `qmv`→`qmm` on Bonsai MLP-up until M≈12. Spec leftover+5 is M=6, so flattening `[1,K,H]→[K,H]` is necessary but not sufficient; the weight-once kernel is what actually streams W once.

## Correctness

- **Packing:** synthetic ternary codes packed as MLX uint32×16, `mx.dequantize` vs numpy unpack **max abs 0**.
- **Hadamard:** `fwht` matches `mx.hadamard_transform` on signed 5120-wide activations (err < 1e-3). Inverse∘forward is an involution within fp16.
- **GEMV vs numpy:** max abs ~0.10 on 17408×5120 (fp16 inputs, f32 accum). vs MLX affine qdot, e2e layer probe **max_abs 0.013** (parity_ok).
- **Weight-once vs stacked GEMV:** max abs 0.016 on 256×512 × M=6.
- **Coherence (wired in `monkeyinference bench`):**
  - “Name the capital of France…” → `Paris`
  - “Explain speculative decoding in two short sentences.” → draft/verify explanation, English, not garbage
  - Long-prefill variant produced the same two-sentence idea as the prior Prism MLX JSON
  - PLD copy prompt reproduced the sentence
- **Token identity (argmax is lossless if rollback is correct):**
  - leftover greedy == `mlx_lm.stream_generate` on Paris
  - leftover greedy == PLD on the copy prompt
  - leftover greedy == early-exit N=4 on explain (48 tok)

A fast engine emitting plausible-looking noise, or a spec loop that silently diverges on reject, would have failed this loop. It did not.

Tokenizer still warns about the Mistral-Small regex. Outputs were English and correct on these prompts; a tokenizer-sensitive eval is still owed.

## What's in the repo

```
src/monkeyinference/
  kernels.py     Metal STREAM copy + qdot GEMV + weight-once verify (mx.fast.metal_kernel)
  hadamard.py    Prism fwht
  packed.py      PackedLinear / PackedEmbedding (qdot default; MLX qmm for prefill)
  load.py        text-only safetensors → mlx_lm Qwen3.5 TextModel
  generate.py    greedy (mlx_lm.stream_generate) + leftover greedy + PLD/early spec
  spec.py        n-gram drafter, early-exit drafter, CoW GDN/KV pins
  roofline.py    STREAM + GEMV + the 7.674 GB math
  bench.py       coherence + same-prompt PLD vs greedy + identity + early-exit
```

Weights stay at `~/.monkey/models/Ternary-Bonsai-2-27B-mlx-2bit/`. The repo is source only.

```bash
export PYTHONPATH=src
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli profile
~/.monkey/mlx-venv/bin/python tests/test_kernels.py
~/.monkey/mlx-venv/bin/python tests/test_spec.py
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate --speculative --draft pld
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate --speculative --draft early --early-layers 4
```

Uses the existing `~/.monkey/mlx-venv` (mlx 0.32.0). No second venv. `--no-custom` forces the Prism MLX matmul.

## What's left

1. **A draft that actually accepts ~2× per pass on real prompts.** PLD only wins when the prompt contains the continuation (copy/code). Early-exit N-layer + shared `lm_head` plateaus at 1.07. That is the modelling wall. A trained 5-layer DFlash / early-exit head / tiny 248320-vocab draft needs **disk + (maybe) training** — ask Ashvin, do not download.
2. **Prefill.** 40.9 tok/s vs llama-bench pp512 47. Still the compute-bound side; not the chat bottleneck.
3. Tokenizer regex; sampling; optional vision.
4. If traces after this qdot pass still show Python dispatch in the hot path, move the layer loop to compiled Metal. GEMV is now ~82–86% of measured STREAM, so that is second-order.

## Disk / cleanliness

Added: git repo under `~/projects/monkeyinference` (source, kilobytes). No new model files. Did not delete user data. Bench processes exited. No servers. Free disk still ~17 GB.

## Rulings log

- 2026-09-18 **HOLD** bandwidth-bound decode. Greedy 9.74 tok/s ≈ 86% of measured STREAM / 7.674 GB (82% of the no-bias ceiling).
- 2026-09-18 **HOLD** speculation required to beat ~11 tok/s in general chat. PLD 11.70 is a copy-prompt number (2.8 accepts/pass), not a chat number.
- 2026-09-18 **UPDATE** custom qdot GEMV **is** production decode. `--no-custom` remains the Prism MLX path. Layer vs MLX max_abs 0.013; greedy tokens match.
- 2026-09-18 **HOLD** do not download a draft until Ashvin okays the bytes.
- 2026-09-18 **NEW** self-speculation with first N layers + shared `lm_head` plateaus at **1.07 accepts/pass** for N∈{2,4,6,8}. Report to Ashvin rather than train/download.
- 2026-09-18 **NEW** GDN snapshot tax is gone (CoW pin). Weight-once verify is why PLD copy is 11.70 vs 8.19 leftover greedy, not 7.7 vs 8.4.
