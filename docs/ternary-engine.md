# Ternary engine for Bonsai 2 27B on this M4 Air

Living design-and-results doc. Engine repo: [github.com/ashvinar/monkeyinference](https://github.com/ashvinar/monkeyinference) · branch `cursor/ternary-metal-engine-09ea`. Standalone Metal/MLX runtime, not a Splash fork.

**Headline (2026-09-18, evening):** Decode is still DRAM-bandwidth bound. Greedy explain is **9.74 tok/s** (86% of the 11.3 measured-STREAM ceiling). Same-prompt PLD copy is **11.70 vs 8.19 leftover-greedy, 1.43×, 10/10, token-identical**. Self-speculation from the first N∈{2,4,6,8} layers plateaus at **1.07 accepts/pass**.

The cheaper draft lead is **in**. Official Splash `draft/` (`incoai/Qwen3.8-27B-Splash`) is 5 layers, **1.266 GB**, vocab **248320** and hidden **5120** matching Bonsai. Unpacked `MDFD0004` Q4 into MLX affine-4bit. On the explain prompt, DFlash 2 leftover-verify is **3.00 accepts/pass** (32 accepted / 103 proposed, 16 verify passes for 48 tokens), **token-identical** to leftover greedy. That is the chat number early-exit never reached. The dflash *loop* is not yet a tok/s win (0.80 vs leftover 2.77 this run) because context KV is rebuilt every propose and rejects still replay — implementation tax, not a modelling wall. Training a new draft is not required to clear 2× accepts/pass.

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
| Free disk | **~16–18 GB** after adding **1.266 GB** Splash `draft/` only. |

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
| Self-spec draft = first N layers + shared `lm_head` | Vocab 248320 matches by construction; 0 new bytes. N∈{2,4,6,8} all plateau at ~1.07 accepts/pass. | Do not tune N further. |
| Chat draft = Splash DFlash 2 `draft/` | Official 5-layer block-diffusion drafter trained on Qwen3.8-27B. Vocab/hidden match. Context is concatenated Bonsai hiddens at layers 5/19/33/47/61. Logits use Bonsai's `lm_head`. Residual stream kept float32 (Q4 o_proj biases overflow fp16). | `--draft dflash`. Do not download the 17.4 GB Splash package or the 3.85 GB BF16 DFlash2 repo. |
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

**Plateau below 2× accepted tokens per pass.** Intermediate activations plus the final `lm_head` are not a usable draft for this pack.

### Splash DFlash 2 (`incoai/Qwen3.8-27B-Splash` `draft/` only)

Hub listing first. Whole package 17.4 GB; **not downloaded**. `draft/` is 6 files, **1,266,040,832 B**, SHA-256 matching the manifest. Magic `MDFD0004`, Q4 group 64, StorageN=256. Tokenizer: 248044 vocab strings + 33 specials **id-identical** to Bonsai; `tokenizer.json` hashes differ (`06b95093…` vs `0997f410…`) because merges are list-vs-string and the pretokenizer regex adds combining marks.

Unpack: scatter StorageN tiles → MLX affine-4bit (`dequant = scale*q + bias`, 8 nibbles/uint32, low first). Section accounting matches the 187,449,344 B layer files and 328,794,112 B `model.bin` exactly. Synthetic pack/unpack vs numpy max abs 0.044.

Explain prompt, leftover protocol, 48 gen tokens, argmax, token-identical to leftover greedy:

| Engine | Accepts/pass | Accepted/proposed | Verify passes | Decode tok/s | Peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| leftover greedy | 1.00 | 0 / 0 | 48 | 2.77 | 8.28 GB |
| early-exit N=4 | 1.07 | 3 / 170 | — | 1.93 | — |
| **DFlash 2** | **3.00** | **32 / 103** | **16** | 0.80 | **9.54 GB** |

3.00 is committed tokens per target pass (accepted drafts + leftover/bonus). Published DFlash 2 on matching dense Qwen3.8-27B is ~4.1–5.5; Bonsai is ternary Hadamard, so 3.0 on explain is the transfer number. fp16 residuals proposed `[0,0,0,…]` (`!!!!!!!`); float32 residuals proposed the greedy continuation on the first step (`Speculative decoding is a technique that`).

The 0.80 tok/s is **not** the claim. Propose currently rebuilds draft KV from the full aux context every step and pays reject+replay. Accepts/pass is the number that decided the lead.

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
  - leftover greedy == DFlash 2 on explain (48 tok, `identity_ok`)

A fast engine emitting plausible-looking noise, or a spec loop that silently diverges on reject, would have failed this loop. It did not.

Tokenizer still warns about the Mistral-Small regex. Outputs were English and correct on these prompts; a tokenizer-sensitive eval is still owed.

## What's in the repo

```
src/monkeyinference/
  kernels.py     Metal STREAM copy + qdot GEMV + weight-once verify (mx.fast.metal_kernel)
  hadamard.py    Prism fwht
  packed.py      PackedLinear / PackedEmbedding (qdot default; MLX qmm for prefill)
  load.py        text-only safetensors → mlx_lm Qwen3.5 TextModel
  generate.py    greedy (mlx_lm.stream_generate) + leftover greedy + PLD/early/dflash spec
  spec.py        n-gram drafter, early-exit drafter, CoW GDN/KV pins
  splash_q4.py   MDFD0004 unpack → MLX affine-4bit
  dflash.py      5-layer DFlash 2 + selector + Bonsai aux hiddens
  roofline.py    STREAM + GEMV + the 7.674 GB math
  bench.py       coherence + same-prompt PLD vs greedy + identity + early-exit + `--dflash`
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
~/.monkey/mlx-venv/bin/python tests/test_splash_q4.py
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench --dflash --out results/dflash.json
```

Uses the existing `~/.monkey/mlx-venv` (mlx 0.32.0). No second venv. `--no-custom` forces the Prism MLX matmul.

## What's left

1. **Make DFlash tok/s follow accepts/pass.** 3.00 accepts/pass on explain is enough. The loop still rebuilds draft KV every propose and replays on partial reject, so 0.80 tok/s is not a speed claim. Persist draft cache, STREAM the Q4 projections, then re-bench.
2. **Prefill.** 40.9 tok/s vs llama-bench pp512 47. Still the compute-bound side; not the chat bottleneck.
3. Tokenizer regex; sampling; optional vision.
4. If traces after the qdot pass still show Python dispatch in the hot path, move the layer loop to compiled Metal. GEMV is now ~82–86% of measured STREAM, so that is second-order.

## Disk / cleanliness

Added: git repo under `~/projects/monkeyinference` (source). **+1.266 GB** at `~/.monkey/models/Qwen3.8-27B-Splash-draft/draft/` (`layer-0.bin`…`layer-4.bin` + `model.bin`). Did not download `target/`, `vision/`, or `incoai/Qwen3.8-27B-DFlash2`. Tokenizer compare used a 20 MB Hub file in `/tmp` and deleted it. Did not delete user data. Bench processes exited. No servers. Free disk ~16–18 GB.

## Rulings log

- 2026-09-18 **HOLD** bandwidth-bound decode. Greedy 9.74 tok/s ≈ 86% of measured STREAM / 7.674 GB (82% of the no-bias ceiling).
- 2026-09-18 **HOLD** speculation required to beat ~11 tok/s in general chat. PLD 11.70 is a copy-prompt number (2.8 accepts/pass), not a chat number.
- 2026-09-18 **UPDATE** custom qdot GEMV **is** production decode. `--no-custom` remains the Prism MLX path. Layer vs MLX max_abs 0.013; greedy tokens match.
- 2026-09-18 **UPDATE** Ashvin authorized Splash `draft/` bytes. Fetched **1.266 GB** only. Vocab/hidden match. Do **not** fetch the 17.4 GB package or the 3.85 GB BF16 DFlash2 repo.
- 2026-09-18 **HOLD** self-speculation with first N layers + shared `lm_head` plateaus at **1.07 accepts/pass** for N∈{2,4,6,8}.
- 2026-09-18 **NEW** Splash DFlash 2 on Bonsai explain is **3.00 accepts/pass**, token-identical to leftover greedy. Modelling lead **ruled in**. Training is not required to clear 2×. Next cost is making the dflash loop STREAM-bound, not a new draft.
- 2026-09-18 **NEW** GDN snapshot tax is gone (CoW pin). Weight-once verify is why PLD copy is 11.70 vs 8.19 leftover greedy, not 7.7 vs 8.4.
