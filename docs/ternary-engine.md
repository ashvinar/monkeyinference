# Ternary engine for Bonsai 2 27B on an M4 Air

Standalone Metal/MLX runtime for [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit). Not a Splash fork. Repo: [github.com/ashvinar/monkeyinference](https://github.com/ashvinar/monkeyinference).

This document is the complete technical record. It is written for a reader who has not followed the work. Every speed claim below was gated on **token identity**: argmax tokens from a speculative (or mixed-pack) path must equal leftover-greedy tokens on the same prompt.

**Production decode is 2-bit qdot greedy.** Speculation, five-trit packing, and hybrid CPU/GPU are in the tree as gated experiments. They are net-negative or no-ops on this machine. Do not build always-8 plus GDN commit. Do not convert the pack. Do not train a drafter.

## Straight answer: is there a path past ~11 tok/s?

**No software path remaining on this MacBook Air materially beats ~11 tok/s for this model.** The binding constraint is the M4 Air’s realized GPU STREAM of **~86–94 GB/s** (historical copy **86.6**), not a missing kernel and not the published 120 GB/s DRAM figure. Each greedy token streams **7.254 GB** of language codes+scales+signs. Arithmetic ceiling:

| Bus | tok/s = GB/s ÷ 7.254 GB |
| --- | ---: |
| Measured copy 86.6 GB/s | **11.9** |
| Measured read 85 GB/s (1 GiB, decode-like) | **11.7** |
| Measured read 92 GB/s (512 MiB) | **12.7** |
| Published 120 GB/s | 16.5 *(not realized on this GPU)* |

This engine’s greedy explain is **10.21–10.22 tok/s** (Low Power Mode off), starting from Prism MLX **8.0**. That is **80–90%** of the measured STREAM ceiling. The qdot pass recovered the ~2 tok/s sitting in MLX `affine_qmv_fast` occupancy and redundant bias traffic. The leftover 1–2 tok/s of STREAM slack is thermal/occupancy on a 10-core GPU, not a second kernel project.

Speculative decode is how you beat a STREAM ceiling in general. On this GPU it does not:

- An 8-row verify costs **2.68–2.93×** a greedy step (software 2-bit dequant into threadgroup memory, then MMA). The 1.3× gate was missed.
- Splash DFlash 2 transfers **3.00 accepts/pass**. Beating 10.22 needs **3.26–3.51**. Predicted best **8.71–9.39 tok/s** even with replay deleted.
- Closing 3.00 → 4.1+ requires **training** a drafter on this 2-bit target. Even the published dense-target 4.1 accepts at current MMA cost is ~12 tok/s — still the STREAM ceiling, not past it. The high end (5.5) would be ~16 tok/s and is not the number we measured.

**The honest sentence:** more memory bandwidth. Prism’s laptop table (llama-bench, no spec) is M4 Pro **18** tok/s, M5 Pro **28**, M5 Max **47**. 10.22 on 86–92 GB/s is in family. A different Mac with a faster bus is the upgrade. This engine is at this hardware’s greedy limit.

Splash’s ~74 tok/s is a different pack (`splash-packed-q4`) with a matched draft and a native 4-bit MMA path. It is existence proof that *some* 27B decode on Apple silicon can be fast. It is not a path for this affine-2bit Hadamard pack on a 10-core Air.

## What the engine is

Bonsai 2 27B is a Qwen3.5-class text model (hidden 5120, 64 layers, 16 KV / 48 GDN, vocab 248320) stored as Prism affine-2-bit groups of 128 with a 1024-point activation Hadamard. Splash cannot load that pack. Prism’s bundled MLX path ran it at ~8 tok/s decode / ~40 tok/s prefill on this Air.

This repo:

- Loads **text only** (skips the 0.92 GB FP16 vision tower).
- Replaces packed Linear/Embedding modules so the Hadamard contract is applied. Stock `mlx_lm` without that transform emits garbage.
- Decodes with a custom Metal **qdot GEMV** (clone of MLX `qmv_fast`: 64-thread threadgroups, mask-and-accumulate, no bias load, because Prism stores `bias = −scale`).
- Verifies speculative leftover+K with an 8-row MMA when asked. Default generate does **not** speculate.
- Gates every end-to-end claim on coherence (`Paris`, a two-sentence spec-decode explanation) and token identity.

Python 3.11, `mlx` 0.32.0, `mlx-lm` 0.31.3, existing venv at `~/.monkey/mlx-venv`. No second venv.

## Hardware these numbers assume

| | |
| --- | ---: |
| Computer | MacBook Air (Mac16,12) |
| Chip | Apple M4, 10 CPU (4P+6E), **10 GPU**, Metal 4 |
| Unified memory | 24 GB |
| Published DRAM | **120 GB/s** (LPDDR5X spec) |
| GPU STREAM copy | **87–94 GB/s** (median **89** at 512 MiB; historical **86.6**). Counts read+write. |
| GPU STREAM read | **85–94 GB/s** (median **92** at 512 MiB, **85** at 1 GiB). Reduce-to-scalar; decode-like. |
| GPU STREAM write | **75–84 GB/s** fill |
| CPU STREAM read | **72 GB/s** (4 P-cores, QoS USER_INTERACTIVE, 512 MiB) |
| CPU+GPU disjoint read | **103 GB/s** aggregate (GPU 54 + CPU 49 under contention). **Ruled out as a decode plan.** |
| OS | macOS 27.0 |
| Low Power Mode | **Must be off.** LPM on battery drops greedy explain to **2.78 tok/s** while STREAM stays ~79 GB/s — clocks, not the bus.

Thermal swing on microbenches is ±10 GB/s / ±30% kernel. E2e greedy/spec numbers are the ones that matter. 2-token `Paris` rows read ~19 tok/s (launch + EOS); ignore them.

## Measured performance

Greedy, thinking off, temperature 0. Language tensors **7.674 GB** on disk (6.822 GB 2-bit codes + 0.420 GB scales + 0.420 GB redundant biases + 0.012 GB Hadamard signs). Custom qdot does not load biases, so the decode stream is **7.254 GB**.

| Run | Decode tok/s | Notes |
| --- | ---: | --- |
| Prism MLX starting point | **8.0** (explain 8.16) | `mx.quantized_matmul` affine-2bit |
| This engine, MLX affine, no custom qdot | 8.38 | Hadamard wired, vision skipped |
| **This engine, qdot greedy explain** | **9.74 then 10.21–10.22** | LPM off; 9.74 and 10.22 are the same band |
| qdot greedy long (717-token prefill) | 9.33 | prefill **40.9** tok/s |
| STREAM ceiling | **11.3–12.7** | 80–90% reached |
| PLD copy-prompt | **11.70** vs 8.19 leftover greedy | 10/10 accepts; **copyable prompt only** |
| DFlash K=2 (default) | **7.32** | 2.29 accepts/pass; net-negative vs greedy |
| DFlash K=7 e2e (MMA + replay) | **5.07** | 3.00 accepts/pass; token-identical |
| DFlash K=7 predicted, replay=0 | **8.71–9.39** | still below 10.22 |
| llama.cpp Metal PQ2_0 tg128 | 6.71 ± 1.40 | different pack |

France → `Paris`. Explain prompt emits a coherent draft/verify explanation, token-stable across greedy / leftover / DFlash.

## Ratio convention

Several tables use a wall-time ratio `A / B`. **Read the numerator.**

| Written | Means | How to read |
| --- | --- | --- |
| **trit_over_qdot** | `wall(five-trit GEMV) / wall(2-bit qdot)` | **>1 = five-trit is slower.** 2.02× on mlp_up is a loss, not a win. 0.60× would have been a win. |
| **T=8 / T=1** | `wall(8-row forward) / wall(1-row forward)` | **>1 = verify is more expensive than greedy.** 2.68× means you need >2.68 accepted tokens per verify (plus draft cost) to break even. |
| **MMA T=8 / MMA T=1** | same kernel, 8 rows vs 1 row padded to 8 | **1.00×** means the MMA is flat in M. That is not the same as beating greedy qdot. |
| **census-weighted 0.88×** | byte-weighted mean of trit_over_qdot | A mean of losses. Not “some tensors win.” |

Win gate used in code: five-trit is enabled for an `(N,K)` only if `trit_over_qdot < 0.95`. Empty `TRIT_WIN_NK` means every measured shape missed that bar.

## Production path

| Path | When | Kernel |
| --- | --- | --- |
| Greedy decode (default) | `M=1` | 2-bit `ternary_qmm` qdot |
| Spec verify | `M=2..8` | 2-bit `ternary_qmm_m8` MMA (pads short M) |
| Wide leftover | `M=9..16` | 2-bit `ternary_qmv_once` |
| Prefill | `M>16` | MLX `quantized_matmul` |
| Five-trit | never, unless `mix_five_trit=True` and `(N,K)` in `TRIT_WIN_NK` | `ternary_trit_qmm` (LUT5). Table is empty. |
| Hadamard | every packed linear with `block=1024` | `mx.hadamard_transform` on the 10 KB activation, immediately before the GEMV |

`load_text_model(..., mix_five_trit=False)` is the default. `generate(..., speculative=False)` is the default. `--no-custom` forces Prism MLX matmul for an A/B.

GDN cache pins are copy-on-write: `gated_delta` already writes a new `state_out`, so holding previous array refs is a real pin (~151 MB × 48 layers is **not** memcpy’d). Partial reject reverts pointers and replays leftover+accepted. GDN is recurrent; there are no per-timestep states to trim.

## Ruled out

Each row is a thing that was measured and killed. None of these are “not yet tried.”

### Chunked / WY GDN

**Hypothesis:** mlx_lm has a fast prefill GDN and a slow decode GDN, so speculative leftover+K pays a sequential tax that a chunked kernel would remove.

**Measurement:** `Qwen3_5.GatedDeltaNet` always calls `gated_delta_update` → Metal `gated_delta_kernel`. The kernel is one threadgroup per `(batch, value-head)` with `for (int t = 0; t < T; ++t)`. Prefill and verify are the **same sequential kernel**. Isolated GDN ×48 layers: T=1 16.9–30.2 ms, T=8 31.9–65 ms (~12% of a T=8 model forward). Forcing leftover+K onto MLX `quantized_matmul` made T=8 **worse** (863 ms): MLX stays on `qmv` until M≈12.

**Ruling:** there is no second “prefill GDN path” to switch to. A trainable WY-chunked gated-delta kernel would be new work, would still leave 401 ternary GEMVs, and is out of scope.

### Self-speculation (early-exit)

**Hypothesis:** the first N layers plus the shared `lm_head` are a free drafter (vocab matches by construction, 0 new bytes).

**Measurement:** leftover protocol, explain prompt, 48 gen, token-identical at every N.

| N | Accepts/pass | Decode tok/s |
| ---: | ---: | ---: |
| 2 | 1.04 | 1.40 |
| **4** | **1.07** | 1.93 |
| 6 | 1.07 | 1.28 |
| 8 | 1.07 | 1.14 |

**Ruling:** plateau below 2× accepted tokens per pass. Intermediate activations are not a usable draft for this pack. Do not tune N further.

### Five-trit packing, global switch

**Hypothesis:** the weights are ternary, so 5 trits/byte (`3^5=243<256`) is a lossless −18.75% re-encode (stream 5.975 GB). Briefing STREAM 86.6 / 5.975 ≈ 14.5 tok/s, if unpack is free.

**Measurement:** histogram on 3.54e9 codes is `{0,1,2}` only — **0 code-3**, so the pack *is* lossless (resolved below). Synthetic-weight GEMV, census-weighted, was **0.88×** vs 2-bit qdot (predicted greedy **9.00 tok/s**). mlp_up **2.02× slower**. The 11.5–13 survey number assumed unpack cheaper than the bytes saved.

**Ruling:** a global five-trit switch loses. Keep 2-bit qdot.

### Five-trit packing, per-shape mix

**Hypothesis:** format is per-tensor. Store mlp_up/down five-trit and leave the shapes where unpack loses on 2-bit. Take the max of the two everywhere; cannot do worse than 2-bit by construction.

**Measurement:** interleaved qdot vs trit on **real Bonsai tensors**, one per unique `(N,K)`, win iff `trit_over_qdot < 0.95`.

| Shape | N×K | trit/qdot | gate |
| --- | ---: | ---: | --- |
| mlp gate/up | 17408×5120 | **1.64×** | 2-bit |
| mlp_down | 5120×17408 | 1.59× | 2-bit |
| gdn_qkv | 10240×5120 | 2.14× | 2-bit |
| gdn_z | 6144×5120 | 1.81× | 2-bit |
| gdn_out / attn_o | 5120×6144 | 1.74× | 2-bit |
| attn_q | 12288×5120 | 2.09× | 2-bit |
| attn_k/v | 1024×5120 | 1.31× | 2-bit |
| lm_head | 248320×5120 | skipped (pack tax) | 2-bit |

**0 winning shapes.** Mix enabled **0 of 401** linears. Synthetic 0.60–0.72× “wins” on GDN/attn were qdot occupancy artifacts; they did not reproduce on real weights. Mixed greedy **10.21 tok/s**, leftover-identical, France → `Paris`.

**Ruling:** per-shape max equals all-2-bit. `apply_mixed_five_trit` stays in-tree with an empty `TRIT_WIN_NK`. Do not convert the 27B pack. Do not pursue packing further.

### LUT methods

**Hypothesis:** a 243×5 compile-time LUT unpacking 5 trits/byte is cheaper than 2-bit mask-and-accumulate, especially on narrow GDN/attn shapes.

**Measurement:** that LUT **is** the five-trit kernel (`LUT5` in `ternary_trit_qmm`). It is the 1.31–2.14× table above. Native `metal::uint2b_format` (MPP matmul2d 2-bit operand) **does not compile** (`unknown type name 'uint2b_format'`). Stuffing 2-bit into `uint4b` doubles DRAM.

**Ruling:** LUT unpack loses to register qdot on this 10-core GPU. mlp_up qdot is already ALU/occupancy bound (~32–43 GB/s), so extra LUT work cannot be hidden behind a smaller load.

### Hybrid CPU/GPU

**Hypothesis:** decode is almost pure read, so (a) read-only STREAM is 100–110 GB/s not 86.6, and/or (b) 4 P-cores plus the GPU read disjoint buffers and add bandwidth.

**Measurement:** GPU read-only STREAM is **the same band as copy** (85–94 vs 87–94 GB/s). At 512 MiB the copy kernel takes ~2× the wall of the read kernel — 2× the traffic at the same instantaneous bus rate. Disjoint CPU+GPU read: **102.8 GB/s** vs 93.7 GPU-only (+10%). Under contention the GPU drops to **54 GB/s** and the CPU to **49**.

**Ruling:** both hypotheses died. Hybrid decode is ruled out. Do not build a layer-split path.

### Always-8 plus GDN commit

**Hypothesis:** Splash decode is leftover+7 as a fixed 8-row MMA, never a 1-token GEMV. If we always verify 8 rows and commit GDN on the prefix (replay=0), T=8 is ~flat and speculation wins.

**Measurement:** `ternary_qmm_m8` is flat **on that kernel** (MMA T=8 / MMA T=1 = **1.00×**) but **2.25×** vs greedy qdot on MLP-up and **2.68–2.93× e2e**. Leftover+k is already 273–282 ms flat for k=1..7. Deleting replay is assumed in the table below, not built. No K beats 10.22.

**Ruling:** do **not** build always-8 + `verify_gdn_commit`. It predicts 8.71–9.39 tok/s.

### Speculation in general

**Hypothesis:** a trained DFlash 2 draft plus an 8-row MMA verify beats greedy on general chat, the way Splash does on its own pack.

**Measurement:** official Splash `draft/` only (1.266 GB, vocab/hidden match). DFlash is block-diffusion: `propose(k)` always runs leftover+MASK×7, then `select()` walks k — backbone cost is **almost flat in K** (37–46 ms), not K sequential AR steps. Accepts/pass at K=7 is **3.00**, token-identical to leftover greedy. Break-even table (next section) is the formal kill.

**Ruling:** speculation is net-negative on this hardware at the measured acceptance. Available behind `--speculative`, off by default. PLD 11.70 is a copy-prompt number (2.8 accepts at n-gram cost ≈ 0), not a chat number.

## Break-even table

Draft cost is measured. GDN commit (replay=0) is assumed, not built. Accepts/pass at K=2, 3, 7 are leftover-protocol k_sweep (token-identical). Pass = T=8 verify + propose(k). Predicted tok/s = accepts / pass. Min accepts to beat greedy = greedy_tps × pass.

**First pass** (pre-mix, greedy 10.22, T=1 103.3 ms, T=8 277.4 ms = 2.68×):

| K | draft ms | pass ms | accepts/pass | pred tok/s | min accepts vs 10.22 | wins? |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.2 | 315 | — | — | 3.22 | — |
| 2 | 37.9 | 315 | 2.29 | **7.25** | 3.22 | no |
| 3 | 38.5 | 316 | 2.53 | **8.00** | 3.23 | no |
| 4 | 39.1 | 317 | — | — | 3.24 | — |
| 5 | 40.1 | 318 | — | — | 3.25 | — |
| 6 | 41.2 | 319 | — | — | 3.26 | — |
| **7** | **41.9** | **319** | **3.00** | **9.39** | **3.26** | **no** |

A GEMV-only envelope `accepts > 2.25 + K × draft_step` would call K=7 a win (3.00 > 2.68). E2e T=8 is 2.68× a 97.8 ms greedy step, so the table that gates “beat greedy” needs **3.26**. Draft is **0.38–0.43 greedy steps for the whole block**; writing it as K× overstates K=1.

**After per-shape five-trit** (mix enabled 0 linears; T=1 101.7 ms, T=8 298.1 ms = 2.93×; mixed greedy 10.21):

| K | draft ms | pass ms | accepts/pass | pred tok/s | min accepts vs 10.21 | wins? |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 41.9 | 340 | 2.29 | 6.72 | 3.47 | no |
| 3 | 41.7 | 340 | 2.53 | 7.44 | 3.47 | no |
| **7** | **46.2** | **344** | **3.00** | **8.71** | **3.51** | **no** |

Five-trit did not drop T=8 (MMA stays 2-bit). Speculation stayed short. DFlash K=7 e2e with MMA+replay is **5.07 tok/s**, identity-ok.

## The open lead: acceptance gap

If anyone reopens speculation, this is the only remaining lever, and **it would need training** to move. Do not train in this repo. Do not chase another kernel.

DFlash K=7, 48 gen, 16 verify passes, 3.00 accepts/pass, token-identical. Mean **2.00** draft tokens accepted per pass (plus bonus). First-reject slot:

| Drafts accepted before reject | passes | frac |
| ---: | ---: | ---: |
| 0 (slot 0) | 3 | **19%** |
| 1 | 6 | **38%** |
| 2 | 2 | 12.5% |
| 3 | 2 | 12.5% |
| 4 | 1 | 6% |
| 5 | 1 | 6% |
| 6 | 0 | 0% |
| 7 (full K) | 1 | 6% |

Rejects are **front-loaded**, not uniform scatter: **56% of passes die at 0 or 1**. Token 1 is usually right (**81%**) and the block then falls over. That matches a target distribution the drafter was not trained on (2-bit Bonsai vs the dense Qwen3.8-27B target), not a random unpack bug.

Checks already done, not the gap:

- Draft Q4 unpack vs numpy, synthetic max abs **0.044** (bf16 scale truncation + fp16 qmm).
- fp16 DFlash residual: `|h|≈5e4`, ULP 32, **0/308** drafts accepted even when finite. Residual stays float32 because the magnitude requires it.
- Token identity holds, so reject/rollback is not silently diverging.

Published DFlash transfer on a dense target is 4.1–5.5 accepts/pass. 3.00 vs that is the shortfall. Training a drafter on this pack would be the experiment. It was not authorized and is not required to *understand* why speculation loses today.

## Packed codes are genuinely ternary — resolved

Sampled 18 PackedLinear tensors from the on-disk safetensors (no GPU load): `embed_tokens`, `lm_head`, GDN qkv/out, MLP up/down, full-attn q/k/v/o, at layers 0 / 3 / 31 / 63. Affine-2bit g128, 16 codes per uint32. **3,538,944,000 codes.**

| | c0 (−s) | c1 (0) | c2 (+s) | c3 |
| --- | ---: | ---: | ---: | ---: |
| count | 1,197,519,307 | 1,160,343,842 | 1,181,080,851 | **0** |
| fraction | 0.3384 | **0.3279** | 0.3337 | **0** |

Every quantization group in the sample is ternary-only (0% of groups use 4 levels). Exact-zero fraction is the code-1 rate, **32.8%**. The survey’s packing condition is closed: a 5-trits-per-byte pack is a lossless re-encoding. Layout: 26 bytes per g128 (128 codes + 2 pad-ones of code-1). Unpack vs qdot is a separate, failed speed question (tables above).

## MMA kernel (why verify is 2.68×)

Splash decode is leftover+7, `SPLASH_TARGET_VERIFY_ROWS=8`, `matmul2d_descriptor(8, TileN, 64)`. `ternary_qmv_once` is a matvec; handing it 8 rows re-streams weights (~4.8× e2e). The shipped `ternary_qmm_m8` is the right class: 256-thread TGs, K=64 dequant into threadgroup half, then `matmul2d`.

MLP-up 17408×5120, LPM off, stable interleaved:

| Kernel | µs | vs qdot T=1 |
| --- | ---: | ---: |
| T=1 qdot (`ternary_gemv`) | **550** | 1.00× |
| T=8 MMA (shipped) | **1235** | **2.25×** |
| T=1 padded to 8 on the MMA kernel | 1235-class | MMA T=8 / MMA T=1 = **1.00×** |
| T=8 `qmv_once` | 2006 | 3.77× |
| T=8 MLX affine | 1632 | 3.0× |
| STREAM floor (23.7 MB / 72 GB/s that session) | ~330 | |

The 1.3×-vs-qdot gate is **not** met. Dequant-into-TG then `matmul2d` is ~2.2× a register qdot that never writes 16 KB of unpacked halves. Tried and worse or similar: tile-major packing, K=32/128, TileN=32/64/256, `execution_simdgroups<2/4>`, relaxed precision, int8 TG MMA, simdgroup 8×8 MMA, 256-thread unrolled qdot. This is the Splash teardown’s stated 10-core failure mode (software 2-bit × 8 rows, expect ~12 tok/s not 23). Greedy stays on qdot. Verify uses MMA anyway so leftover+k is flat.

Hadamard is not the 4.8×: `fwht` M=1 vs M=8 is ~0.17–0.24 ms.

## Correctness

- **Packing:** synthetic ternary codes packed as MLX uint32×16, `mx.dequantize` vs numpy unpack **max abs 0**. Five-trit pack/unpack vs 2-bit codes **exact**. Five-trit GEMV vs qdot max-abs **0** on 128..2048-wide tests (0.0625 on mlp_up accum order).
- **Hadamard:** `fwht` matches `mx.hadamard_transform` on signed 5120-wide activations (err < 1e-3). Inverse∘forward is an involution within fp16.
- **GEMV vs numpy:** max abs ~0.10 on 17408×5120 (fp16 inputs, f32 accum). vs MLX affine qdot, e2e layer probe **max_abs 0.013**.
- **Weight-once vs stacked GEMV:** max abs 0.016 on 256×512 × M=6.
- **Coherence** (wired in `monkeyinference bench`):
  - “Name the capital of France…” → `Paris`
  - “Explain speculative decoding in two short sentences.” → draft/verify explanation, English, not garbage
  - Long-prefill variant produced the same two-sentence idea as the prior Prism MLX JSON
  - PLD copy prompt reproduced the sentence
- **Token identity** (argmax is lossless if rollback is correct):
  - leftover greedy == `mlx_lm.stream_generate` on Paris
  - leftover greedy == PLD on the copy prompt
  - leftover greedy == early-exit N=4 on explain (48 tok)
  - leftover greedy == DFlash 2 on explain (48 tok, K∈{2,3,7}, also == stream greedy)
  - mixed five-trit greedy == leftover greedy on explain (48 tok; mix enabled 0 linears)
  - leftover greedy == DFlash K=7 after mix (48 tok, reject histogram recorded)

A fast engine emitting plausible-looking noise, or a spec loop that silently diverges on reject, would have failed this loop. It did not.

Tokenizer still warns about the Mistral-Small regex. Outputs were English and correct on these prompts; a tokenizer-sensitive eval is still owed.

## What's in the repo

```
src/monkeyinference/
  kernels.py     Metal STREAM copy/read/write + qdot GEMV + 8-row MMA + five-trit LUT GEMV
  trit.py        5-trits-per-byte pack/unpack + per-shape mix gate (empty win table)
  stream_cpu.c   P-core STREAM read for the CPU+GPU aggregate measurement
  hadamard.py    Prism fwht
  packed.py      PackedLinear / PackedEmbedding (qdot default; MMA for M=2..8)
  load.py        text-only safetensors → mlx_lm Qwen3.5 TextModel
  generate.py    greedy (mlx_lm.stream_generate) + leftover + PLD/early/dflash spec
  spec.py        n-gram drafter, early-exit drafter, CoW GDN/KV pins
  splash_q4.py   MDFD0004 unpack → MLX affine-4bit
  dflash.py      5-layer DFlash 2 + selector + Bonsai aux hiddens
  roofline.py    STREAM + GEMV + the 7.674 GB math
  bench.py       coherence + same-prompt PLD vs greedy + identity + early-exit + `--dflash`
scripts/
  probe_mixed.py, probe_five_trit.py, probe_breakeven.py
  probe_bandwidth.py, probe_code_histogram.py, probe_gdn_path.py, probe_qmm_m8.py
```

Weights stay at `~/.monkey/models/Ternary-Bonsai-2-27B-mlx-2bit/`. The DFlash draft (optional) is `~/.monkey/models/Qwen3.8-27B-Splash-draft/draft/`. The repo is source only.

```bash
export PYTHONPATH=src
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli generate
~/.monkey/mlx-venv/bin/python tests/test_kernels.py
~/.monkey/mlx-venv/bin/python tests/test_spec.py
~/.monkey/mlx-venv/bin/python tests/test_splash_q4.py
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench --out results/bench.json
```

`--speculative` turns on a draft (net-negative on this hardware). `--no-custom` forces Prism MLX matmul. `--mix-five-trit` applies the empty per-shape gate.

## Closed

Not a backlog. These are finished measurements.

1. **`ternary_qmm_m8`.** Shipped. MMA is flat. 2.25× GEMV / 2.68–2.93× e2e vs qdot. Greedy stays on qdot.
2. **Always-8 + GDN commit.** Not building. Replay=0 still needs 3.26–3.51 accepts; we have 3.00.
3. **DFlash Q4 MMA.** Draft already 37–46 ms. Skip.
4. **Five-trit.** HOLD. Global 0.88×; per-shape 0 winners; mixed greedy 10.21.
5. **Acceptance gap** (3.00 vs published 4.1–5.5). Front-loaded rejects. Would need training. Did not train.
6. **Hybrid CPU/GPU.** Ruled out.
7. **Chunked GDN.** Same sequential kernel. Ruled out.
8. Prefill 40.9 vs llama-bench pp512 ~47 — not the chat bottleneck.

Do not: train a drafter, tree attention / EAGLE / Medusa, fuse the Hadamard, port PowerInfer / Deja Vu, rewrite in Swift, WY-chunked GDN, `uint4b` 2-bit padding, hybrid CPU/GPU decode, always-8 GDN commit, five-trit convert, Apple10 persistent-wave policy, post-Hadamard sparsity unless a later owner measures 30%+ droppable with token identity.

## Disk / cleanliness

Source: `~/projects/monkeyinference` (git). Authorized add: **+1.266 GB** Splash `draft/` only. Did not download `target/`, `vision/`, or `incoai/Qwen3.8-27B-DFlash2`. Tokenizer compare used a 20 MB Hub file in `/tmp` and deleted it. Did not delete user data.

Safe to delete if Ashvin wants the bytes back — see the README disk section. The 8.0 GB MLX pack is required to run this engine. The 6.7 GB GGUF and the llama-server on port 8080 are **not** this repo; leave them unless he is done with the llama.cpp path.

## Rulings log

- 2026-09-18 **HOLD** bandwidth-bound decode. Greedy 10.22 tok/s is 80–90% of a read-like STREAM / 7.254 GB. Read-only is **not** 100–110 GB/s on this Air.
- 2026-09-18 **HOLD** speculation required to beat ~11 tok/s in general chat. PLD 11.70 is a copy-prompt number (2.8 accepts/pass), not a chat number. **UPDATE 2026-09-19:** DFlash+MMA at 3.00 accepts does not beat 10.22 (see break-even table). Speculation is net-negative on this hardware.
- 2026-09-18 **UPDATE** custom qdot GEMV **is** production decode. `--no-custom` remains the Prism MLX path. Layer vs MLX max_abs 0.013; greedy tokens match.
- 2026-09-18 **UPDATE** Ashvin authorized Splash `draft/` bytes. Fetched **1.266 GB** only. Vocab/hidden match. Do **not** fetch the 17.4 GB package or the 3.85 GB BF16 DFlash2 repo.
- 2026-09-18 **HOLD** self-speculation with first N layers + shared `lm_head` plateaus at **1.07 accepts/pass** for N∈{2,4,6,8}.
- 2026-09-18 **NEW** Splash DFlash 2 on Bonsai explain is **3.00 accepts/pass**, token-identical to leftover greedy. Modelling lead ruled in as a *transfer*; it is not enough to beat greedy.
- 2026-09-18 **NEW** GDN snapshot tax is gone (CoW pin). Weight-once verify is why PLD copy is 11.70 vs 8.19 leftover greedy, not 7.7 vs 8.4.
- 2026-09-18 **HOLD** leftover 2.77 vs greedy 9.74 was **Low Power Mode**, not the DFlash residual. Same-run leftover 8.47 vs greedy 10.22 with LPM off. Leftover never runs the draft stream.
- 2026-09-18 **HOLD** DFlash residual is float32 because `|h|≈5e4`. fp32 `o_proj` accum + fp16 store is finite and drops accepts to 0 (fp16 ULP 32).
- 2026-09-18 **UPDATE** DFlash loop: incremental context KV, query pin, replay without `lm_head`, default K=2. Explain tok/s **7.32** (K=7 still 3.00 accepts/pass). Does not beat 10.22.
- 2026-09-19 **HOLD** GPU **read-only STREAM is the same band as copy** (85–94 vs 87–94 GB/s). Copy wall is ~2× read wall at 512 MiB — same instantaneous bus, 2× traffic. 1 GiB read median 85 GB/s. Decode resembles GPU read. Do not retarget the ceiling to 100–110 GB/s.
- 2026-09-19 **NEW** CPU+GPU disjoint read aggregates **103 GB/s** vs GPU-only 94 (+10%); GPU drops to 54 under contention. llama.cpp layer-split is existence proof. **Hybrid decode ruled out.**
- 2026-09-19 **HOLD** mlx_lm GDN prefill and verify are the **same sequential kernel**. Isolated GDN 12% of T=8. WY-chunked GDN remains out of scope.
- 2026-09-19 **UPDATE** T=8 at 4.8× T=1 was the **wrong kernel class** (`ternary_qmv_once` is a matvec). `ternary_qmm_m8` is the Splash-shaped MMA: **MMA T=8 / MMA T=1 = 1.00×**, **2.25× vs greedy qdot** on MLP-up / **2.68× e2e**. `uint2b_format` does not compile.
- 2026-09-19 **NEW** DFlash break-even with measured draft and replay=0: propose is **37–42 ms flat in K**. No K beats 10.22. Best **9.39 tok/s** at K=7 (3.00 accepts; need **3.26**). Do not build always-8+GDN commit.
- 2026-09-18 **NEW** Bonsai affine-2bit codes are genuinely **ternary**: **0 / 3.54e9** code-3. Exact-zero (code 1) **32.8%**. **RESOLVED** — five-trit is lossless on the table; unpack vs qdot is a separate (failed) speed question.
- 2026-09-19 **HOLD** five-trit unpack. Synthetic global switch 0.88×. **Per-shape mix on real weights: 0 winning (N,K)** (1.31–2.14×). Mixed greedy **10.21 tok/s**, leftover-identical. Keep 2-bit. Does not move T=8 MMA.
- 2026-09-19 **NEW** After mix, break-even still short: T=8 298 ms, need **3.51** accepts, have 3.00, predicted **8.71 tok/s**. DFlash K=7 e2e **5.07 tok/s** with MMA+replay, identity-ok. Rejects front-loaded (19% slot 0, 38% after one). Acceptance gap, not kernels. Did not train.
- 2026-09-19 **CLOSE** no remaining software path on this Air materially beats ~11 tok/s. Binding constraint is GPU STREAM ~86.6 GB/s. Engine is at the hardware’s greedy limit. Faster means more memory bandwidth (a different Mac) or training a draft — not another kernel in this repo.
