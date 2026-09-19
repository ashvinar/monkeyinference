# Ternary engine for Bonsai 2 27B on this M4 Air

Living design-and-results doc. Engine repo: [github.com/ashvinar/monkeyinference](https://github.com/ashvinar/monkeyinference) · branch `cursor/ternary-metal-engine-09ea`. Standalone Metal/MLX runtime, not a Splash fork.

**Headline (2026-09-19):** Decode is DRAM-bandwidth bound. Greedy explain is **10.22 tok/s** (LPM off). Read-only STREAM is **85–94 GB/s**, the same band as copy STREAM **87–94 GB/s** (historical 86.6) — **not** 100–110. Ceiling stays **~11.3–12.7 tok/s**. 10.22 is ~80–90% of it. **Hybrid CPU/GPU is ruled out.**

Splash-shaped spec is always-8 MMA plus a block-diffusion draft. MMA is flat on that kernel (T=8/T=1 = 1.00×) but **2.68× e2e** vs qdot T=1 (277 / 103 ms; GEMV-only 2.25× on MLP-up). DFlash propose is leftover+MASK×7, **37–42 ms, almost flat in K**. With GDN commit (replay=0), **no K beats 10.22**: best is K=7 at **9.39 tok/s** (3.00 accepts/pass; need **3.26**). Speculation is formally dead at the current acceptance. If reopened, chase 3.00 → 4.1 accepts, not another kernel.

Five-trit is lossless (0 of 3.54e9 code-3) and **not** the 11.5–13 greedy lever: census-weighted **0.88×** vs qdot, predicted **9.00 tok/s**. Keep 2-bit qdot. `uint2b_format` does not compile. Do not build always-8+`verify_gdn_commit` — it predicts 9.39.

## Machine

| | |
| --- | ---: |
| Computer | MacBook Air (Mac16,12) |
| Chip | Apple M4, 10 CPU (4P+6E), 10 GPU, Metal 4 |
| Unified memory | 24 GB |
| Published DRAM | **120 GB/s** (Apple M4 Air spec, LPDDR5X) |
| GPU STREAM copy | **87–94 GB/s** (median **89** at 512 MiB; historical 86.6). Counts read+write. |
| GPU STREAM read | **85–94 GB/s** (median **92** at 512 MiB, **85** at 1 GiB). Reduce-to-scalar; decode-like. |
| GPU STREAM write | **75–84 GB/s** fill |
| CPU STREAM read | **72 GB/s** (4 P-cores, QoS USER_INTERACTIVE, 512 MiB) |
| CPU+GPU disjoint read | **103 GB/s** aggregate (GPU 54 + CPU 49). GPU-only 94 in that session. **Ruled out as a decode plan.** |
| STREAM / published | copy ~74%; read ~71–78%; hybrid aggregate 86% |
| OS | macOS 27.0 |
| Free disk | **~16–18 GB** after adding **1.266 GB** Splash `draft/` only. |

## Roofline

Language tensors in the MLX pack: **7.674 GB** (6.822 GB 2-bit codes + 0.420 GB scales + 0.420 GB redundant biases + 0.012 GB Hadamard signs). Vision tower 0.921 GB is skipped. The custom kernel does not load biases, so the decode stream is **7.254 GB** of codes+scales+signs.

Each greedy decode token streams the language weights once (GEMV), plus a negligible activation/KV term at short context.

| Assumption | tok/s = GB/s ÷ bytes |
| --- | ---: |
| Published 120 GB/s, with biases (7.674 GB) | **15.6** |
| Published 120 GB/s, drop biases (7.254 GB) | 16.5 |
| GPU copy 86.6 GB/s (historical), with biases | **11.3** |
| GPU copy 86.6 GB/s, drop biases | **11.9** |
| GPU read 92 GB/s (512 MiB median), drop biases | **12.7** |
| GPU read 85 GB/s (1 GiB median, closer to 7 GB), drop biases | **11.7** |
| CPU+GPU aggregate 103 GB/s, drop biases | 14.2 *(not a plan; see below)* |
| Starting Prism MLX decode | **8.0** |
| This engine greedy explain (qdot, no bias load) | **10.22** (9.74 same band) |
| 10.22 / 11.3 historical copy ceiling | **90%** |
| 10.22 / 11.7 1 GiB-read ceiling | **87%** |
| 10.22 / 12.7 512 MiB-read ceiling | **80%** |
| PLD copy-prompt (same prompt as leftover greedy) | **11.70** |
| five-trit briefing (5.975 GB, copy 86.6) | 14.5 *(not realized; unpack HOLD)* |
| five-trit census-weighted prediction | **9.00** (0.88× qdot) |
| DFlash K=7 always-8 MMA, replay=0, 3.00 accepts | **9.39** predicted; need 3.26 to beat 10.22 |

**Read vs copy.** Ashvin’s objection is right in principle: copy STREAM bills read+write, decode is almost pure read. On this Air it does **not** raise the ceiling into 13–14 tok/s. A Metal reduce-to-scalar (4096 TGs × 256 threads, stride over the buffer, `simd_sum` + one store per TG so loads cannot be DCE’d) lands **in the same band as copy**. At 512 MiB, billed copy 89 GB/s vs read 92 GB/s: the copy kernel takes ~2× the wall of the read kernel, which is exactly 2× the traffic at the same instantaneous bus rate. At 1 GiB, read median **85 GB/s** is slightly *below* copy **94 GB/s**. Thermal swing is ±10 GB/s; do not treat a single 93.7 as 110. **Decode resembles GPU read of a many-GB stream. Keep 11.3–12.7 as the greedy ceiling. 10.22 still has a few percent of STREAM slack, not a second qdot project’s worth.**

**CPU+GPU aggregate.** 4 P-core C threads (clang -O3, 8-wide float add, QoS USER_INTERACTIVE) on a **disjoint** 512 MiB buffer while the GPU read kernel runs: **102.8 GB/s** combined vs **93.7 GPU-only** that session (+10%). Under contention the GPU drops to **54 GB/s** and the CPU to **49** (CPU-only was 72). The extra bytes are real; llama.cpp layer-split is existence proof of the mechanics. **Hybrid decode is ruled out.** Both hypotheses died: read-only is not 100–110 GB/s, and the aggregate does not survive contention. Do not build a layer-split path.

Five-trit packing at 5.975 GB (26/32 of the 2-bit codes, g128 padded to 130): briefing 86.6/5.975 = 14.5 tok/s. **Realized HOLD.** Census-weighted unpack GEMV is **0.88×** 2-bit qdot (mlp_up 2.02×; some GDN/attn shapes 0.60–0.72×). Predicted greedy **9.00 tok/s**. The 11.5–13 survey number assumed the unpack was cheaper than the bytes saved. On this 10-core GPU mlp_up qdot is ALU/occupancy bound (~32–43 GB/s), so extra LUT unpack loses. Keep 2-bit qdot. The packer and kernel stay in-tree behind `PackedLinear.enable_five_trit` with tests; they are not production decode.

Profiling **does not contradict** the two original rulings:

1. **Decode is memory-bandwidth bound.** 10.22 tok/s is most of a **read-like** STREAM of 7.254 GB (80–90% depending on 85 vs 92 GB/s). The qdot pass recovered the ~1.4 tok/s sitting in `affine_qmv_fast` occupancy / bias traffic, not a 2–3×. Microbenches on this 10-core GPU still swing ±10 GB/s / ±30% kernel; e2e numbers above are the ones that matter.
2. **Beating the ceiling requires speculative decode** whose verify is an 8-row matmul, not a matvec handed 8 rows. On this GPU that verify is **2.68×** greedy e2e. With DFlash at 3.00 accepts/pass and a 37–42 ms draft, predicted **9.39 tok/s** does not beat 10.22. PLD on a copyable prompt still crosses 11.7 because it commits 2.8 tokens per target pass at copy-prompt n-gram cost ≈ 0. That is *not* a general chat number. Self-speculation off Bonsai's own early layers does not provide a cheap general draft. **Speculation is formally dead for general chat at the current acceptance.**

Prism's laptop table is llama-bench tg128 depth 0 (no spec): M4 Pro 18 tok/s, M5 Pro 28, M5 Max 47. 10.22 on 86–92 GB/s is **in family**. Prism's Bonsai-demo is PQ2_0 llama.cpp or stock `mlx_vlm` + Hadamard, **no drafter**. There is no vendor fast path we missed.

## Architecture decisions

| Decision | Why | Override |
| --- | --- | --- |
| Standalone Python+MLX repo, not a Splash fork | Splash factory has no AR fallback and only loads `splash-packed-q4` + DFlash. Bonsai is affine-2bit Hadamard safetensors. | Rewrite in Swift/Metal if Python dispatch shows up in traces now that GEMV is STREAM-bound. |
| Text-only load | Vision tower is 0.92 GB FP16 and unused for these prompts. Peak Metal ~8.3 GB. | Keep the tower behind an explicit `--vision` flag later. |
| Keep Prism activation Hadamard (block 1024, `H(x⊙s)/√1024`) | Weights are stored in that basis. Skipping it is silent garbage. | None. |
| Default matmul = custom qdot GEMV (`--no-custom` for MLX) | Clone of MLX `qmv_fast`: 64-thread TGs, 2 simdgroups × 4 rows, pre-shifted x, mask-and-accumulate, no bias load (`y = s·(codes·x − Σx)` because Prism stores `bias = −scale`). Hadamard stays `mx.hadamard_transform` immediately before the GEMV (10 KB; inlining a 1024-point FWHT into an output-tiled TG would recompute H(x) per 8 output rows). E2e greedy 8.38 → **9.74**. Layer-level vs MLX `max_abs` 0.013. | `--no-custom`. |
| Spec verify (M=2..8) = `ternary_qmm_m8` | Splash decode is leftover+7 as an 8-row MMA. 256-thread TGs, N/128 tiles, K=64 dequant into TG half, then `matmul2d`. Pads M<8. **MMA T=8 / MMA T=1 = 1.00×.** Versus greedy qdot: **2.25×** GEMV / **2.68× e2e**. M=9..16 still `qmv_once`. Prefill M>16 stays on `mx.quantized_matmul`. Leftover+k is already flat at 273–282 ms for k=1..7. | Do not pad 2-bit into `uint4b`. Do not build GDN-commit always-8 — predicts 9.39. |
| Hybrid CPU/GPU decode | Measured 103 GB/s aggregate; GPU halves under contention. | **Ruled out.** |
| Five-trit greedy pack | Lossless (0 code-3). 26 bytes / g128 (−18.75%). | **HOLD.** Unpack 0.88× qdot census-wide. Keep 2-bit. |
| GDN cache pin is copy-on-write, not memcpy | `gated_delta` already writes a new `state_out`; `GatedDeltaNet` does `cache[i] = new`. Holding the previous array refs / KV offsets is a real CoW pin (~151 MB × 48 layers is **not** copied). Partial reject reverts pointers and replays leftover+accepted (GDN is recurrent; no per-timestep states). | If Apple adds trimmable GDN cache, switch. |
| First draft = prompt-lookup n-gram (PLD), K=5 | Zero extra weights. K=8+ over-proposes on the copy sentence and pays reject+replay. | Tune K per prompt class. |
| Self-spec draft = first N layers + shared `lm_head` | Vocab 248320 matches by construction; 0 new bytes. N∈{2,4,6,8} all plateau at ~1.07 accepts/pass. | Do not tune N further. |
| Chat draft = Splash DFlash 2 `draft/` | Official 5-layer block-diffusion drafter trained on Qwen3.8-27B. Vocab/hidden match. Context is concatenated Bonsai hiddens at layers 5/19/33/47/61. Logits use Bonsai's `lm_head`. Residual **must** stay float32: `|h|≈5e4` (fp16 ULP 32) zeros the draft even when finite. Default verify K=2 (GDN cost scales with T; typical accept is leftover+2). Draft KV is incremental; query KV is pinned and dropped; reject replay skips `lm_head`. | `--draft dflash --num-draft 7` for the 3.00 accepts/pass number. Do not download the 17.4 GB Splash package or the 3.85 GB BF16 DFlash2 repo. |
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
| **greedy explain (qdot, LPM off)** | 32 | 59 | 34.1 | **9.74** | **8.24 GB** | same Prism explanation, token-stable |
| **greedy long (qdot, LPM off)** | 717 | 60 | **40.9** | 9.33 | 10.41 GB | same two-sentence idea |
| **leftover greedy copy (LPM off)** | 46 | 14 | 30.1 | **8.19** | 8.38 GB | exact copy |
| **PLD copy (CoW + weight-once, LPM off)** | 46 | 14 | 29.9 | **11.70** | 8.38 GB | exact copy; **10/10**; **1.43× vs leftover greedy**; token-identical |

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

Explain prompt, leftover protocol, 48 gen tokens, argmax, **Low Power Mode off**, token-identical to leftover greedy and to stream greedy:

| Engine | K | Accepts/pass | Accepted/proposed | Verify passes | Decode tok/s | draft/verify/replay s |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stream greedy (same load) | — | 1.00 | — | 48 | **10.22** | — |
| leftover greedy | — | 1.00 | 0 / 0 | 48 | 8.47 | 0 / 5.66 / 0 |
| DFlash 2 | 7 | **3.00** | **32 / 103** | 16 | 3.99 | 1.31 / 8.21 / 2.49 |
| DFlash 2 | 3 | 2.53 | 29 / 55 | 19 | 7.03 | 1.08 / 4.37 / 1.37 |
| **DFlash 2 (default)** | **2** | **2.29** | **27 / 41** | **21** | **7.32** | 1.38 / 3.77 / 1.39 |

The leftover **2.77 tok/s** in the earlier DFlash table was the same Air with `lowpowermode 1` on battery. Same-run greedy was 2.78. STREAM was still 79 GB/s; GPU clocks were not. That is not an fp32-residual tax on leftover — leftover never runs the draft.

fp32 `o_proj` accum + **fp16 residual** is finite (`|h|≈53085 < 65504`) and **wrong**: 0/308 drafts accepted. fp16 ULP at 5e4 is 32. Residual stays float32 because the magnitude requires it, not because we failed to isolate the bias.

Target forward time after a short prefill, CoW-pinned (no aux tax — taps add 0 ms):

| M (leftover+drafts) | median |
| ---: | ---: |
| 1 | 114.5 ms |
| 2 | 140.4 ms |
| 3 | 190.0 ms |
| 4 | 253.6 ms |
| 8 | 553.3 ms |

Weight-once qdot does not make T=8 as cheap as T=1. That table is **pre-MMA** (`qmv_once`). Default K=2 is leftover+2 = T=3 (~190 ms on that kernel). Incremental draft KV cut propose from 9.6 s → 1.3–1.4 s. Replay skips `lm_head`. **7.32 tok/s does not beat 10.22.**

### Break-even: always-8 MMA + measured draft, replay=0 (LPM off)

Draft cost is measured, not estimated. DFlash `propose(k)` always runs leftover+MASK×7 then `select()` walks k, so the backbone is **one 8-row pass**, not K sequential AR steps. GDN commit (replay=0) is assumed, not built. Accepts/pass at K=2,3,7 are the leftover-protocol k_sweep (token-identical). Greedy step = 1000/10.22 = **97.8 ms**.

CoW-pinned explain leftover, model + `lm_head`:

| Forward | median |
| --- | ---: |
| T=1 qdot | **103.3 ms** |
| T=8 MMA (leftover+7) | **277.4 ms** |
| T=8 / T=1 | **2.68×** |
| leftover+k MMA, k=1..7 | **273–282 ms** (already flat) |

Isolated GDN ×48 layers (commit-cost proxy; already inside the 277 ms verify):

| T | ms ×48 |
| ---: | ---: |
| 1 | 16.9 |
| 3 | 19.4 |
| 8 | 31.9 |

DFlash `propose(k)` after prefill context KV is committed:

| K | draft ms | n_ids |
| ---: | ---: | ---: |
| 1 | 37.2 | 1 |
| 2 | 37.9 | 2 |
| 3 | 38.5 | 3 |
| 4 | 39.1 | 4 |
| 5 | 40.1 | 5 |
| 6 | 41.2 | 6 |
| 7 | 41.9 | 7 |

Pass = T=8 verify + propose(k). Predicted tok/s = accepts / pass. Min accepts to beat 10.22 = 10.22 × pass.

| K | draft ms | pass ms | accepts/pass | pred tok/s | min accepts vs 10.22 | GEMV formula `2.25 + K·draft_step` | wins? |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 37.2 | 315 | — | — | 3.22 | 2.63 | — |
| 2 | 37.9 | 315 | 2.29 | **7.25** | 3.22 | 2.64 | no |
| 3 | 38.5 | 316 | 2.53 | **8.00** | 3.23 | 2.64 | no |
| 4 | 39.1 | 317 | — | — | 3.24 | 2.65 | — |
| 5 | 40.1 | 318 | — | — | 3.25 | 2.66 | — |
| 6 | 41.2 | 319 | — | — | 3.26 | 2.67 | — |
| **7** | **41.9** | **319** | **3.00** | **9.39** | **3.26** | 2.68 | **no** |

The back-of-envelope `accepts > 2.25 + K × draft_step` uses the MLP-up GEMV ratio and would call K=7 a win (3.00 > 2.68). E2e T=8 is **2.68×** T=1, and 10.22 is 97.8 ms not 103.3, so the table that gates “beat greedy” needs **3.26**. Draft is **0.38–0.43 greedy steps for the whole block**; writing it as K× overstates the K=1 cost and matches K=7 only because propose is flat.

**Ruling:** no K beats 10.22 on paper. Best predicted **9.39 tok/s** at K=7 (8% short). Do **not** build always-8 + `verify_gdn_commit` — leftover+k is already MMA-flat, and deleting replay still lands at 9.39. A Q4 MMA that cut draft 42→15 ms would make pass 292 ms, min accepts 2.98, and 3.00 would “win” 10.27 — thermal noise, not a project. If spec is reopened, chase the **acceptance gap** (3.00 vs published 4.1–5.5). At 4.1 / 0.319 s ≈ **12.9 tok/s** on this cost model.

The 3.99 tok/s K=7 e2e row above is the pre-MMA loop (verify 513 ms/pass `qmv_once` + 155 ms replay). MMA without replay is the 9.39 prediction. Peak Metal this session **9.51 GB**.

### GDN path: prefill and verify are the same sequential kernel

mlx_lm `Qwen3_5.GatedDeltaNet.__call__` always calls `gated_delta_update` → Metal `gated_delta_kernel` (`use_kernel=not self.training`). The kernel source is one threadgroup per `(batch, value-head)` with an explicit

```c
for (int t = 0; t < T; ++t) { /* decay, delta, write y[t] */ }
```

The ops fallback is documented as “prompt prefill (sequential loop)” and does `for t in range(T)`. There is no chunkwise-parallel / WY / block-scan form in mlx_lm 0.31.3.

Hooked on a real Bonsai forward (LPM off):

| Forward | Calls | unique T | kernel? | wall |
| --- | ---: | --- | --- | ---: |
| leftover prefill (31 tok) | 48 | {31} | yes | 0.88 s |
| T=1 decode | 48 | {1} | yes | 132 ms |
| T=8 leftover+K | 48 | {8} | yes | 582 ms |

`ArraysCache` is written once per forward (`cache[1] = state`; `cache.advance(S)`), not once per token. Verify is not stepping GDN in Python.

Isolated `gated_delta_kernel` at Bonsai shapes (Hv=48, Dv=Dk=128), ×48 layers:

| T | ms / 48 layers | ms / token |
| ---: | ---: | ---: |
| 1 | 30.2 | 30.3 |
| 8 | 65.0 | 8.1 |
| 64 | 167.4 | 2.6 |
| 512 | 295.9 | 0.58 |

GDN is **65 ms of a 550 ms T=8 model forward** (12%). T=8 path counts: **401× `ternary_qmv_once`, 0× qmm**. Forcing leftover+K onto the prefill linear path (`mx.quantized_matmul`) makes T=8 **863 ms** (MLX stays on `qmv` until M≈12 and re-streams). Prefill 40.9 tok/s (~24 ms/token on 717 tok) is large-M qmm amortizing 7.25 GB over T=512 plus GDN at 0.58 ms/token. Leftover+K cannot take that deal.

Kernel vs ops on one layer: T=1 y max-abs 1.2e-4; T=8 y max-abs 0.031 (fp16). Same recurrence.

**Ruling:** a chunked verify that is just “the prefill GDN path” does not exist as a second implementation. A trainable WY-chunked gated-delta kernel would be new work, would still leave 401 ternary GEMVs, and is out of scope.

### Small-M GEMV: not near-flat (LPM off, pipelined)

Per-call `mx.synchronize` made a first pass look ~flat (launch latency ≈ kernel). Reran as a pipelined stream (warmup, then N `mx.eval` under one sync) on every PackedLinear shape in a Bonsai forward, including `lm_head`. Bytes below are codes+scales+x+y. STREAM floor is 86.6 GB/s. `mlp_gate` is the same geometry as `mlp_up` and was the first kernel of the run (cold); use `mlp_up` as the warmed number for that shape.

| Shape | N×K | count | bytes once (M=1 / M=8) | M=1 us | M=8 us | M=8/M=1 | GB/s if W once (M=1 / M=8) | GB/s if W×M at M=8 | STREAM us if once |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp_up (warm) | 17408×5120 | 64 | 23.72 / 24.04 MB | **552** | **1982** | **3.59** | 43.0 / 12.1 | 95.7 | 274 |
| mlp_down | 5120×17408 | 64 | 23.72 / 24.04 MB | 488 | 2061 | **4.22** | 48.6 / 11.7 | ~93 | 274 |
| gdn_qkv | 10240×5120 | 48 | ~14 MB | 372 | 1315 | 3.54 | 37.6 / 10.8 | |  |
| gdn_z | 6144×5120 | 48 | | 346 | 911 | 2.63 | 24.2 / 9.4 | |  |
| gdn_out | 5120×6144 | 48 | | 309 | 924 | 2.99 | 27.1 / 9.2 | |  |
| attn_q | 6144×5120 | 16 | | 309 | 947 | 3.06 | 27.1 / 9.0 | |  |
| attn_k | 1024×5120 | 16 | | 209 | 319 | 1.53 | 6.7 / 4.7 | launch-bound |  |
| attn_v | 1024×5120 | 16 | | 199 | 287 | 1.44 | 7.1 / 5.2 | launch-bound |  |
| attn_o | 5120×6144 | 16 | | 336 | 889 | 2.65 | 25.0 / 9.6 | |  |
| lm_head | 248320×5120 | 1 | 338 / 341 MB | **4283** | **22576** | **5.27** | **79.0 / 15.1** | ~119 | 3900 |

Census-weighted (401 linears): predicted GEMV **220 ms at M=1, 689 ms at M=8, ratio 3.13**. Inflated vs in-model T=1 (~84 ms of linears inside 114 ms) because isolated GEMVs do not overlap GDN/Hadamard and `mlp_gate` was cold. In-model T=8 is **553 ms vs 114 ms = 4.83×**.

**Ruling:** amortization of `ternary_qmv_once` is **broken because it is a matvec kernel**, not because a tile is “subtly” failing to keep weights in registers. Splash never runs a 1-token GEMV: decode is leftover+7, `SPLASH_TARGET_VERIFY_ROWS=8`, `matmul2d_descriptor(8, TileN, 64)`, 256-thread TGs.

**`ternary_qmm_m8` (LPM off, stable interleaved, MLP-up 17408×5120):**

| Kernel | µs | vs qdot T=1 |
| --- | ---: | ---: |
| T=1 qdot (`ternary_gemv`) | **550** | 1.00× |
| T=8 MMA (shipped) | **1235** | **2.25×** |
| T=1 padded to 8 on the MMA kernel | 1235-class | MMA T=8 / MMA T=1 = **1.00×** |
| T=8 `qmv_once` | 2006 | 3.77× |
| T=8 MLX affine | 1632 | 3.0× |
| STREAM floor (23.7 MB / 72 GB/s that session) | ~330 | |

The kernel **class** is right: eight rows cost the same as one *on the MMA*. The **1.3× vs greedy qdot** gate is **not** met. Dequant-into-TG then `matmul2d` is ~2.2× a register qdot that never writes 16 KB of unpacked halves. Tried and worse or similar: tile-major packing, K=32/128, TileN=32/64/256, `execution_simdgroups<2/4>`, relaxed precision, int8 TG MMA, simdgroup 8×8 MMA, 256-thread unrolled qdot. `metal::uint2b_format` is listed in the MPP matmul2d tables and **does not compile** (`unknown type name 'uint2b_format'`). Stuffing 2-bit into `uint4b` still doubles DRAM — do not.

This is the Splash teardown’s stated 10-core failure mode (software 2-bit × 8 rows, expect ~12 tok/s not 23). Greedy stays on qdot. Verify M=2..8 uses MMA anyway: leftover+k is **273–282 ms flat**, 2.68× T=1 e2e. Always-8 + GDN commit would not beat 10.22 at 3.00 accepts (see break-even table).

Hadamard is not the 4.8×: `fwht` M=1 vs M=8 is ~0.17–0.24 ms and does not track T.

### Packed codes are genuinely ternary

Sampled 18 PackedLinear tensors from the on-disk safetensors (no GPU load): `embed_tokens`, `lm_head`, GDN qkv/out, MLP up/down, full-attn q/k/v/o, at layers 0 / 3 / 31 / 63. Affine-2bit g128, 16 codes per uint32. **3,538,944,000 codes.**

| | c0 (−s) | c1 (0) | c2 (+s) | c3 |
| --- | ---: | ---: | ---: | ---: |
| count | 1,197,519,307 | 1,160,343,842 | 1,181,080,851 | **0** |
| fraction | 0.3384 | **0.3279** | 0.3337 | **0** |

Every quantization group in the sample is ternary-only (no code 3, 0% of groups use 4 levels). Exact-zero fraction is the code-1 rate, **32.8%**, stable across tensors (embed is a hair heavier on c0: 0.343 / 0.328 / 0.329). Layer 31 has no `linear_attn` (it is full-attn); GDN is represented by layer 0.

**Read:** a 5-trits-per-byte pack (`3^5 = 243 < 256`) is a **lossless re-encoding**. Layout: 26 bytes per g128 (128 codes + 2 pad-ones), 26/32 = **−18.75%** vs 2-bit. Codes 6.822 GB → 5.543 GB; stream with scales+signs ≈ **5.975 GB**. Briefing STREAM 86.6 / 5.975 ≈ **14.5 tok/s**.

**HOLD after unpack microbench (LPM off, pipelined, vs the same-run qdot):**

| Shape | N×K | trit/qdot | notes |
| --- | ---: | ---: | --- |
| mlp_up | 17408×5120 | **2.02×** | ALU-bound qdot ~32 GB/s; unpack loses |
| mlp_down | 5120×17408 | 1.12× | |
| gdn_qkv / gdn_z / attn_q / attn_o | | **0.60–0.72×** | some wins |
| lm_head (8192-row slice, scaled) | 248320×5120 | 1.86× | |
| census-weighted (401 linears) | | **0.88×** | predicted greedy **9.00 tok/s** |

Pack roundtrip is exact. GEMV vs qdot max-abs 0 on 1024×2048, 0.0625 on mlp_up (fp16 accum order). Do **not** convert the 27B pack. Kernel + `PackedLinear.enable_five_trit` stay in-tree; production decode stays 2-bit qdot. The 11.5–13 survey number assumed unpack cheaper than the bytes saved. It is not, on this GPU, on the shapes that dominate the stream.

### LPM-throttled rows (do not compare to 10.22)

Measured with `lowpowermode 1` on battery. STREAM was still ~79 GB/s; GPU clocks were not.

| Engine | tok/s | Notes |
| --- | ---: | --- |
| leftover greedy explain | 2.77 | same process greedy 2.78 |
| DFlash K=7 first loop | 0.80 | 3.00 accepts/pass, identity-ok; rebuild KV |

The 9.74 greedy / 11.70 PLD table above was LPM off (same band as 10.22).

### Microbench (no 27B load)

STREAM 86.6 GB/s. GEMV numbers on this Air swing with thermal; treat as ±30% and prefer the e2e table. After the qdot pass, MLP-up is tied-to-ahead of MLX, attn-q ahead, lm_head slice well ahead (wide N loves 64-thread TGs). E2e greedy +1.4 tok/s is the claim, not a single-shape GB/s.

MLX itself will not switch `qmv`→`qmm` on Bonsai MLP-up until M≈12. Spec leftover+5 is M=6, so flattening `[1,K,H]→[K,H]` is necessary but not sufficient; the weight-once kernel is what actually streams W once.

## Correctness

- **Packing:** synthetic ternary codes packed as MLX uint32×16, `mx.dequantize` vs numpy unpack **max abs 0**. Five-trit pack/unpack vs 2-bit codes **exact**. Five-trit GEMV vs qdot max-abs **0** on 128..2048-wide tests (0.0625 on mlp_up accum).
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
  - leftover greedy == DFlash 2 on explain (48 tok, K∈{2,3,7}, also == stream greedy)

A fast engine emitting plausible-looking noise, or a spec loop that silently diverges on reject, would have failed this loop. It did not.

Tokenizer still warns about the Mistral-Small regex. Outputs were English and correct on these prompts; a tokenizer-sensitive eval is still owed.

## What's in the repo

```
src/monkeyinference/
  kernels.py     Metal STREAM copy/read/write + qdot GEMV + 8-row MMA + five-trit GEMV
  trit.py        5-trits-per-byte pack/unpack (g128 padded to 26 bytes)
  stream_cpu.c   P-core STREAM read (QoS USER_INTERACTIVE) for CPU+GPU aggregate
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
~/.monkey/mlx-venv/bin/python scripts/probe_five_trit.py
~/.monkey/mlx-venv/bin/python scripts/probe_breakeven.py
```

Uses the existing `~/.monkey/mlx-venv` (mlx 0.32.0). No second venv. `--no-custom` forces the Prism MLX matmul.

## What's left

1. **`ternary_qmm_m8`.** Shipped. MMA is flat. Versus greedy qdot **2.25× GEMV / 2.68× e2e** (1.3× gate missed). `uint2b` does not compile. Greedy stays on qdot.
2. **Always-8 + `verify_gdn_commit`.** **Not building.** Leftover+k is already MMA-flat (273–282 ms). Replay=0 predicts **9.39 tok/s** at K=7 (need 3.26 accepts). Table is the evidence speculation is dead at current acceptance.
3. **DFlash Q4 MMA.** Draft is already **37–42 ms** isolated. Cutting to 15 ms is a 0.05 tok/s paper win at 3.00 accepts. Skip.
4. **Five-trit pack.** **HOLD.** Lossless, 0.88× qdot census-wide, predicted 9.00 tok/s. Keep 2-bit qdot. Kernel stays behind `enable_five_trit`.
5. If spec is reopened: **acceptance gap** 3.00 → 4.1+, not kernels. Token identity still gates any e2e claim.
6. **Then** post-Hadamard `|H(x)|` sparsity; TEAL-style qdot only if 30%+ droppable with token identity. Otherwise stop.
7. Prefill 40.9 vs llama-bench pp512 47 — not the chat bottleneck.

Do not: train a drafter, tree attention / EAGLE / Medusa, fuse the Hadamard, port PowerInfer / Deja Vu, rewrite in Swift, WY-chunked GDN, `uint4b` 2-bit padding, **hybrid CPU/GPU decode (ruled out)**, always-8 GDN commit, five-trit convert, Apple10 persistent-wave policy.

## Disk / cleanliness

Added: git repo under `~/projects/monkeyinference` (source). **+1.266 GB** at `~/.monkey/models/Qwen3.8-27B-Splash-draft/draft/` (`layer-0.bin`…`layer-4.bin` + `model.bin`). Did not download `target/`, `vision/`, or `incoai/Qwen3.8-27B-DFlash2`. Tokenizer compare used a 20 MB Hub file in `/tmp` and deleted it. Did not delete user data. Bench processes exited. No servers. Free disk ~16–18 GB.

## Rulings log

- 2026-09-18 **HOLD** bandwidth-bound decode. Greedy 10.22 tok/s is 80–90% of a read-like STREAM / 7.254 GB. Read-only is **not** 100–110 GB/s on this Air.
- 2026-09-18 **HOLD** speculation required to beat ~11 tok/s in general chat. PLD 11.70 is a copy-prompt number (2.8 accepts/pass), not a chat number. **UPDATE 2026-09-19:** DFlash+MMA at 3.00 accepts does not beat 10.22 (see break-even table).
- 2026-09-18 **UPDATE** custom qdot GEMV **is** production decode. `--no-custom` remains the Prism MLX path. Layer vs MLX max_abs 0.013; greedy tokens match.
- 2026-09-18 **UPDATE** Ashvin authorized Splash `draft/` bytes. Fetched **1.266 GB** only. Vocab/hidden match. Do **not** fetch the 17.4 GB package or the 3.85 GB BF16 DFlash2 repo.
- 2026-09-18 **HOLD** self-speculation with first N layers + shared `lm_head` plateaus at **1.07 accepts/pass** for N∈{2,4,6,8}.
- 2026-09-18 **NEW** Splash DFlash 2 on Bonsai explain is **3.00 accepts/pass**, token-identical to leftover greedy. Modelling lead **ruled in**. Training is not required to clear 2×.
- 2026-09-18 **NEW** GDN snapshot tax is gone (CoW pin). Weight-once verify is why PLD copy is 11.70 vs 8.19 leftover greedy, not 7.7 vs 8.4.
- 2026-09-18 **HOLD** leftover 2.77 vs greedy 9.74 was **Low Power Mode**, not the DFlash residual. Same-run leftover 8.47 vs greedy 10.22 with LPM off. Leftover never runs the draft stream.
- 2026-09-18 **HOLD** DFlash residual is float32 because `|h|≈5e4`. fp32 `o_proj` accum + fp16 store is finite and drops accepts to 0 (fp16 ULP 32). That is as narrow as the overflow/precision actually requires.
- 2026-09-18 **UPDATE** DFlash loop: incremental context KV, query pin, replay without `lm_head`, default K=2. Explain tok/s **7.32** (K=7 still 3.00 accepts/pass at 3.99 tok/s). Does not beat 10.22.
- 2026-09-19 **HOLD** GPU **read-only STREAM is the same band as copy** (85–94 vs 87–94 GB/s). Copy wall is ~2× read wall at 512 MiB — same instantaneous bus, 2× traffic. 1 GiB read median 85 GB/s. Decode resembles GPU read. Do not retarget the ceiling to 100–110 GB/s.
- 2026-09-19 **NEW** CPU+GPU disjoint read aggregates **103 GB/s** vs GPU-only 94 (+10%); GPU drops to 54 under contention. llama.cpp layer-split is existence proof. **Hybrid decode ruled out.**
- 2026-09-19 **HOLD** mlx_lm GDN prefill and verify are the **same sequential kernel**. Isolated GDN 12% of T=8. WY-chunked GDN remains out of scope.
- 2026-09-19 **UPDATE** T=8 at 4.8× T=1 was the **wrong kernel class** (`ternary_qmv_once` is a matvec). `ternary_qmm_m8` is the Splash-shaped MMA: **MMA T=8 / MMA T=1 = 1.00×**, **2.25× vs greedy qdot** on MLP-up / **2.68× e2e** (103 → 277 ms). `uint2b_format` does not compile.
- 2026-09-19 **NEW** DFlash break-even with measured draft and replay=0: propose is **37–42 ms flat in K**. No K beats 10.22. Best **9.39 tok/s** at K=7 (3.00 accepts; need **3.26**). Do not build always-8+GDN commit. If spec is reopened, chase acceptance 3.00→4.1, not kernels.
- 2026-09-18 **NEW** Bonsai affine-2bit codes are genuinely **ternary**: **0 / 3.54e9** code-3. Exact-zero (code 1) **32.8%**.
- 2026-09-19 **HOLD** five-trit unpack. Lossless pack, census-weighted **0.88×** qdot, predicted greedy **9.00 tok/s**. mlp_up 2.02×. Keep 2-bit qdot.
