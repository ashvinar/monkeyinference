# Ternary engine for Bonsai 2 27B on this M4 Air

Living design-and-results doc. Engine repo: [github.com/ashvinar/monkeyinference](https://github.com/ashvinar/monkeyinference) · branch `cursor/ternary-metal-engine-09ea`. Standalone Metal/MLX runtime, not a Splash fork.

**Headline (2026-09-18):** Decode at 27B / 2.25 bpw (MLX affine) **is** DRAM-bandwidth bound on this machine. Measured STREAM copy is **86.6 GB/s** of the published **120 GB/s**. That puts a no-speculation ceiling at **~11.3 tok/s** (measured) / **15.6 tok/s** (published). Prism MLX started at **~8 tok/s**; monkeyinference greedy is **8.38 tok/s decode, 41.6 tok/s prefill**, coherent (`Paris`, speculative-decoding explanation). A custom ternary GEMV compiles and matches dequant packing exactly, but does **not** beat Apple's `affine_qmv_fast` on this 10-core GPU, so production decode still uses MLX quantized matmul plus the Prism Hadamard. Prompt-lookup speculative decode is wired and accepted 10/10 draft tokens on a copy prompt; GDN state snapshot cost ate the speedup. Beating ~11 tok/s still requires a cheap draft + verify, not a faster vibe.

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
| Free disk when built | ~17–18 GB. **No extra weights downloaded.** |

## Roofline

Language tensors in the MLX pack: **7.674 GB** (6.822 GB 2-bit codes + 0.420 GB scales + 0.420 GB redundant biases + 0.012 GB Hadamard signs). Vision tower 0.921 GB is skipped.

Each greedy decode token streams the language weights once (GEMV), plus a negligible activation/KV term at short context.

| Assumption | tok/s = GB/s ÷ 7.674 GB |
| --- | ---: |
| Published 120 GB/s, with biases | **15.6** |
| Published 120 GB/s, drop biases | 16.5 |
| Measured STREAM 86.6 GB/s, with biases | **11.3** |
| Measured STREAM 86.6 GB/s, drop biases | 11.9 |
| Starting Prism MLX decode | **8.0** |
| 8.0 / 15.6 published | **51%** |
| 8.0 / 11.3 measured | **71%** |

Profiling **does not contradict** the two rulings:

1. **Decode is memory-bandwidth bound.** 8 tok/s is already most of what this Air can stream. A tuned ternary kernel can recover the remaining ~3 tok/s up to the STREAM ceiling, not a 2–3×. MLX `affine_qmv` on an MLP-up-shaped matrix hits ~25–36 GB/s in microbenches (well under STREAM 87 GB/s) because GEMV reuse of `x` is poor and the 10-core GPU cannot keep DRAM busy on batch-1. That is occupancy / launch / inner-loop, not “wrong algorithm class.”
2. **Beating the ceiling requires speculative decode.** Splash’s 74 tok/s on an M5 Pro is DFlash 2 (several accepted tokens per target weight pass) on a wider memory bus, not a 4-bit kernel miracle. Same math here: 2× accepted tokens per pass → ~16–22 tok/s; 4× → ~30–45 tok/s, still under Splash’s published M5 Pro number.

Prism’s own laptop table (PQ2_0 GGUF, no speculation): M4 Pro 18 tok/s, M5 Pro 28 tok/s streaming ~204 GB/s. This Air landing at 8 tok/s on 87 GB/s STREAM is in family.

## Architecture decisions

| Decision | Why | Override |
| --- | --- | --- |
| Standalone Python+MLX repo, not a Splash fork | Splash factory has no AR fallback and only loads `splash-packed-q4` + DFlash. Bonsai is affine-2bit Hadamard safetensors. | Rewrite in Swift/Metal if Python dispatch shows up in traces after GEMV is STREAM-bound. |
| Text-only load | Vision tower is 0.92 GB FP16 and unused for these prompts. Peak Metal 8.25 GB vs Prism VL 9.3–12.1 GB. | Keep the tower behind an explicit `--vision` flag later. |
| Keep Prism activation Hadamard (block 1024, `H(x⊙s)/√1024`) | Weights are stored in that basis. Skipping it is silent garbage. | None. |
| Default matmul = `mx.quantized_matmul` (Prism path) | Custom ternary GEMV is packing-correct (`mx.dequantize` vs our unpack is exact) but **slower or tied** vs `affine_qmv_fast` on this GPU (MLP-up ~0.3–1.2× depending on thermal; attn-q consistently slower). Shipping a slower kernel would fail the speed claim. | `--custom` flag. Next kernel pass should clone MLX’s 64-thread qmv_fast qdot (bit-mask, no per-lane shifts) and fuse Hadamard. |
| Do not store/load bias in the custom kernel | Prism sets `bias = -scale`. 0.420 GB of DRAM traffic for no information. | Only matters once the kernel beats MLX. |
| Speculative decode is first-class, with GDN snapshot/restore | 48/64 layers are gated-delta (`ArraysCache`), which is **not trimmable**. mlx_lm’s `speculative_generate_step` would refuse this model. Snapshot conv + SSM state, restore on partial reject, replay accepted prefix. | If Apple adds trimmable GDN cache, switch. |
| First draft = prompt-lookup n-gram (PLD), zero extra weights | Disk is tight (~18 GB free). Qwen3-0.6B has vocab 151936 vs Bonsai 248320 — cannot draft. A trained 5-layer DFlash is the Splash move and needs data + space. | Ask before downloading any draft. |
| Correctness gates in the bench loop | Kernel packing vs `dequantize`; greedy `Paris` + speculative-decoding explanation vs the prior Prism MLX strings. | Token-level logit KL vs Prism reference if a claim depends on 0.1 tok/s. |

## Benchmark progression

Greedy, thinking off, temp 0. Same prompts as the Splash-investigation table.

| Engine | Prompt tok | Gen tok | Prefill tok/s | Decode tok/s | Peak | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prism MLX (prior) explain | 32 | 59 | 30.0 | **8.16** | 9.26 GB | coherent spec-decode explanation |
| Prism MLX (prior) long | 717 | 60 | 40.2 | 7.81 | 12.09 GB | coherent |
| llama.cpp Metal PQ2_0 tg128 | — | 128 | — | 6.71 ± 1.40 | — | forced tokens |
| **monkeyinference greedy explain** | 32 | 59 | 33.1 | **8.38** | **8.25 GB** | same idea, coherent |
| **monkeyinference greedy long** | 717 | 60 | **41.6** | 8.22 | 10.42 GB | same two-sentence explanation as Prism |
| **monkeyinference PLD copy prompt** | 46 | 14 | 32.0 | 7.72 | 8.33 GB | exact copy; **10/10 draft accepted** |

Starting point for this project: **8 tok/s decode, ~40 tok/s prefill**. Current greedy: **8.38 / 41.6**. That is not a kernel win; it is “same Prism compute, no vision tower, mlx_lm generate.” The 0.2 tok/s bump is noise / thermal / instrumentation.

2-token `Paris` rows still read ~16 tok/s on both engines (kernel launch + EOS). Ignore them.

### Microbench (no 27B load)

STREAM 86.6 GB/s. GEMV (read weights+scales, 20–30 iters; Air throttles, treat as ±30%):

| Shape | Custom ternary GB/s | MLX affine GB/s |
| --- | ---: | ---: |
| MLP up 17408×5120 | 9–29 | 25–36 |
| MLP down 5120×17408 | 20–26 | 22–32 |
| attn q 6144×5120 | 6–13 | 13–21 |
| lm_head 248320×5120 | 56 | 71 |

Custom path is not the production decoder until it is above the MLX column on a quiet machine.

## Correctness

- **Packing:** synthetic ternary codes packed as MLX uint32×16, `mx.dequantize` vs numpy unpack **max abs 0**.
- **Hadamard:** `fwht` matches `mx.hadamard_transform` on signed 5120-wide activations (err < 1e-3). Inverse∘forward is an involution within fp16.
- **GEMV vs numpy:** max abs ~0.10 on 17408×5120 (fp16 inputs, f32 accum). vs MLX affine qdot ~0.25 — different inner product, not a code mismatch.
- **Coherence (wired in `monkeyinference bench`):**
  - “Name the capital of France…” → `Paris`
  - “Explain speculative decoding in two short sentences.” → draft/verify explanation, English, not garbage
  - Long-prefill variant produced the same two-sentence idea as the prior Prism MLX JSON
  - PLD copy prompt reproduced the sentence (plus EOS, now stripped)

A fast engine emitting plausible-looking noise would have failed this loop. It did not.

Tokenizer still warns about the Mistral-Small regex. Outputs were English and correct on these prompts; a tokenizer-sensitive eval is still owed.

## What’s in the repo

```
src/monkeyinference/
  kernels.py     Metal STREAM copy + ternary GEMV/QMM (mx.fast.metal_kernel)
  hadamard.py    Prism fwht
  packed.py      PackedLinear / PackedEmbedding
  load.py        text-only safetensors → mlx_lm Qwen3.5 TextModel
  generate.py    greedy (mlx_lm.stream_generate) + PLD spec loop
  spec.py        n-gram drafter + GDN/KV snapshot
  roofline.py    STREAM + GEMV + the 7.674 GB math
  bench.py       coherence + speed vs the 8 tok/s starting point
```

Weights stay at `~/.monkey/models/Ternary-Bonsai-2-27B-mlx-2bit/`. The repo is source only.

```bash
export PYTHONPATH=src
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli profile
~/.monkey/mlx-venv/bin/python tests/test_kernels.py
~/.monkey/mlx-venv/bin/python -m monkeyinference.cli bench
```

Uses the existing `~/.monkey/mlx-venv` (mlx 0.32.0). No second venv.

## What’s left

1. **GEMV that actually beats `affine_qmv_fast`.** Clone MLX’s qdot (pre-shifted `x`, mask-and-accumulate), 64-thread TGs, fuse the 1024 Hadamard into the x-load. Goal: close 8.4 → ~11 tok/s greedy. If traces then show Python dispatch, move the layer loop to compiled Metal.
2. **Spec decode that is faster than greedy.** PLD already accepts on copy/code. The 10/10 run was *slower* (7.7 tok/s) because each proposal snapshots ~48 GDN states. Need in-place dual buffers or copy-on-write, and a verify GEMM that streams weights once for K draft tokens.
3. **A real draft with the 248320 vocab.** Not downloaded. Disk ~18 GB free; a 1 GB 4-bit Qwen3.8-tiny does not exist in the obvious catalog, and Qwen3-0.6B’s tokenizer does not match. Ask before spending space on training or a Hub draft.
4. **Prefill.** 41.6 tok/s vs llama-bench pp512 47. Still the compute-bound side; not the first bottleneck for chat.
5. Tokenizer regex; sampling; optional vision.

## Disk / cleanliness

Added: git repo under `~/projects/monkeyinference` (source, kilobytes). No new model files. Did not delete user data. Bench processes exited. No servers.

## Rulings log

- 2026-09-18 **HOLD** bandwidth-bound decode; 8 tok/s ≈ 71% of measured STREAM roofline.
- 2026-09-18 **HOLD** speculation required to beat ~11 tok/s.
- 2026-09-18 **HOLD** custom kernel not yet production; MLX affine + Hadamard is the decode path.
- 2026-09-18 **HOLD** do not download a draft until Ashvin okays the bytes.
