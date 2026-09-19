"""Distill + holdout prompts for the bounded DFlash LoRA retry.

The first run overfit 64 sequences. This bank is thousands of disjoint
chat / explanation / code / long-context prompts. EXPLAIN_PROMPT is holdout.
"""

from __future__ import annotations

from monkeyinference.bench import EXPLAIN_PROMPT, FRANCE_PROMPT

# Success metric lives here. Never put these strings in the train bank.
HOLDOUT_PROMPTS: list[tuple[str, int]] = [
    (EXPLAIN_PROMPT, 48),
    (
        "Explain how a B-tree insert splits a full node, then give a 6-line Python sketch.",
        48,
    ),
    (
        "Write a polite Slack message asking to postpone a design review, then a firmer follow-up.",
        48,
    ),
    (
        "Debug this: `for i in range(len(xs)): xs.pop(i)`. What happens, and how do you fix it?",
        48,
    ),
    (
        "Why might unified memory make a 27B model slower than a discrete GPU with the same FLOPS?",
        48,
    ),
    (
        "Implement Dijkstra in Python using heapq, then state time complexity in one sentence.",
        48,
    ),
    (
        "Give a weekday lentil-soup recipe with timing, then a shopping list of eight items.",
        48,
    ),
    (
        "You are reviewing a PR that adds caching. List three questions you would ask the author.",
        48,
    ),
]

SEED_PROMPTS: list[tuple[str, int]] = [
    ("What is speculative decoding, and why does it help on a memory-bound GPU?", 48),
    ("Explain residual connections in a transformer in two short paragraphs.", 48),
    ("Give a practical checklist for reviewing a Python pull request.", 48),
    ("How does a Hadamard transform differ from a DFT? Keep it concrete.", 48),
    (
        "Write a Python function that merges two sorted lists without allocating extra lists beyond the output.",
        64,
    ),
    ("Explain gated delta networks as if I have used Mamba but not GDN.", 48),
    ("What should I pack for a three-day hiking trip in the Cascades in September?", 48),
    ("Debug this: a Metal kernel compiles but returns zeros. Where do you look first?", 48),
    ("Summarize copy-on-write vs memcpy for a recurrent state cache.", 48),
    ("Write a bash loop that prints the ten largest files under a directory.", 48),
    ("Why might 2-bit quantization change a language model's next-token distribution?", 48),
    ("Explain DRAM bandwidth vs compute roofline for a 27B GEMV.", 48),
    ("Draft a polite email asking a teammate to review a performance regression.", 48),
    ("Implement binary search in Python over a list of integers, with tests in comments.", 64),
    ("What is block diffusion, in the sense used by speculative draft models?", 48),
    ("How do I keep a MacBook Air cool while running a long GPU job?", 40),
    ("Explain token identity as a correctness gate for speculative decoding.", 48),
    ("Write a regex that matches IPv4 addresses and mention a false-positive.", 48),
    ("Compare LoRA and full fine-tuning when the base weights are 4-bit.", 48),
    ("What's a good way to structure a living design doc for a systems project?", 48),
    ("Explain why fp16 residuals can break a draft model when activations are ~1e4.", 48),
    ("Write a Python dataclass for a generate() result with tok/s fields.", 48),
    ("How does prompt-lookup decoding differ from a trained drafter?", 48),
    ("Give three reasons a Q4 GEMM can beat a 2-bit GEMV on small-M batches.", 48),
    ("Walk through git rebase vs merge for a long-lived feature branch.", 48),
    ("Write a SQL query that counts orders per day for the last 14 days.", 40),
    ("What is a threadgroup in Metal, and why does 256 threads matter for MMA?", 48),
    ("Explain the difference between accepts/pass and tokens per second.", 48),
    ("Implement FizzBuzz in Rust, then in Python, both under 20 lines.", 48),
    ("How would you teach a new hire to read a STREAM benchmark?", 48),
    ("Describe a chat-app feature that summarizes a long thread in one paragraph.", 48),
    ("Why freeze a quantized backbone and train adapters instead of dequantizing?", 48),
    ("Write a Makefile target that runs unit tests and a short bench.", 40),
    ("Explain KV cache pins versus rebuilding the cache after a rejected draft.", 48),
    ("What's the difference between greedy decoding and sampling at temp 1.0?", 48),
    ("Give a short architecture review of a 5-layer draft sitting under a 64-layer target.", 48),
    ("Write a Python generator that yields sliding windows of length 8.", 48),
    ("How do I interpret a front-loaded rejection histogram in speculative decoding?", 48),
    ("Explain unified memory on Apple silicon in one paragraph for a CUDA person.", 48),
    ("Draft a README section: hardware assumptions and honest performance numbers.", 48),
    ("What does it mean for a pack to be lossless-ternary but slower to unpack?", 48),
    ("Write a function that computes cross-entropy of logits vs integer targets in numpy.", 48),
    ("How should I choose K for a block-diffusion drafter that always runs 8 query rows?", 48),
    ("Explain why leftover-greedy is the right identity baseline for speculation.", 48),
    ("A user asks: is my laptop slow or is the model just big? Answer carefully.", 48),
    ("Write a pytest for a function that packs 5 ternary codes per byte.", 48),
    ("What is an occupancy artifact in a GEMV microbench?", 48),
    ("Describe a good on-call handoff note after a failed training run.", 40),
    ("Compare attention and GDN layers in Qwen3.5-style models at a high level.", 48),
    ("Write a Python CLI with argparse for collect/train/eval subcommands.", 48),
    ("Why might a draft be right on token 1 and collapse after?", 48),
    ("Explain cosine LR with warmup without formulas first, then give the formula.", 48),
    ("How do you keep secrets out of a git repo that has a results/ folder?", 40),
    ("Implement a ring buffer of token ids in Python with a fixed capacity.", 48),
    ("What should a performance claim include besides tok/s?", 48),
    (
        "Give an example of a chat turn that asks for a refactor, then one that asks for an explanation.",
        48,
    ),
    ("Why is AdamW state so large compared to the trainable parameter count?", 48),
    ("Write a short comment-only walkthrough of rms_norm in float32.", 40),
    ("How would you estimate wall-clock for 5000 GPU training steps from one timed step?", 40),
    ("Explain the difference between a seed main branch and a feature branch on GitHub.", 40),
    ("A teammate inverted a speedup ratio. Write the correction without being snide.", 40),
    (
        "What is the smallest experiment that would tell you a drafter adapted to a quantized target?",
        48,
    ),
    ("Write a JSON schema for a break-even table row (k, draft_ms, pred_tok_s).", 40),
    ("Explain why you would not delete someone else's GGUF to free training disk.", 40),
]

TOPICS = [
    "speculative decoding",
    "quantization-aware training",
    "KV cache paging",
    "FlashAttention",
    "mixture of experts routing",
    "rotary position embeddings",
    "grouped-query attention",
    "speculative sampling",
    "block diffusion",
    "LoRA rank selection",
    "AdamW vs SGD",
    "gradient clipping",
    "mixed precision",
    "activation checkpointing",
    "tensor parallelism",
    "pipeline parallelism",
    "sequence packing",
    "byte-pair encoding",
    "sentencepiece",
    "RMSNorm",
    "LayerNorm",
    "SwiGLU",
    "Gated DeltaNet",
    "Mamba state spaces",
    "Hadamard transform",
    "Walsh functions",
    "roofline models",
    "DRAM tRFC",
    "GPU occupancy",
    "Metal threadgroups",
    "CUDA warps",
    "simdgroup MMA",
    "tile-based deferred rendering",
    "unified memory",
    "copy-on-write",
    "reference counting",
    "arena allocators",
    "lock-free queues",
    "seqlocks",
    "RCU",
    "Bloom filters",
    "cuckoo hashing",
    "LSM trees",
    "B+ trees",
    "CRDTs",
    "Raft elections",
    "Paxos acceptors",
    "exactly-once delivery",
    "idempotent APIs",
    "backpressure",
    "token buckets",
    "circuit breakers",
    "structured logging",
    "distributed tracing",
    "p99 latency",
    "head-of-line blocking",
    "Nagle's algorithm",
    "QUIC vs TCP",
    "TLS 1.3 handshake",
    "AEAD ciphers",
    "constant-time compare",
    "timing side channels",
    "ASLR",
    "stack canaries",
    "use-after-free",
    "data races",
    "undefined behavior in C",
    "move semantics",
    "RAII",
    "Python GIL",
    "asyncio vs threads",
    "vectorization",
    "cache-oblivious algorithms",
    "prefetching",
    "false sharing",
    "NUMA locality",
    "page cache",
    "fallocate",
    "mmap vs read",
    "write-ahead logs",
    "snapshot isolation",
    "MVCC",
    "vacuum in Postgres",
    "join order",
    "covering indexes",
    "cardinality estimates",
    "hyperloglog",
    "t-digest",
    "reservoir sampling",
    "A/B test power",
    "calibration plots",
    "temperature scaling",
    "label smoothing",
    "dropout at inference",
    "weight tying",
    "tied embeddings",
    "vocab mismatch",
    "tokenizer fertility",
    "byte fallback",
    "chat templates",
    "tool calling",
    "JSON mode",
    "constrained decoding",
    "grammar-guided generation",
    "rejection sampling",
    "typical sampling",
    "min-p sampling",
    "repetition penalty",
    "prompt injection",
    "jailbreak patterns",
    "eval contamination",
    "train/test leakage",
    "held-out acceptance",
    "early stopping",
    "weight decay",
    "learning-rate warmup",
    "cosine decay",
    "EMA of weights",
    "gradient accumulation",
    "microbatching",
    "loss spikes",
    "exploding norms",
    "NaN in softmax",
    "logsumexp",
    "cross-entropy",
    "KL distillation",
    "reverse KL",
    "teacher forcing",
    "scheduled sampling",
    "exposure bias",
    "draft/verify mismatch",
    "front-loaded rejects",
    "token identity",
    "leftover greedy",
    "always-8 MMA",
    "qdot GEMV",
    "affine 2-bit packs",
    "group size 128",
    "scale/bias quant",
    "Hadamard block 1024",
    "STREAM bandwidth",
    "Low Power Mode clocks",
    "thermal throttling",
    "MacBook Air M4",
    "24 GB unified memory",
    "llama.cpp Metal",
    "mlx vs pytorch",
    "safetensors mmap",
    "git worktrees",
    "rebase vs merge",
    "code review tone",
    "incident retrospectives",
    "SLOs vs SLAs",
    "error budgets",
    "feature flags",
    "dark launches",
    "canary deploys",
    "blue/green",
    "schema migrations",
    "expand/contract",
    "backfills",
    "idempotent jobs",
    "dead letter queues",
    "poison pills",
    "retry storms",
    "jittered backoff",
    "hedged requests",
    "tail latency",
    "priority inversion",
    "work stealing",
    "actor model",
    "CSP channels",
    "async cancellation",
    "structured concurrency",
    "effect systems",
    "algebraic data types",
    "GADTs",
    "type erasure",
    "monomorphization",
    "inlining",
    "PGO",
    "LTO",
    "debug vs release",
    "UBSan",
    "ASan",
    "tsan",
    "fuzzing",
    "property tests",
    "golden files",
    "snapshot tests",
    "contract tests",
    "consumer-driven contracts",
    "OpenAPI",
    "protobuf vs JSON",
    "cap'n proto",
    "flatbuffers",
    "zero-copy parse",
    "SIMD JSON",
    "arena parsing",
    "interned strings",
    "small-string opt",
    "SSO pitfalls",
    "UTF-8 validation",
    "NFC vs NFD",
    "grapheme clusters",
    "BIDI text",
    "timezone databases",
    "leap seconds",
    "monotonic clocks",
    "NTP slew",
    "HLC timestamps",
    "vector clocks",
    "dotted versions",
    "CRDT maps",
    "LWW registers",
    "OR-sets",
    "tombstones",
    "compaction",
    "bloom on SST",
    "partitioning keys",
    "hot shards",
    "scatter-gather",
    "fanout",
    "in-process caches",
    "Redis eviction",
    "memcached slabs",
    "cache stampede",
    "singleflight",
    "request coalescing",
    "stale-while-revalidate",
    "ETags",
    "conditional GET",
    "Range requests",
    "multipart upload",
    "content hashing",
    "merkledags",
    "CAS loops",
    "compare-and-swap",
    "LL/SC",
    "memory orders",
    "acquire-release",
    "seq_cst tax",
    "false wakeups",
    "condition variables",
    "futexes",
    "io_uring",
    "kqueue",
    "epoll",
    "Grand Central Dispatch",
    "quality of service classes",
    "priority aging",
    "interactive vs batch",
    "energy-aware scheduling",
    "big.LITTLE",
    "P-cores vs E-cores",
    "GPU preemption",
    "command buffer reuse",
    "argument buffers",
    "heap resources",
    "hazard tracking",
    "residency sets",
    "MTLIO",
    "shader compilation cache",
    "pipeline descriptors",
    "indirect command buffers",
    "mesh shaders",
    "workgraphs",
    "persistent threads",
    "occupancy vs ILP",
    "register pressure",
    "shared memory banks",
    "conflict-free layouts",
    "swizzling",
    "Morton codes",
    "Z-order curves",
    "tiling for GEMM",
    "wtile vs ntile",
    "split-K",
    "stream-k",
    "persistent GEMM",
    "epilogue fusion",
    "bias-as-minus-scale",
    "absmax vs percentile quant",
    "NF4",
    "GGUF Q4_K",
    "PQ2_0",
    "HQQ",
    "AWQ",
    "GPTQ",
    "SmoothQuant",
    "rotation quant",
    "QuaRot",
    "SpinQuant",
    "incoherence processing",
    "outlier channels",
    "per-token vs per-tensor",
    "dynamic quant",
    "static cal",
    "KL to teacher",
    "hidden-state MSE",
    "argmax matching",
    "top-k overlap",
    "draft length K",
    "tree attention",
    "EAGLE",
    "Medusa heads",
    "lookahead decoding",
    "prompt lookup n-grams",
    "self-speculation",
    "early-exit layers",
    "shared lm_head",
    "vocab alignment",
    "multilingual tok/s",
    "code vs chat accepts",
    "long-context drift",
    "lost in the middle",
    "needle-in-haystack",
    "prefix caching",
    "prompt cache hits",
    "chunked prefill",
    "continuous batching",
    "iteration-level sched",
    "preempt decode",
    "fairness vs throughput",
    "goodput",
    "SLA tokens",
    "time to first token",
    "inter-token latency",
    "jitter",
    "warmup kernels",
    "first-batch tax",
    "autotune caches",
    "shape specialization",
    "dynamic shapes",
    "graph capture",
    "MPS graphs",
    "MLX compile",
    "mx.eval vs lazy",
    "synchronize costs",
    "unified vs discrete copies",
    "PCIe vs fabric",
    "NVLink myths on a laptop",
    "battery vs plugged",
    "clamshell mode",
    "Amphetamine vs caffeinate",
    "tmux detach",
    "SIGHUP trainers",
    "checkpoint every N",
    "adapter safetensors",
    "rank-8 vs rank-16",
    "zero-init B",
    "overfitting 64 prompts",
    "held-out explain",
    "break-even 3.70",
]

EXPLAIN_TEMPLATES = [
    "Explain {topic} in two short paragraphs for an engineer who has not seen it.",
    "What is {topic}? Give a concrete example and one failure mode.",
    "Compare {topic} to the obvious alternative. When would you pick each?",
    "Teach {topic} as if I already know the neighboring idea but not this one.",
    "Give a practical checklist for applying {topic} in a production system.",
    "Why does {topic} show up in performance work? Keep it specific.",
]

CODE_TEMPLATES = [
    "Write a small Python function related to {topic}. Include a comment on complexity.",
    "Sketch a unit test that would catch a bug in a {topic} implementation.",
    "Show a before/after refactor that makes {topic} easier to reason about.",
    "Write a bash one-liner or short script that helps debug {topic}.",
]

CHAT_TEMPLATES = [
    "A teammate is confused about {topic}. Reply in chat, no jargon dump.",
    "Draft an email asking for a review of a change that touches {topic}.",
    "A user is frustrated that {topic} seems slow. Answer without being defensive.",
    "Write a standup note: yesterday I worked on {topic}, today I will...",
]


def _long_context(topic: str, n: int) -> str:
    other = TOPICS[(n * 7) % len(TOPICS)]
    return (
        f"You are reading an internal design note about {topic}. "
        f"Doc id D-{n:04d}. Audience: on-call engineers.\n\n"
        f"Background: token identity must hold even if throughput drops. "
        f"A 298 ms 8-row verify against a 93 ms greedy step is "
        f"{298 / 93:.2f} greedy-token-equivalents of tax before the draft runs. "
        f"Related idea: {other}.\n\n"
        f"We already saw a 64-prompt LoRA memorize continuations and move "
        f"held-out accepts from 3.00 to 2.09. Do not repeat that. "
        f"Keep the prior close to the stock Q4 draft.\n\n"
        f"Summarize this note on {topic} in one tight paragraph, then list "
        f"two risks a reviewer should push on."
    )


def _holdout_set() -> set[str]:
    s = {p for p, _ in HOLDOUT_PROMPTS}
    s.add(FRANCE_PROMPT)
    s.add(EXPLAIN_PROMPT)
    return s


def distill_prompts(*, n: int = 1200) -> list[tuple[str, int]]:
    """Unique train prompts, never overlapping the holdout set.

    The first `n` are interleaved across chat / explanation / code /
    long-context so a 1200-cap is not 1200 copies of one template.
    """
    banned = _holdout_set()
    out: list[tuple[str, int]] = []
    seen: set[str] = set()

    def add(text: str, max_new: int) -> bool:
        text = text.strip()
        if text in banned or text in seen:
            return False
        seen.add(text)
        out.append((text, max_new))
        return True

    for p, k in SEED_PROMPTS:
        add(p, k)

    for i, topic in enumerate(TOPICS):
        add(EXPLAIN_TEMPLATES[i % len(EXPLAIN_TEMPLATES)].format(topic=topic), 48)
        add(CODE_TEMPLATES[i % len(CODE_TEMPLATES)].format(topic=topic), 48)
        add(CHAT_TEMPLATES[i % len(CHAT_TEMPLATES)].format(topic=topic), 40)
        if i % 4 == 0:
            add(_long_context(topic, i), 48)
        if len(out) >= n:
            return out[:n]

    for topic in TOPICS:
        for tmpl in EXPLAIN_TEMPLATES + CODE_TEMPLATES + CHAT_TEMPLATES:
            add(tmpl.format(topic=topic), 48)
            if len(out) >= n:
                return out[:n]

    for i in range(800):
        topic = TOPICS[i % len(TOPICS)]
        add(
            f"Variant {i}: explain {topic} using a cooking analogy, then drop the analogy "
            "and give the real definition in two sentences.",
            48,
        )
        add(
            f"Code kata {i}: write a Python snippet that would be wrong if {topic} "
            "were ignored. Keep it under 25 lines.",
            64,
        )
        if len(out) >= n:
            return out[:n]

    if len(out) < n:
        raise RuntimeError(f"prompt bank too small: {len(out)} < {n}")
    return out[:n]


def assert_holdout_disjoint(train: list[tuple[str, int]]) -> None:
    banned = _holdout_set()
    overlap = [p for p, _ in train if p in banned]
    if overlap:
        raise RuntimeError(f"holdout leaked into train: {overlap[:3]}")
