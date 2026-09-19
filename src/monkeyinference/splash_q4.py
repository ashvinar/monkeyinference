"""Splash-packed Q4 (`MDFD0004`) → MLX affine-4bit.

Splash stores each projection as one 16 KiB-aligned section:

    packed = nibble_weights || bfloat16_scales || bfloat16_biases

Nibbles are tiled StorageN=256. Dequant is the affine map Splash's kernels
use: `W[n, k] = scale[n, g] * q[n, k] + bias[n, g]` with group 64. That is
the same affine MLX `mx.dequantize(..., bits=4, group_size=64)` implements,
so after scatter-unpack we repack 8 consecutive K-nibbles per uint32 (low
nibble first) and call `mx.quantized_matmul`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MAGIC = b"MDFD0004"
ALIGN = 16 * 1024
GROUP = 64
STORAGE_N = 256
HEADER = 16

HIDDEN = 5120
VOCAB = 248320
DYNAMIC = 1280
QKV = 6144
ATTENTION = 4096
INTERMEDIATE = 17408
HEAD_DIM = 128
TARGET_HIDDEN = 25600
SELECTOR_RANK = 256
LAYERS = 5
KV_HEADS = 8
N_HEADS = 32
MASK_TOKEN_ID = 248070
BLOCK = 8
CONV_TAPS = 2
CONV_GROUP = 16
SELECTOR_TOP_K = 16
PROPOSAL_TOKENS = 7


def q4_packed_bytes(output_size: int, input_size: int) -> int:
    elements = output_size * input_size
    if input_size % GROUP:
        raise ValueError(f"Q4 input {input_size} is not aligned to {GROUP}")
    if output_size % STORAGE_N:
        raise ValueError(f"Q4 output {output_size} is not aligned to {STORAGE_N}")
    return elements // 16 * 9


def _align(offset: int) -> int:
    return (offset + ALIGN - 1) & ~(ALIGN - 1)


def bf16_to_fp16(u16: np.ndarray) -> np.ndarray:
    """IEEE bfloat16 bit patterns → float16 via float32."""
    bits = u16.astype(np.uint32, copy=False) << 16
    return bits.view(np.float32).astype(np.float16, copy=False)


def pack_mlx_q4(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray):
    """Pack row-major uint4 codes into MLX affine-4bit uint32 weights."""
    n, k = codes.shape
    if k % 8:
        raise ValueError(f"K={k} is not divisible by 8")
    grouped = codes.reshape(n, k // 8, 8).astype(np.uint32, copy=False)
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    packed = np.bitwise_or.reduce(grouped << shifts, axis=-1)
    return (
        mx.array(packed),
        mx.array(np.ascontiguousarray(scales)),
        mx.array(np.ascontiguousarray(biases)),
    )


def unpack_splash_q4_codes(weight_bytes: bytes | memoryview, n: int, k: int) -> np.ndarray:
    """Scatter Splash StorageN tiles into a dense [N, K] uint8 code matrix."""
    groups = k // GROUP
    tiles = (n + STORAGE_N - 1) // STORAGE_N
    raw = np.frombuffer(weight_bytes, dtype=np.uint8, count=n * k // 2)
    expected = tiles * groups * STORAGE_N * (GROUP // 2)
    if raw.size < expected:
        raise ValueError(f"Q4 weight section is truncated: {raw.size} < {expected}")
    codes = np.empty((n, k), dtype=np.uint8)
    tile_stride = groups * STORAGE_N * (GROUP // 2)
    for tile in range(tiles):
        n0 = tile * STORAGE_N
        rows = min(STORAGE_N, n - n0)
        blk = raw[tile * tile_stride : (tile + 1) * tile_stride]
        blk = np.asarray(blk).reshape(groups, STORAGE_N, GROUP // 2)[:, :rows, :]
        lo = blk & np.uint8(0x0F)
        hi = blk >> np.uint8(4)
        nib = np.empty((groups, rows, GROUP), dtype=np.uint8)
        nib[..., 0::2] = lo
        nib[..., 1::2] = hi
        codes[n0 : n0 + rows] = np.transpose(nib, (1, 0, 2)).reshape(rows, k)
    return codes


def unpack_splash_q4_params(param_bytes: bytes | memoryview, n: int, k: int) -> np.ndarray:
    """Scales/biases: bf16 [tiles, groups, 256] → float16 [N, groups]."""
    groups = k // GROUP
    tiles = (n + STORAGE_N - 1) // STORAGE_N
    u16 = np.frombuffer(param_bytes, dtype="<u2", count=tiles * groups * STORAGE_N)
    tiled = np.asarray(u16).reshape(tiles, groups, STORAGE_N)
    out = np.empty((n, groups), dtype=np.uint16)
    for tile in range(tiles):
        n0 = tile * STORAGE_N
        rows = min(STORAGE_N, n - n0)
        out[n0 : n0 + rows] = tiled[tile, :, :rows].T
    return bf16_to_fp16(out)


def splash_q4_to_mlx(section: bytes | memoryview, n: int, k: int):
    elements = n * k
    weight_bytes = elements // 2
    param_bytes = elements // 32
    packed_len = q4_packed_bytes(n, k)
    if len(section) < packed_len:
        raise ValueError(f"Q4 section {len(section)} < {packed_len}")
    codes = unpack_splash_q4_codes(section[:weight_bytes], n, k)
    scales = unpack_splash_q4_params(section[weight_bytes : weight_bytes + param_bytes], n, k)
    biases = unpack_splash_q4_params(
        section[weight_bytes + param_bytes : weight_bytes + 2 * param_bytes], n, k
    )
    return pack_mlx_q4(codes, scales, biases)


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.frombuffer(np.ascontiguousarray(values, dtype=np.float32).tobytes(), dtype=np.uint32)
    return ((bits >> 16) & 0xFFFF).astype(np.uint16)


def pack_splash_q4(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> bytes:
    """Reference packer matching Splash `packSlab` (used by tests)."""
    n, k = codes.shape
    groups = k // GROUP
    elements = n * k
    tiles = (n + STORAGE_N - 1) // STORAGE_N
    raw = np.zeros(tiles * groups * STORAGE_N * (GROUP // 2), dtype=np.uint8)
    scale_u16 = np.zeros(tiles * groups * STORAGE_N, dtype=np.uint16)
    bias_u16 = np.zeros_like(scale_u16)

    def parameter(row: int, g: int) -> int:
        return ((row // STORAGE_N) * groups + g) * STORAGE_N + (row % STORAGE_N)

    for row in range(n):
        for col in range(k):
            nibble = parameter(row, col // GROUP) * GROUP + (col % GROUP)
            byte_i = nibble // 2
            val = int(codes[row, col]) & 15
            if nibble & 1:
                raw[byte_i] = np.uint8((int(raw[byte_i]) & 0x0F) | (val << 4))
            else:
                raw[byte_i] = np.uint8((int(raw[byte_i]) & 0xF0) | val)
    scale_u16_2d = scale_u16
    bias_u16_2d = bias_u16
    s_bits = _f32_to_bf16_bits(np.ascontiguousarray(scales, dtype=np.float32))
    b_bits = _f32_to_bf16_bits(np.ascontiguousarray(biases, dtype=np.float32))
    for row in range(n):
        for g in range(groups):
            p = parameter(row, g)
            scale_u16_2d[p] = s_bits[row * groups + g]
            bias_u16_2d[p] = b_bits[row * groups + g]
    packed = q4_packed_bytes(n, k)
    out = bytearray(packed)
    wlen = elements // 2
    plen = elements // 32
    n_params = n * groups
    out[:wlen] = raw[:wlen].tobytes()
    out[wlen : wlen + plen] = np.ascontiguousarray(scale_u16[:n_params]).tobytes()
    out[wlen + plen : wlen + 2 * plen] = np.ascontiguousarray(bias_u16[:n_params]).tobytes()
    return bytes(out)


class PackedFile:
    def __init__(self, path: Path, magic: bytes, layer: int, kind: int):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        if len(self.data) % ALIGN:
            raise ValueError(f"{self.path} is not {ALIGN}-aligned ({len(self.data)})")
        if self.data[:8] != magic:
            raise ValueError(f"{self.path} magic {self.data[:8]!r} != {magic!r}")
        got_layer = int.from_bytes(self.data[8:12], "little")
        got_kind = int.from_bytes(self.data[12:16], "little")
        if got_layer != layer or got_kind != kind:
            raise ValueError(
                f"{self.path} header layer/type {got_layer}/{got_kind} != {layer}/{kind}"
            )
        self.offset = HEADER

    def section(self, nbytes: int, label: str) -> memoryview:
        start = _align(self.offset)
        end = start + nbytes
        if end > len(self.data):
            raise ValueError(f"{self.path} truncated at {label}: {end} > {len(self.data)}")
        self.offset = end
        return memoryview(self.data)[start:end]

    def finish(self) -> None:
        if _align(self.offset) != len(self.data):
            raise ValueError(
                f"{self.path} has unconsumed bytes: offset={self.offset} size={len(self.data)}"
            )


class Q4Linear(nn.Module):
    """Splash Q4 projection as MLX affine-4bit `quantized_matmul`."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight.shape[1]) * 8

    def __call__(self, x: mx.array) -> mx.array:
        orig = x.shape
        flat = x.reshape(-1, orig[-1])
        y = mx.quantized_matmul(
            flat,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=GROUP,
            bits=4,
        )
        return y.reshape(*orig[:-1], self.out_features)


def _q4_linear(section: memoryview, n: int, k: int) -> Q4Linear:
    w, s, b = splash_q4_to_mlx(section, n, k)
    return Q4Linear(w, s, b)


def _norm_vector(section: memoryview, n: int) -> mx.array:
    u16 = np.frombuffer(section, dtype="<u2", count=n)
    return mx.array(bf16_to_fp16(np.asarray(u16)))


def _bf16_matrix(section: memoryview, rows: int, cols: int) -> mx.array:
    u16 = np.frombuffer(section, dtype="<u2", count=rows * cols)
    return mx.array(bf16_to_fp16(np.asarray(u16).reshape(rows, cols)))


@dataclass
class DraftLayer:
    input_norm: mx.array
    attn_conv: mx.array  # [2, taps, hidden]
    attn_dynamic: Q4Linear
    qkv: Q4Linear
    q_norm: mx.array
    k_norm: mx.array
    o_proj: Q4Linear
    post_attn_norm: mx.array
    mlp_conv: mx.array
    mlp_dynamic: Q4Linear
    gate: Q4Linear
    up: Q4Linear
    down: Q4Linear


@dataclass
class DraftWeights:
    layers: list[DraftLayer]
    context_proj: Q4Linear
    hidden_norm: mx.array
    final_norm: mx.array
    selector: Q4Linear
    predecessor: mx.array
    successor: mx.array
    directory: Path
    bytes_on_disk: int


def _load_layer(path: Path, index: int) -> DraftLayer:
    f = PackedFile(path, MAGIC, index, 0)
    input_norm = _norm_vector(f.section(HIDDEN * 2, "input-norm"), HIDDEN)
    attn_conv = _bf16_matrix(f.section(4 * HIDDEN * 2, "attention-convolution"), 2 * CONV_TAPS, HIDDEN)
    attn_dynamic = _q4_linear(f.section(q4_packed_bytes(DYNAMIC, HIDDEN), "attention-dynamic"), DYNAMIC, HIDDEN)
    qkv = _q4_linear(f.section(q4_packed_bytes(QKV, HIDDEN), "qkv"), QKV, HIDDEN)
    q_norm = _norm_vector(f.section(HEAD_DIM * 2, "query-norm"), HEAD_DIM)
    k_norm = _norm_vector(f.section(HEAD_DIM * 2, "key-norm"), HEAD_DIM)
    o_proj = _q4_linear(f.section(q4_packed_bytes(HIDDEN, ATTENTION), "attention-output"), HIDDEN, ATTENTION)
    post_attn_norm = _norm_vector(f.section(HIDDEN * 2, "post-attention-norm"), HIDDEN)
    mlp_conv = _bf16_matrix(f.section(4 * HIDDEN * 2, "mlp-convolution"), 2 * CONV_TAPS, HIDDEN)
    mlp_dynamic = _q4_linear(f.section(q4_packed_bytes(DYNAMIC, HIDDEN), "mlp-dynamic"), DYNAMIC, HIDDEN)
    gate = _q4_linear(f.section(q4_packed_bytes(INTERMEDIATE, HIDDEN), "mlp-gate"), INTERMEDIATE, HIDDEN)
    up = _q4_linear(f.section(q4_packed_bytes(INTERMEDIATE, HIDDEN), "mlp-up"), INTERMEDIATE, HIDDEN)
    down = _q4_linear(f.section(q4_packed_bytes(HIDDEN, INTERMEDIATE), "mlp-down"), HIDDEN, INTERMEDIATE)
    f.finish()
    return DraftLayer(
        input_norm=input_norm,
        attn_conv=attn_conv.reshape(2, CONV_TAPS, HIDDEN),
        attn_dynamic=attn_dynamic,
        qkv=qkv,
        q_norm=q_norm,
        k_norm=k_norm,
        o_proj=o_proj,
        post_attn_norm=post_attn_norm,
        mlp_conv=mlp_conv.reshape(2, CONV_TAPS, HIDDEN),
        mlp_dynamic=mlp_dynamic,
        gate=gate,
        up=up,
        down=down,
    )


def load_dflash_draft(directory: str | Path | None = None) -> DraftWeights:
    directory = Path(directory or (Path.home() / ".monkey/models/Qwen3.8-27B-Splash-draft/draft"))
    layers = [_load_layer(directory / f"layer-{i}.bin", i) for i in range(LAYERS)]
    f = PackedFile(directory / "model.bin", MAGIC, LAYERS, 1)
    context_proj = _q4_linear(
        f.section(q4_packed_bytes(HIDDEN, TARGET_HIDDEN), "context-projection"),
        HIDDEN,
        TARGET_HIDDEN,
    )
    hidden_norm = _norm_vector(f.section(HIDDEN * 2, "hidden-norm"), HIDDEN)
    final_norm = _norm_vector(f.section(HIDDEN * 2, "final-norm"), HIDDEN)
    selector = _q4_linear(
        f.section(q4_packed_bytes(SELECTOR_RANK, HIDDEN), "selector"),
        SELECTOR_RANK,
        HIDDEN,
    )
    codebook_bytes = VOCAB * SELECTOR_RANK * 2
    predecessor = _bf16_matrix(f.section(codebook_bytes, "predecessor-codebook"), VOCAB, SELECTOR_RANK)
    successor = _bf16_matrix(f.section(codebook_bytes, "successor-codebook"), VOCAB, SELECTOR_RANK)
    f.finish()
    on_disk = sum((directory / name).stat().st_size for name in [*(f"layer-{i}.bin" for i in range(LAYERS)), "model.bin"])
    weights = DraftWeights(
        layers=layers,
        context_proj=context_proj,
        hidden_norm=hidden_norm,
        final_norm=final_norm,
        selector=selector,
        predecessor=predecessor,
        successor=successor,
        directory=directory,
        bytes_on_disk=on_disk,
    )
    mx.eval(
        *[p for layer in layers for p in (layer.input_norm, layer.attn_conv, layer.q_norm, layer.k_norm, layer.post_attn_norm, layer.mlp_conv)]
        + [hidden_norm, final_norm, predecessor, successor]
    )
    return weights
