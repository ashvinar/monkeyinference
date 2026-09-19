"""Histogram of Prism affine-2bit codes: ternary (0,1,2) vs all 4 levels.

Reads packed uint32 weights from the Bonsai safetensors via memory map.
Does not load the 27B model onto the GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

PACK = Path.home() / ".monkey/models/Ternary-Bonsai-2-27B-mlx-2bit/model.safetensors"
OUT = Path("results/code_histogram.json")
GROUP = 128
PACK_N = 16  # 2-bit codes per uint32
WORDS_PER_GROUP = GROUP // PACK_N  # 8

# Representative packed modules. Keys use the on-disk language_model. prefix.
SAMPLES = [
    ("embed_tokens", "language_model.model.embed_tokens.weight"),
    ("lm_head", "language_model.lm_head.weight"),
    ("L0_gdn_qkv", "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"),
    ("L0_gdn_out", "language_model.model.layers.0.linear_attn.out_proj.weight"),
    ("L0_mlp_up", "language_model.model.layers.0.mlp.up_proj.weight"),
    ("L0_mlp_down", "language_model.model.layers.0.mlp.down_proj.weight"),
    ("L3_attn_q", "language_model.model.layers.3.self_attn.q_proj.weight"),
    ("L3_attn_k", "language_model.model.layers.3.self_attn.k_proj.weight"),
    ("L3_attn_v", "language_model.model.layers.3.self_attn.v_proj.weight"),
    ("L3_attn_o", "language_model.model.layers.3.self_attn.o_proj.weight"),
    ("L3_mlp_up", "language_model.model.layers.3.mlp.up_proj.weight"),
    ("L3_mlp_down", "language_model.model.layers.3.mlp.down_proj.weight"),
    ("L31_gdn_qkv", "language_model.model.layers.31.linear_attn.in_proj_qkv.weight"),
    ("L31_mlp_up", "language_model.model.layers.31.mlp.up_proj.weight"),
    ("L31_mlp_down", "language_model.model.layers.31.mlp.down_proj.weight"),
    ("L63_attn_q", "language_model.model.layers.63.self_attn.q_proj.weight"),
    ("L63_attn_o", "language_model.model.layers.63.self_attn.o_proj.weight"),
    ("L63_mlp_up", "language_model.model.layers.63.mlp.up_proj.weight"),
    ("L63_mlp_down", "language_model.model.layers.63.mlp.down_proj.weight"),
]


def _count_codes(w: np.ndarray) -> np.ndarray:
    """w: uint32 [N, K/16] → counts of codes {0,1,2,3}."""
    counts = np.zeros(4, dtype=np.int64)
    for lane in range(PACK_N):
        c = (w >> np.uint32(2 * lane)) & np.uint32(3)
        counts += np.bincount(c.ravel(), minlength=4).astype(np.int64)
    return counts


def _group_stats(w: np.ndarray, row_chunk: int = 2048) -> dict:
    """Per g128 group: level occupancy and exact-zero (code==1) fraction."""
    n, words = w.shape
    if words % WORDS_PER_GROUP:
        raise ValueError(f"K/16={words} is not a multiple of {WORDS_PER_GROUP}")
    ng = words // WORDS_PER_GROUP
    n_groups = 0
    groups_with_3 = 0
    groups_ternary_only = 0  # no code 3
    groups_using_all4 = 0
    groups_using_only_01 = 0
    zero_code_sum = 0  # code==1 counts across groups (then / 128)
    code3_fracs = []
    for r0 in range(0, n, row_chunk):
        block = np.ascontiguousarray(w[r0 : r0 + row_chunk])
        g = block.reshape(block.shape[0], ng, WORDS_PER_GROUP)
        hist = np.zeros(g.shape[:2] + (4,), dtype=np.int32)
        for lane in range(PACK_N):
            c = (g >> np.uint32(2 * lane)) & np.uint32(3)
            for v in range(4):
                hist[:, :, v] += (c == v).sum(axis=-1)
        used = hist > 0
        n_lev = used.sum(axis=-1)
        has3 = used[:, :, 3]
        groups_with_3 += int(has3.sum())
        groups_ternary_only += int((~has3).sum())
        groups_using_all4 += int((n_lev == 4).sum())
        groups_using_only_01 += int(((~used[:, :, 2]) & (~used[:, :, 3])).sum())
        n_here = int(hist.shape[0] * hist.shape[1])
        n_groups += n_here
        zero_code_sum += int(hist[:, :, 1].sum())
        # subsample fracs to keep the json small: mean/p50/p99 of code-3 fraction
        code3_fracs.append((hist[:, :, 3].astype(np.float64) / GROUP).ravel())
    fr = np.concatenate(code3_fracs)
    return {
        "n_groups": n_groups,
        "groups_with_code3": groups_with_3,
        "groups_ternary_only": groups_ternary_only,
        "groups_using_all4": groups_using_all4,
        "groups_no_plus": groups_using_only_01,
        "frac_groups_with_code3": groups_with_3 / n_groups if n_groups else None,
        "frac_groups_ternary_only": groups_ternary_only / n_groups if n_groups else None,
        "frac_groups_all4": groups_using_all4 / n_groups if n_groups else None,
        "code3_frac_mean": float(fr.mean()) if fr.size else None,
        "code3_frac_p50": float(np.median(fr)) if fr.size else None,
        "code3_frac_p99": float(np.quantile(fr, 0.99)) if fr.size else None,
        "code3_frac_max": float(fr.max()) if fr.size else None,
        "exact_zero_from_groups": zero_code_sum / (n_groups * GROUP) if n_groups else None,
    }


def main():
    rows = []
    agg = np.zeros(4, dtype=np.int64)
    print(f"{'tensor':<16} {'shape':>18}  {'c0':>7} {'c1=0':>7} {'c2':>7} {'c3':>7}  "
          f"{'%c3':>7} {'%zero':>7}  {'grp%ternary':>12} {'grp%all4':>9}")
    with safe_open(str(PACK), framework="numpy") as f:
        keys = set(f.keys())
        for name, key in SAMPLES:
            if key not in keys:
                # try without language_model. prefix
                alt = key.removeprefix("language_model.")
                if alt in keys:
                    key = alt
                else:
                    print(f"MISSING {name} tried {key}")
                    continue
            w = np.asarray(f.get_tensor(key))
            if w.dtype != np.uint32:
                w = w.view(np.uint32)
            counts = _count_codes(w)
            agg += counts
            total = int(counts.sum())
            gstat = _group_stats(w)
            frac = counts / total
            row = {
                "name": name,
                "key": key,
                "shape": [int(x) for x in w.shape],
                "n_codes": total,
                "counts": {str(i): int(counts[i]) for i in range(4)},
                "frac": {str(i): float(frac[i]) for i in range(4)},
                "exact_zero_frac": float(frac[1]),
                "code3_frac": float(frac[3]),
                "groups": gstat,
            }
            rows.append(row)
            print(
                f"{name:<16} {str(tuple(w.shape)):>18}  "
                f"{frac[0]:7.3f} {frac[1]:7.3f} {frac[2]:7.3f} {frac[3]:7.3f}  "
                f"{100*frac[3]:6.3f}% {100*frac[1]:6.2f}%  "
                f"{100*gstat['frac_groups_ternary_only']:11.2f}% "
                f"{100*gstat['frac_groups_all4']:8.2f}%"
            )
    tot = int(agg.sum())
    frac = agg / tot
    uses4 = bool(agg[3] > 0)
    if uses4:
        read = (
            "USES_ALL_4: code 3 is present. The pack is a 4-level affine-2bit grid, "
            "not a trit store. 5-trits-per-byte is not a lossless re-encoding. "
            "The idea dies here."
        )
    else:
        read = (
            "TERNARY: codes are in {0,1,2} only. A 5-trits-per-byte pack is a "
            "lossless re-encoding on the table (codes 6.822 GB → ~5.5 GB; stream "
            "with biases dropped ~7.674 → ~5.9 GB). Unpack ALU vs bytes is the "
            "remaining cost question; do not build the pack in this pass."
        )
    report = {
        "pack": str(PACK),
        "n_tensors": len(rows),
        "aggregate_counts": {str(i): int(agg[i]) for i in range(4)},
        "aggregate_frac": {str(i): float(frac[i]) for i in range(4)},
        "aggregate_n_codes": tot,
        "exact_zero_frac": float(frac[1]),
        "code3_frac": float(frac[3]),
        "uses_all_4_levels": uses4,
        "read": read,
        "tensors": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print()
    print(
        f"AGGREGATE  n={tot:,}  "
        f"c0={frac[0]:.4f}  c1(zero)={frac[1]:.4f}  c2={frac[2]:.4f}  c3={frac[3]:.4f}"
    )
    print(read)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
