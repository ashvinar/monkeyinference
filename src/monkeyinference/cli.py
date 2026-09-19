"""CLI: profile, kernel parity, and end-to-end benches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="monkeyinference")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prof = sub.add_parser("profile", help="STREAM + ternary GEMV roofline (no 27B load)")
    p_prof.add_argument("--out", type=Path, default=None)

    p_gen = sub.add_parser("generate", help="Greedy or speculative generate")
    p_gen.add_argument("--pack", type=Path, default=None)
    p_gen.add_argument("--prompt", default="Name the capital of France. Reply with only the city name.")
    p_gen.add_argument("--max-tokens", type=int, default=32)
    p_gen.add_argument("--speculative", action="store_true")
    p_gen.add_argument("--draft", choices=("none", "pld", "early"), default="pld")
    p_gen.add_argument("--num-draft", type=int, default=None)
    p_gen.add_argument("--early-layers", type=int, default=4)
    p_gen.add_argument("--custom", action=argparse.BooleanOptionalAction, default=True,
                       help="Custom qdot GEMV on M=1 (default on; --no-custom uses MLX)")
    p_gen.add_argument("--parity", type=int, default=0)
    p_gen.add_argument("--out", type=Path, default=None)

    p_bench = sub.add_parser("bench", help="Coherence + speed vs the 8 tok/s starting point")
    p_bench.add_argument("--pack", type=Path, default=None)
    p_bench.add_argument("--out", type=Path, default=None)
    p_bench.add_argument("--parity", type=int, default=8)
    p_bench.add_argument("--quick", action="store_true", help="Skip long prefill; early-exit N=4 only")

    args = parser.parse_args(argv)

    if args.cmd == "profile":
        from monkeyinference.roofline import run

        rep = run(args.out)
        print(json.dumps({
            "stream_gbs": rep["stream"]["gbs"],
            "roofline_decode_tps_measured": rep["roofline"]["decode_tps_at_measured_with_bias"],
            "starting_decode_tps": rep["roofline"]["starting_decode_tps"],
            "fraction_of_measured": rep["roofline"]["starting_fraction_of_measured"],
            "qmv_speedups": [
                {
                    "name": r["custom"]["name"],
                    "custom_gbs": r["custom"]["gbs"],
                    "mlx_gbs": r["mlx_affine"]["gbs"],
                    "speedup": r["speedup"],
                }
                for r in rep.get("qmv", [])
            ],
            "qmm_batch": rep.get("qmm_batch"),
        }, indent=2))
        return 0

    if args.cmd == "generate":
        from monkeyinference.generate import generate
        from monkeyinference.load import load_text_model

        loaded = load_text_model(args.pack, use_custom_kernels=args.custom)
        result = generate(
            loaded,
            args.prompt,
            max_tokens=args.max_tokens,
            speculative=args.speculative,
            draft=args.draft,
            num_draft=args.num_draft,
            early_layers=args.early_layers,
            parity_layers=args.parity,
        )
        payload = result.to_dict()
        print(json.dumps(payload, indent=2))
        if args.out:
            args.out.write_text(json.dumps(payload, indent=2))
        return 0

    if args.cmd == "bench":
        from monkeyinference.bench import run_bench

        rep = run_bench(args.pack, parity_layers=args.parity, quick=args.quick)
        print(json.dumps(rep, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rep, indent=2))
        ok = rep.get("coherence_ok") and rep.get("identity_ok")
        return 0 if ok else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
