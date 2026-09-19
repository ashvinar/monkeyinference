# Measurement probes

These scripts produced the tables in `docs/ternary-engine.md`. They are not part of production decode. Production is greedy 2-bit qdot (`python -m monkeyinference.cli generate`).

| Script | What it measured |
| --- | --- |
| `probe_mixed.py` | Per-shape five-trit on real weights, mixed greedy tok/s, token identity, T=1/T=8 break-even, DFlash reject histogram |
| `probe_five_trit.py` | Global five-trit pack + synthetic-shape GEMV vs qdot (the 0.88× census) |
| `probe_breakeven.py` | DFlash propose(k) + T=8 MMA vs greedy 10.22 |
| `probe_bandwidth.py` | GPU copy/read/write STREAM, CPU read, CPU+GPU aggregate |
| `probe_code_histogram.py` | On-disk affine-2bit codes: 0 of 3.54e9 code-3 |
| `probe_gdn_path.py` | mlx_lm GDN prefill vs verify: same sequential kernel |
| `finetune_dflash.py` | LoRA distill retry: 1200 prompts, r=8, held-out accepts, `~/.monkey/dflash-ft2/` |
| `run_dflash_ft_detached.sh` | tmux + `caffeinate -dims`; default dir `dflash-ft2` |

```bash
export PYTHONPATH=src
~/.monkey/mlx-venv/bin/python scripts/probe_code_histogram.py
~/.monkey/mlx-venv/bin/python scripts/probe_bandwidth.py
~/.monkey/mlx-venv/bin/python scripts/probe_mixed.py
~/.monkey/mlx-venv/bin/python scripts/finetune_dflash.py --scope
# long run: detached from the agent session, Mac kept awake
scripts/run_dflash_ft_detached.sh
scripts/run_dflash_ft_detached.sh --watch <pid>
```

`results/*.json` is gitignored. One-off kernel-tuning probes on disk are not tracked.
