"""Parity instrumentation for PackedLinear."""

from __future__ import annotations

PARITY_REMAINING = 0
PARITY_HITS: list[dict] = []


def reset_parity(n: int) -> None:
    global PARITY_REMAINING, PARITY_HITS
    PARITY_REMAINING = int(n)
    PARITY_HITS = []


def parity_report() -> list[dict]:
    return list(PARITY_HITS)


def record_parity(max_abs: float, rms: float, shape: tuple) -> None:
    global PARITY_REMAINING
    if PARITY_REMAINING <= 0:
        return
    PARITY_HITS.append({"max_abs": float(max_abs), "rms": float(rms), "shape": list(shape)})
    PARITY_REMAINING -= 1
