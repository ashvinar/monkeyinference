"""Resume + paper-table helpers + retry prompt bank. No 27B load."""

from __future__ import annotations

import json
from pathlib import Path

from monkeyinference.bench import EXPLAIN_PROMPT
from monkeyinference.dflash_distill import (
    GREEDY_TPS,
    MEASURED_GREEDY_TPS,
    MIN_ACCEPTS_TO_BEAT_GREEDY,
    MIN_WINDOWS,
    WIN_ACCEPTS,
    infer_resume_step,
    iter_windows,
    predicted_k7,
    read_train_state,
)
from monkeyinference.dflash_prompts import (
    HOLDOUT_PROMPTS,
    assert_holdout_disjoint,
    distill_prompts,
)


def test_infer_resume_empty(tmp_path: Path):
    assert infer_resume_step(tmp_path) == 0
    assert read_train_state(tmp_path) is None


def test_infer_resume_prefers_train_state(tmp_path: Path):
    (tmp_path / "train_state.json").write_text(
        json.dumps({"step": 3600, "steps": 5000, "loss": 0.1})
    )
    (tmp_path / "run.log").write_text("  step 3999/5000 loss=0.01 0.35s\n")
    assert infer_resume_step(tmp_path) == 3600


def test_infer_resume_from_log_mid_chunk(tmp_path: Path):
    (tmp_path / "run.log").write_text(
        "  step 3400/5000 loss=0.1 0.35s\n"
        "  step 3675/5000 loss=0.02 0.35s\n"
        "  step 3700/5000 loss=0.02 0.34s\n"
    )
    # CKPT_EVERY is 100 on the retry; 3700 // 100 * 100 = 3700
    assert infer_resume_step(tmp_path) == 3700


def test_infer_resume_from_log_just_checkpointed(tmp_path: Path):
    (tmp_path / "run.log").write_text("  step 3599/5000 loss=0.05 0.35s\n")
    assert infer_resume_step(tmp_path) == 3600


def test_infer_resume_from_log_no_checkpoint_yet(tmp_path: Path):
    (tmp_path / "run.log").write_text("  step 25/5000 loss=1.2 0.40s\n")
    assert infer_resume_step(tmp_path) == 0


def test_predicted_k7_stock_accepts_lose():
    row = predicted_k7(3.00)
    assert abs(row["min_accepts_to_beat_10_22"] - MIN_ACCEPTS_TO_BEAT_GREEDY) < 1e-6
    assert MIN_ACCEPTS_TO_BEAT_GREEDY > 3.50
    assert MIN_ACCEPTS_TO_BEAT_GREEDY < 3.53
    assert row["pred_tok_s"] < GREEDY_TPS
    assert row["wins_on_paper"] is False


def test_predicted_k7_threshold_wins():
    need = MIN_ACCEPTS_TO_BEAT_GREEDY
    assert predicted_k7(need + 0.01)["wins_on_paper"] is True
    assert predicted_k7(need - 0.01)["wins_on_paper"] is False


def test_win_bar_from_measured_greedy():
    # 10.75 tok/s × 344.2 ms pass = 3.700 accepts
    assert MEASURED_GREEDY_TPS == 10.75
    assert abs(WIN_ACCEPTS - 3.70) < 0.01
    assert WIN_ACCEPTS > MIN_ACCEPTS_TO_BEAT_GREEDY
    assert MIN_WINDOWS == 30320


def test_prompt_bank_disjoint_and_diverse():
    train = distill_prompts(n=1200)
    assert len(train) == 1200
    assert len({p for p, _ in train}) == 1200
    assert_holdout_disjoint(train)
    assert EXPLAIN_PROMPT not in {p for p, _ in train}
    assert EXPLAIN_PROMPT == HOLDOUT_PROMPTS[0][0]
    n_long = sum(1 for p, _ in train if p.startswith("You are reading an internal design note"))
    assert n_long >= 80, n_long
    kinds = " ".join(p for p, _ in train[:200])
    assert "Write a small Python" in kinds or "unit test" in kinds or "Code" in kinds


def test_iter_windows_are_assistant_tokens():
    row = {"tokens": list(range(80)), "prompt_len": 30}
    wins = list(iter_windows(row))
    assert wins[0] == 29
    assert wins[-1] == 80 - 7 - 1


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_infer_resume_empty(p)
        test_infer_resume_prefers_train_state(p)
    with tempfile.TemporaryDirectory() as d:
        test_infer_resume_from_log_mid_chunk(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_infer_resume_from_log_just_checkpointed(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_infer_resume_from_log_no_checkpoint_yet(Path(d))
    test_predicted_k7_stock_accepts_lose()
    test_predicted_k7_threshold_wins()
    test_win_bar_from_measured_greedy()
    test_prompt_bank_disjoint_and_diverse()
    test_iter_windows_are_assistant_tokens()
    print("all passed")
