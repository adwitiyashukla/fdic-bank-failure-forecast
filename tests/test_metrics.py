import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bankfail.metrics import (
    calibration_table,
    capture,
    cluster_bootstrap_auc,
    flag_top,
    warning_quarters,
)


def test_flag_top_takes_the_ceiling_per_quarter():
    q = np.array([1, 1, 1, 2, 2])
    score = np.array([0.1, 0.9, 0.5, 0.2, 0.3])
    assert flag_top(q, score, 0.34).tolist() == [False, True, True, False, True]


def test_capture_matches_brute_force():
    rng = np.random.default_rng(3)
    q = rng.integers(0, 6, 3000)
    y = (rng.random(3000) < 0.05).astype(int)
    score = y * 0.3 + rng.random(3000)
    recall, precision = capture(q, y, score, 0.1)
    flagged = np.zeros(3000, dtype=bool)
    for value in np.unique(q):
        rows = np.flatnonzero(q == value)
        top = rows[np.argsort(-score[rows], kind="stable")][: int(np.ceil(len(rows) * 0.1))]
        flagged[top] = True
    assert recall == (flagged & (y == 1)).sum() / y.sum()
    assert precision == (flagged & (y == 1)).sum() / flagged.sum()


def test_warning_time_needs_an_unbroken_streak():
    frame = pd.DataFrame(
        {
            "CERT": [7] * 5 + [8] * 5,
            "q": list(range(10, 15)) * 2,
            "quarters_to_event": [5, 4, 3, 2, 1] * 2,
        }
    )
    score = np.array([0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1])
    out = warning_quarters(frame, score, 0.5, 12, 4).set_index("CERT")
    assert out.loc[7, "warning_quarters"] == 3
    assert out.loc[7, "first_flag_quarters"] == 5
    assert out.loc[8, "warning_quarters"] == 0
    assert bool(out.loc[8, "flagged_at_last_report"]) is False


def test_warning_time_skips_banks_that_fail_after_the_window():
    frame = pd.DataFrame({"CERT": [1, 1], "q": [3, 4], "quarters_to_event": [7, 6]})
    assert warning_quarters(frame, np.array([1.0, 1.0]), 1.0, 12, 4).empty


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(5)
    y = (rng.random(4000) < 0.1).astype(int)
    score = y + rng.normal(0, 1, 4000)
    groups = np.arange(4000) // 4
    low, high = cluster_bootstrap_auc(y, score, groups, 60, 2)
    assert low < roc_auc_score(y, score) < high


def test_calibration_table_splits_into_bins():
    y = np.array([0, 0, 0, 1] * 25)
    p = np.linspace(0, 1, 100)
    table = calibration_table(y, p, 10)
    assert len(table) == 10
    assert table["rows"].sum() == 100
