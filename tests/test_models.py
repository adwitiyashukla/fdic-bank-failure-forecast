from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bankfail.config import load_config
from bankfail.models import (
    fit_gbm,
    fit_logit,
    monotone_signs,
    prepare,
    sample_rows,
    validation_split,
)


@pytest.fixture
def config(config_path):
    base = load_config(config_path)
    gbm = replace(
        base.gbm,
        max_iter=80,
        fallback_iter=40,
        min_iter=10,
        patience=10,
        validation_quarters=8,
        min_validation_failures=20,
        min_samples_leaf=20,
    )
    return replace(base, gbm=gbm)


def toy(rows=6000, seed=0):
    rng = np.random.default_rng(seed)
    equity = rng.uniform(2, 20, rows)
    bad = rng.uniform(0, 10, rows)
    risk = -4 + 0.5 * bad - 0.3 * equity
    event = (rng.random(rows) < 1 / (1 + np.exp(-risk))).astype("float32")
    return pd.DataFrame(
        {
            "CERT": np.arange(rows) % 300,
            "q": 8000 + np.arange(rows) % 40,
            "equity_ratio": equity.astype("float32"),
            "noncurrent_loans": bad.astype("float32"),
            "charter": (np.arange(rows) % 3).astype("float32"),
            "event": event,
        }
    )


def test_boosting_respects_monotone_signs(config):
    frame = toy()
    model = fit_gbm(frame, ("equity_ratio", "noncurrent_loans", "charter"), config)
    grid = pd.DataFrame(
        {
            "equity_ratio": np.linspace(2, 20, 50, dtype="float32"),
            "noncurrent_loans": np.full(50, 5.0, dtype="float32"),
            "charter": np.zeros(50, dtype="float32"),
        }
    )
    p = model.predict(grid)
    assert np.all(np.diff(p) <= 1e-12)
    grid["equity_ratio"] = 8.0
    grid["noncurrent_loans"] = np.linspace(0, 10, 50, dtype="float32")
    assert np.all(np.diff(model.predict(grid)) >= -1e-12)


def test_boosting_skips_columns_that_have_no_data_yet(config):
    frame = toy()
    frame["past_due_30_89"] = np.where(frame["q"] >= 8030, 1.0, np.nan).astype("float32")
    frame["unrealized_loss_to_equity"] = np.float32("nan")
    columns = ("equity_ratio", "past_due_30_89", "unrealized_loss_to_equity")
    model = fit_gbm(frame, columns, config)
    assert model.columns == ("equity_ratio", "past_due_30_89")
    assert np.isfinite(model.predict(frame)).all()


def test_monotone_signs_leave_unknown_columns_free():
    assert monotone_signs(("equity_ratio", "texas_ratio", "charter")) == [-1, 1, 0]


def test_validation_window_is_the_latest_and_leaves_a_gap(config):
    frame = toy()
    fitting, validation = validation_split(frame, config)
    q = frame["q"].to_numpy()
    assert q[validation].min() > q[fitting].max() + config.labels.horizon_quarters
    assert q[validation].max() == q.max()


def test_too_few_failures_falls_back(config):
    frame = toy()
    frame["event"] = 0.0
    frame.loc[frame.index[:3], "event"] = 1.0
    assert validation_split(frame, config) is None


def test_negative_sampling_keeps_every_failure(config):
    frame = toy()
    sample, weights = sample_rows(frame, 0.25, 1)
    assert sample["event"].sum() == frame["event"].sum()
    assert set(np.unique(weights)) == {1.0, 4.0}
    assert abs(weights.sum() - len(frame)) / len(frame) < 0.05


def test_logit_handles_missing_and_constant_columns(config):
    frame = toy()
    frame["constant"] = 1.0
    frame.loc[frame.index[::7], "noncurrent_loans"] = np.nan
    model = fit_logit(frame, ("equity_ratio", "noncurrent_loans", "constant"), config)
    p = model.predict(frame)
    assert np.isfinite(p).all()
    assert model.prepared.flagged.tolist() == [False, True, False]
    assert model.contributions(frame.head()).shape == (5, 3)


def test_prepare_clips_to_training_quantiles():
    frame = pd.DataFrame({"x": np.arange(1000, dtype="float64")})
    prepared = prepare(frame, ("x",), 0.01, 0.99)
    wild = pd.DataFrame({"x": [-1e9, 1e9]})
    out = prepared.transform(wild) * prepared.scale + prepared.mean
    assert out[0, 0] == pytest.approx(prepared.lower[0])
    assert out[1, 0] == pytest.approx(prepared.upper[0])
