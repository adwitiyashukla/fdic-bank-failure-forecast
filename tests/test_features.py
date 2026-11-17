import numpy as np
import pandas as pd

from bankfail.download import FINANCIAL_FIELDS
from bankfail.features import (
    CAP,
    bank_features,
    lagged,
    local_features,
    ratio,
    stressed_ratio,
)


def raw_frame(rows):
    frame = pd.DataFrame(rows)
    for field in FINANCIAL_FIELDS:
        if field not in frame:
            frame[field] = np.nan
    frame["NAME"] = frame["NAME"].fillna("BANK")
    frame["STALP"] = frame["STALP"].fillna("OH")
    frame["BKCLASS"] = frame["BKCLASS"].fillna("NM")
    frame["ESTYMD"] = frame["ESTYMD"].fillna("20000101")
    numeric = [
        f
        for f in FINANCIAL_FIELDS
        if f not in ("CERT", "NAME", "STALP", "BKCLASS", "ESTYMD", "REPDTE")
    ]
    frame[numeric] = frame[numeric].astype("float32")
    return frame


def test_ratio_skips_zero_denominators():
    assert np.isnan(ratio([1.0], [0.0])[0])
    assert ratio([1.0], [4.0])[0] == 25.0


def test_stressed_ratio_caps_broken_banks():
    out = stressed_ratio([10.0, 10.0, 1.0], [-5.0, 0.0, 1e-9])
    assert out.tolist() == [CAP, CAP, CAP]


def test_lag_uses_quarter_numbers_not_row_order():
    frame = pd.DataFrame({"CERT": [1, 1, 1, 2], "q": [100, 104, 105, 104]})
    values = np.array([10.0, 20.0, 30.0, 40.0])
    out = lagged(frame, values, 4)
    assert np.isnan(out[0]) and out[1] == 10.0 and np.isnan(out[2]) and np.isnan(out[3])


def test_texas_ratio_and_growth():
    frame = raw_frame(
        [
            {
                "CERT": 1,
                "q": 8000,
                "ASSET": 100,
                "LNLS": 60,
                "EQ": 10,
                "INTAN": 2,
                "NCLNLS": 3,
                "ORE": 1,
                "LNATRES": 2,
                "DEP": 80,
            },
            {
                "CERT": 1,
                "q": 8004,
                "ASSET": 120,
                "LNLS": 90,
                "EQ": 5,
                "INTAN": 0,
                "NCLNLS": 9,
                "ORE": 3,
                "LNATRES": 1,
                "DEP": 60,
            },
        ]
    )
    out = bank_features(frame)
    assert out["texas_ratio"].tolist() == [40.0, 200.0]
    assert np.isnan(out["asset_growth"].iloc[0])
    assert np.isclose(out["asset_growth"].iloc[1], 20.0)
    assert np.isclose(out["deposit_growth"].iloc[1], -25.0)
    assert np.isclose(out["noncurrent_change"].iloc[1], 10.0 - 5.0)


def test_features_never_look_ahead():
    rng = np.random.default_rng(4)
    rows = []
    for cert in range(1, 6):
        for q in range(8000, 8012):
            rows.append(
                {
                    "CERT": cert,
                    "q": q,
                    "ASSET": rng.uniform(50, 150),
                    "LNLS": rng.uniform(20, 50),
                    "EQ": rng.uniform(1, 15),
                    "NCLNLS": rng.uniform(0, 5),
                    "DEP": rng.uniform(40, 120),
                    "ROA": rng.normal(),
                    "ORE": rng.uniform(0, 2),
                    "LNATRES": 1.0,
                }
            )
    frame = raw_frame(rows)
    before = bank_features(frame)
    future = frame["q"] > 8006
    changed = frame.copy()
    for column in ("ASSET", "LNLS", "EQ", "NCLNLS", "DEP", "ROA", "ORE"):
        changed.loc[future, column] = changed.loc[future, column] * 7 + 3
    after = bank_features(changed)
    past = ~future
    pd.testing.assert_frame_equal(before[past], after[past])


def test_state_failure_rate_counts_only_the_last_four_quarters():
    frame = pd.DataFrame(
        {"CERT": [1, 2, 3, 1, 2, 3], "q": [10, 10, 10, 14, 14, 14], "STALP": ["TX"] * 6}
    )
    failures = pd.DataFrame({"CERT": [3], "fail_q": [10]})
    out = local_features(frame, failures, pd.Series(np.ones(6)))
    assert np.allclose(out["state_failure_rate"], [100 / 3] * 3 + [0.0] * 3)
