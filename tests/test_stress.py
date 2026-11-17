from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from bankfail.config import load_config
from bankfail.stress import PATH, path_table, scenario_paths, stress_frame


def levels(q, unemployment, house, cre):
    return pd.DataFrame(
        {"q": q, "unemployment": unemployment, "house_prices": house, "cre_prices": cre}
    )


def test_path_table_looks_ahead_over_the_horizon():
    table = path_table(
        levels(
            range(8),
            [4, 5, 7, 6, 5, 4, 4, 4],
            [100, 90, 80, 85, 80, 80, 80, 80],
            [100] * 4 + [50] * 4,
        ),
        4,
    )
    assert table.loc[0].tolist() == [3, -20, -50]
    assert table.loc[4:].isna().all(axis=None)


def test_stress_labels_use_their_own_horizon():
    q0 = 2008 * 4
    panel = pd.DataFrame(
        {
            "CERT": [1] * 10,
            "q": range(q0, q0 + 10),
            "event": [0.0] * 10,
            "quarters_to_event": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "quarters_to_other": [np.nan] * 10,
        }
    )
    history = levels(range(q0 - 4, q0 + 30), 5.0, 100.0, 100.0)
    frame = stress_frame(panel, history, date(2026, 9, 15), 8)
    assert frame["event"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert set(PATH) <= set(frame.columns)


def test_assistance_within_the_horizon_is_left_out():
    q0 = 2008 * 4
    panel = pd.DataFrame(
        {
            "CERT": [1, 1],
            "q": [q0, q0 + 1],
            "event": [0.0, np.nan],
            "quarters_to_event": [np.nan, np.nan],
            "quarters_to_other": [9, 8],
        }
    )
    frame = stress_frame(panel, levels(range(q0, q0 + 20), 5.0, 100.0, 100.0), date(2026, 9, 15), 8)
    assert frame["event"].tolist()[0] == 0
    assert np.isnan(frame["event"].tolist()[1])


def test_scenario_paths_start_from_the_last_actual_quarter(config_path):
    config = load_config(config_path)
    config = replace(
        config,
        stress=replace(config.stress, horizon_quarters=2, reference_origin=date(2000, 3, 31)),
    )
    jump = 2025 * 4 + 3
    history = levels(range(2000 * 4, jump + 1), 4.0, 200.0, 100.0)
    future = levels(
        [jump + 1, jump + 2, jump + 3], [6.0, 9.0, 11.0], [180.0, 150.0, 100.0], [90.0, 70.0, 10.0]
    )
    calm = levels([jump + 1, jump + 2, jump + 3], 4.0, 200.0, 100.0)
    table = scenario_paths(
        {"historic": history, "baseline": calm, "severely_adverse": future}, config
    )
    severe = table.set_index("scenario").loc["severely_adverse"]
    assert severe["path_unemployment_rise"] == pytest.approx(5.0)
    assert severe["path_house_price_change"] == pytest.approx(-25.0)
    assert severe["path_cre_price_change"] == pytest.approx(-30.0)
    assert table.set_index("scenario").loc["baseline", "path_unemployment_rise"] == 0
    assert table["jump_off"].unique().tolist() == ["2025Q4"]


def test_scenario_that_is_too_short_fails(config_path):
    config = load_config(config_path)
    jump = 2025 * 4 + 3
    history = levels(range(2000 * 4, jump + 1), 4.0, 200.0, 100.0)
    short = levels([jump + 1], 5.0, 190.0, 95.0)
    with pytest.raises(ValueError, match="fewer than"):
        scenario_paths({"historic": history, "baseline": short, "severely_adverse": short}, config)
