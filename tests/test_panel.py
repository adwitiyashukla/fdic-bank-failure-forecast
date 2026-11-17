from datetime import date

import numpy as np
import pandas as pd
import pytest

from bankfail.panel import attach_labels, last_complete_origin, read_fed
from bankfail.quarters import quarter_index


def q(year, quarter):
    return year * 4 + quarter - 1


def events(*rows):
    frame = pd.DataFrame(rows, columns=["CERT", "fail_date"])
    frame["fail_date"] = pd.to_datetime(frame["fail_date"])
    frame["fail_q"] = [quarter_index(d.date()) for d in frame["fail_date"]]
    return frame


def panel(cert, first, count):
    return pd.DataFrame({"CERT": [cert] * count, "q": list(range(first, first + count))})


def test_only_the_four_reports_before_a_failure_are_positive():
    frame = panel(1, q(2008, 1), 6)
    out = attach_labels(frame, events((1, "2009-08-14")), 4, date(2026, 9, 15))
    assert out["event"].tolist() == [0, 0, 1, 1, 1, 1]
    assert out["quarters_to_event"].tolist() == [6, 5, 4, 3, 2, 1]


def test_failure_on_a_quarter_end_counts_for_the_report_before_it():
    frame = panel(1, q(2010, 1), 2)
    out = attach_labels(frame, events((1, "2010-06-30")), 4, date(2026, 9, 15))
    assert out["event"].tolist() == [1, 0]


def test_open_windows_stay_unlabeled_unless_the_bank_already_failed():
    frame = pd.concat([panel(1, q(2025, 2), 4), panel(2, q(2025, 2), 4)], ignore_index=True)
    out = attach_labels(frame, events((2, "2026-07-10")), 4, date(2026, 9, 15))
    first = out[out["CERT"] == 1]["event"]
    second = out[out["CERT"] == 2]["event"]
    assert first.iloc[0] == 0 and first.iloc[1:].isna().all()
    assert second.tolist() == [0, 1, 1, 1]


def test_assistance_rows_are_left_out_of_training():
    frame = panel(3, q(2008, 1), 6)
    ambiguous = events((3, "2009-01-16"))
    out = attach_labels(frame, events(), 4, date(2026, 9, 15), ambiguous)
    assert out["event"].isna().tolist() == [True, True, True, True, False, False]


def test_last_complete_origin():
    assert last_complete_origin(4, date(2026, 9, 15)) == q(2025, 2)
    assert last_complete_origin(4, date(2026, 9, 30)) == q(2025, 3)
    assert last_complete_origin(1, date(2026, 7, 1)) == q(2026, 1)


def test_fed_file_is_read_by_quarter(tmp_path):
    path = tmp_path / "historic.csv"
    path.write_text(
        "Scenario Name,Date,Unemployment rate,House Price Index (Level),"
        "Commercial Real Estate Price Index (Level),Real GDP growth\n"
        "Actual,2025 Q4,4.5,323.4,305.8,1.0\nActual,2025 Q3,4.3,320.0,300.0,2.0\n"
    )
    frame = read_fed(path)
    assert frame["q"].tolist() == [q(2025, 3), q(2025, 4)]
    assert frame["unemployment"].tolist() == [4.3, 4.5]


def test_fed_file_without_unemployment_fails(tmp_path):
    path = tmp_path / "historic.csv"
    path.write_text("Scenario Name,Date,Real GDP growth\nActual,2025 Q4,1.0\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_fed(path)


def test_labels_keep_row_order_for_many_banks():
    rng = np.random.default_rng(0)
    frames = [panel(cert, q(2000, 1) + int(rng.integers(0, 8)), 20) for cert in range(1, 60)]
    frame = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=1)
    failed = events((5, "2004-02-01"), (9, "2002-11-30"))
    out = attach_labels(frame, failed, 4, date(2026, 9, 15))
    assert out.index.equals(frame.index)
    positive = out[out["event"] == 1]
    assert set(positive["CERT"]) == {5, 9}
    assert len(positive) == 8
