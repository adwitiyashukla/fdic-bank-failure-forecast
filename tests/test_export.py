import pandas as pd
import pytest

from bankfail.export import check_public


def write(folder, name, frame):
    frame.to_csv(folder / f"{name}.csv", index=False)


def test_named_tables_may_only_hold_failed_banks(tmp_path):
    write(tmp_path, "case2023", pd.DataFrame({"CERT": [24735, 1], "name": ["SVB", "LIVE BANK"]}))
    with pytest.raises(ValueError, match="did not fail"):
        check_public(tmp_path, {24735})


def test_aggregate_tables_may_not_carry_identifiers(tmp_path):
    write(tmp_path, "current_size", pd.DataFrame({"size": ["under $100M"], "NAME": ["LIVE BANK"]}))
    with pytest.raises(ValueError, match="identifiers"):
        check_public(tmp_path, set())


def test_run_screen_names_only_failed_banks(tmp_path):
    screen = pd.DataFrame(
        {"uninsured_deposits": [90, 10], "failed_within_2y": [False, True], "name": ["LIVE", "SVB"]}
    )
    write(tmp_path, "run_screen", screen)
    with pytest.raises(ValueError, match="did not fail"):
        check_public(tmp_path, set())
    screen["name"] = ["", "SVB"]
    write(tmp_path, "run_screen", screen)
    check_public(tmp_path, set())


def test_clean_folder_passes(tmp_path):
    write(tmp_path, "case2023", pd.DataFrame({"CERT": [24735], "name": ["SVB"]}))
    write(tmp_path, "current_size", pd.DataFrame({"size": ["under $100M"], "banks": [584]}))
    check_public(tmp_path, {24735})
