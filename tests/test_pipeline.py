import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from bankfail.config import load_config
from bankfail.export import NAMED, PUBLIC
from bankfail.pipeline import run_all
from bankfail.synth import write_synthetic

ROOT = Path(__file__).resolve().parents[1]


def quiet(message):
    return None


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    base = load_config(ROOT / "config.toml")
    paths = replace(
        base.paths,
        raw_dir=root / "raw",
        work_dir=root / "work",
        results_dir=root / "results",
        demo_dir=root / "demo",
    )
    gbm = replace(
        base.gbm,
        max_iter=60,
        fallback_iter=30,
        min_iter=10,
        patience=10,
        min_validation_failures=20,
        min_samples_leaf=50,
    )
    evaluation = replace(
        base.evaluation, bootstrap_reps=5, rolling_first_year=2005, rolling_last_year=2010
    )
    config = replace(
        base,
        paths=paths,
        gbm=gbm,
        evaluation=evaluation,
        stress=replace(base.stress, simulations=200),
    )
    write_synthetic(config, 400, log=quiet)
    run_all(config, log=quiet)
    shutil.copy(ROOT / "app.py", config.paths.demo_dir.parent / "app.py")
    return config


def test_every_demo_table_is_written(built):
    names = {path.stem for path in built.paths.demo_dir.glob("*.csv")}
    assert set(PUBLIC) | set(NAMED) | {"run_screen"} <= names


def test_boosting_finds_the_synthetic_signal(built):
    metrics = pd.read_csv(built.paths.results_dir / "design_metrics.csv")
    assert set(metrics["design"]) == {"crisis", "recent"}
    crisis = metrics[(metrics["design"] == "crisis") & (metrics["model"] == "gbm")].iloc[0]
    assert crisis["positive_rows"] > 0
    assert crisis["auc"] > 0.7


def test_severe_scenario_expects_more_failures(built):
    scenarios = pd.read_csv(built.paths.results_dir / "stress_scenarios.csv").set_index("scenario")
    assert (
        scenarios.loc["severely_adverse", "expected_failures"]
        >= (scenarios.loc["baseline", "expected_failures"])
    )
    assert scenarios.loc["severely_adverse", "path_unemployment_rise"] > 0


def test_rolling_covers_the_configured_years(built):
    rolling = pd.read_csv(built.paths.results_dir / "rolling.csv")
    assert set(rolling["year"]) == set(range(2005, 2011))


def test_app_renders_every_tab(built):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(built.paths.demo_dir.parent / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception
    assert len(app.tabs) == 6
    assert not app.error
