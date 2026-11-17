import json
import shutil
from pathlib import Path

import pandas as pd

from bankfail.config import Config

PUBLIC = (
    "design_metrics",
    "design_curve",
    "design_quarterly",
    "design_calibration",
    "design_importance",
    "rolling",
    "stress_backtest",
    "stress_scenarios",
    "stress_size",
    "current_summary",
    "current_size",
    "current_histogram",
)
NAMED = ("design_warnings", "case2023", "live_check", "failed_paths")
IDENTIFIERS = ("CERT", "name", "NAME", "place")


def failed_certs(config: Config) -> set[int]:
    path = config.paths.work_dir / "failures.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found, run the panel step first")
    return set(pd.read_parquet(path, columns=["CERT"])["CERT"].astype(int))


def read_result(config: Config, name: str) -> pd.DataFrame:
    path = config.paths.results_dir / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found, run the earlier steps first")
    return pd.read_csv(path)


def screen_table(config: Config) -> pd.DataFrame:
    screen = read_result(config, "run_screen")
    case = read_result(config, "case2023")
    names = case.drop_duplicates("CERT").set_index("CERT")["name"]
    failed = screen["failed_within_2y"].astype(bool)
    screen["name"] = screen["CERT"].map(names).where(failed, "")
    return screen.drop(columns="CERT").fillna({"name": ""})


def check_public(folder: Path, failed: set[int]) -> None:
    for path in sorted(folder.glob("*.csv")):
        table = pd.read_csv(path, keep_default_na=False)
        if path.stem in NAMED:
            unknown = set(pd.to_numeric(table["CERT"]).astype(int)) - failed
            if unknown:
                raise ValueError(f"{path.name} names banks that did not fail: {sorted(unknown)}")
            continue
        if path.stem == "run_screen":
            named = table[table["name"] != ""]
            if not named["failed_within_2y"].astype(str).isin(["True", "true", "1"]).all():
                raise ValueError("run_screen.csv names a bank that did not fail")
            continue
        leaked = [column for column in IDENTIFIERS if column in table.columns]
        if leaked:
            raise ValueError(f"{path.name} carries bank identifiers {leaked}")


def export_demo(config: Config, log=print) -> Path:
    target = config.paths.demo_dir
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    failed = failed_certs(config)
    for name in PUBLIC:
        read_result(config, name).to_csv(target / f"{name}.csv", index=False)
    for name in NAMED:
        table = read_result(config, name)
        table = table[table["CERT"].astype(int).isin(failed)]
        table.to_csv(target / f"{name}.csv", index=False)
    screen_table(config).to_csv(target / "run_screen.csv", index=False)
    summary = json.loads((config.paths.results_dir / "panel_summary.json").read_text())
    (target / "panel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    check_public(target, failed)
    size = sum(path.stat().st_size for path in target.iterdir())
    log(f"demo tables written to {target} ({size / 1e6:.1f} MB), no live bank is named")
    return target
