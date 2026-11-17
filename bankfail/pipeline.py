import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from bankfail.cases import failed_paths, run_case, run_current
from bankfail.config import Config
from bankfail.evaluate import labeled_rows, run_design, run_rolling
from bankfail.export import export_demo
from bankfail.features import build_features
from bankfail.panel import (
    attach_labels,
    downloaded_on,
    last_complete_origin,
    load_failures,
    load_financials,
    load_macro,
    matched_failures,
)
from bankfail.quarters import label, quarter_index
from bankfail.stress import run_stress


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def work_file(config: Config, name: str) -> Path:
    return config.paths.work_dir / name


def read_work(config: Config, name: str) -> pd.DataFrame:
    path = work_file(config, name)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found, run the panel step first")
    return pd.read_parquet(path)


def stage_panel(config: Config, log=print) -> dict:
    started = time.perf_counter()
    raw = config.paths.raw_dir
    horizon = config.labels.horizon_quarters
    financials = load_financials(raw, config.download.first_quarter, config.download.last_quarter)
    log(f"loaded {len(financials):,} bank quarters for {financials['CERT'].nunique():,} banks")
    failures = load_failures(raw, config.labels.event_types)
    ambiguous = load_failures(raw, config.labels.ambiguous_types)
    macro = load_macro(raw)
    known = downloaded_on(raw)
    labeled = attach_labels(financials, failures, horizon, known, ambiguous)
    frame = build_features(labeled, failures)
    del labeled
    keep = (frame["q"] >= quarter_index(config.panel.first_origin)) & (
        frame["ASSET"] > config.panel.min_assets
    )
    frame = frame[keep].reset_index(drop=True)
    matched = matched_failures(financials, failures, lookback=8)
    del financials
    work = config.paths.work_dir
    work.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(work / "panel.parquet", index=False)
    matched.to_parquet(work / "failures.parquet", index=False)
    for name, table in macro.items():
        table.to_parquet(work / f"macro_{name}.parquet", index=False)
    last = last_complete_origin(horizon, known)
    labeled_mask = frame["event"].notna()
    complete = frame["q"] <= last
    in_panel = matched["fail_q"] >= quarter_index(config.panel.first_origin)
    summary = {
        "bank_quarters": int(len(frame)),
        "banks": int(frame["CERT"].nunique()),
        "first_quarter": label(int(frame["q"].min())),
        "last_quarter": label(int(frame["q"].max())),
        "outcomes_known_through": known.isoformat(),
        "last_complete_origin": label(last),
        "labeled_rows": int((labeled_mask & complete).sum()),
        "positive_rows": int((frame["event"] == 1).sum()),
        "events_in_window": int(in_panel.sum()),
        "events_matched": int((matched["matched"] & in_panel).sum()),
        "event_types": list(config.labels.event_types),
        "ambiguous_rows": int((~labeled_mask & complete).sum()),
        "seconds": round(time.perf_counter() - started, 1),
    }
    write_json(config.paths.results_dir / "panel_summary.json", summary)
    log(
        f"panel {summary['bank_quarters']:,} rows, {summary['positive_rows']:,} positive rows, "
        f"{summary['events_matched']:,} of {summary['events_in_window']:,} failures matched"
    )
    return summary


def load_labeled(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = read_work(config, "panel.parquet")
    known = downloaded_on(config.paths.raw_dir)
    last = last_complete_origin(config.labels.horizon_quarters, known)
    return panel, labeled_rows(panel, last)


def save(config: Config, tables: dict[str, pd.DataFrame]) -> None:
    folder = config.paths.results_dir
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)


def stage_evaluate(config: Config, log=print) -> dict:
    panel, labeled = load_labeled(config)
    del panel
    failures = read_work(config, "failures.parquet")
    collected: dict[str, list[pd.DataFrame]] = {}
    for design in config.evaluation.designs:
        for key, table in run_design(labeled, design, failures, config, log).items():
            collected.setdefault(key, []).append(table)
    tables = {
        f"design_{key}": pd.concat(parts, ignore_index=True) for key, parts in collected.items()
    }
    save(config, tables)
    for row in tables["design_metrics"].to_dict("records"):
        log(
            f"{row['design']} {row['model']}: AUC {row['auc']:.3f}, "
            f"top {config.evaluation.flag_fraction:.0%} catches "
            f"{row[f'recall_top_{config.evaluation.flag_fraction:g}']:.1%} of failure rows"
        )
    return tables


def stage_rolling(config: Config, log=print) -> pd.DataFrame:
    panel, labeled = load_labeled(config)
    del panel
    rolling_config = replace(
        config, gbm=replace(config.gbm, negative_sample_rate=config.evaluation.rolling_sample_rate)
    )
    metrics, scored = run_rolling(labeled, rolling_config, log)
    failures = read_work(config, "failures.parquet")
    scored.to_parquet(work_file(config, "rolling_scores.parquet"), index=False)
    paths = failed_paths(scored, failures, config.evaluation.warning_lookback)
    save(config, {"rolling": metrics, "failed_paths": paths})
    return metrics


def stage_case(config: Config, log=print) -> dict:
    panel, labeled = load_labeled(config)
    origins = [quarter_index(day) for day in config.case2023.score_origins]
    scored = panel[panel["q"].isin(origins)]
    del panel
    failures = read_work(config, "failures.parquet")
    tables = run_case(scored, labeled, failures, config, log)
    save(config, {"case2023": tables["case"], "run_screen": tables["screen"]})
    return tables


def stage_current(config: Config, log=print) -> dict:
    panel, labeled = load_labeled(config)
    recent = panel[panel["q"] > labeled["q"].max()]
    del panel
    failures = read_work(config, "failures.parquet")
    tables = run_current(recent, labeled, failures, config, log)
    tables["live_check"] = tables.pop("live")
    save(config, tables)
    return tables


def stage_stress(config: Config, log=print) -> dict:
    panel = read_work(config, "panel.parquet")
    macro = {
        name: read_work(config, f"macro_{name}.parquet")
        for name in ("historic", "baseline", "severely_adverse")
    }
    known = downloaded_on(config.paths.raw_dir)
    tables = run_stress(panel, macro, known, config, log)
    save(config, tables)
    return tables


def stage_export(config: Config, log=print) -> Path:
    return export_demo(config, log)


STAGES = {
    "panel": stage_panel,
    "evaluate": stage_evaluate,
    "rolling": stage_rolling,
    "case": stage_case,
    "current": stage_current,
    "stress": stage_stress,
    "export": stage_export,
}


def run_stage(name: str, config: Config, log=print):
    with threadpool_limits(limits=config.gbm.threads, user_api="openmp"):
        return STAGES[name](config, log)


def run_all(config: Config, log=print) -> None:
    for name in STAGES:
        started = time.perf_counter()
        log(f"== {name}")
        run_stage(name, config, log)
        log(f"== {name} done in {time.perf_counter() - started:.1f} s")
