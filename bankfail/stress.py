from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from bankfail.cases import SIZE_LABELS, simulate_total, size_bucket
from bankfail.config import Config
from bankfail.evaluate import window
from bankfail.features import ALL_FEATURES
from bankfail.models import fit_gbm
from bankfail.panel import last_complete_origin
from bankfail.quarters import label, quarter_index

PATH = ("path_unemployment_rise", "path_house_price_change", "path_cre_price_change")
CONDITIONAL_FEATURES = ALL_FEATURES + PATH
SCENARIO_LABELS = {
    "baseline": "Fed 2026 baseline",
    "severely_adverse": "Fed 2026 severely adverse",
}


def path_table(levels: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = levels.set_index("q").sort_index()
    frame = frame.reindex(range(int(frame.index.min()), int(frame.index.max()) + 1))
    unemployment = frame["unemployment"]
    ahead = pd.concat([unemployment.shift(-k) for k in range(1, horizon + 1)], axis=1)
    out = pd.DataFrame(index=frame.index)
    out["path_unemployment_rise"] = ahead.max(axis=1, skipna=False) - unemployment
    for column, source in (
        ("path_house_price_change", "house_prices"),
        ("path_cre_price_change", "cre_prices"),
    ):
        out[column] = (frame[source].shift(-horizon) / frame[source] - 1) * 100
    return out.astype("float32")


def stress_settings(config: Config) -> Config:
    labels = replace(config.labels, horizon_quarters=config.stress.horizon_quarters)
    return replace(config, labels=labels)


def stress_frame(
    panel: pd.DataFrame, historic: pd.DataFrame, known_through: date, horizon: int
) -> pd.DataFrame:
    paths = path_table(historic, horizon)
    frame = panel.drop(columns="event").join(paths, on="q")
    event = frame["quarters_to_event"].between(1, horizon)
    unclear = frame["quarters_to_other"].between(1, horizon) & ~event
    complete = frame["q"] <= last_complete_origin(horizon, known_through)
    label_values = np.where(event, 1.0, np.where(complete & ~unclear, 0.0, np.nan))
    frame["event"] = label_values.astype("float32")
    return frame


def scenario_paths(macro: dict[str, pd.DataFrame], config: Config) -> pd.DataFrame:
    horizon = config.stress.horizon_quarters
    history = macro["historic"].set_index("q").sort_index()
    jump = int(history.index.max())
    start = history.loc[jump]
    rows = []
    for name, text in SCENARIO_LABELS.items():
        path = macro[name].set_index("q").sort_index()
        path = path[path.index > jump].iloc[:horizon]
        if len(path) < horizon:
            raise ValueError(f"the {name} scenario has fewer than {horizon} quarters")
        rows.append(
            {
                "scenario": name,
                "label": text,
                "path_unemployment_rise": path["unemployment"].max() - start["unemployment"],
                "path_house_price_change": (
                    path["house_prices"].iloc[-1] / start["house_prices"] - 1
                )
                * 100,
                "path_cre_price_change": (path["cre_prices"].iloc[-1] / start["cre_prices"] - 1)
                * 100,
            }
        )
    reference = quarter_index(config.stress.reference_origin)
    realized = path_table(macro["historic"], horizon)
    if reference not in realized.index or realized.loc[reference].isna().any():
        raise ValueError("stress.reference_origin has no complete path in the Fed history")
    end = label(reference + horizon)
    rows.append(
        {
            "scenario": "reference",
            "label": f"What happened {label(reference)} to {end}",
            **{column: float(realized.loc[reference, column]) for column in PATH},
        }
    )
    table = pd.DataFrame(rows)
    table.insert(2, "jump_off", label(jump))
    return table


def backtest(labeled: pd.DataFrame, config: Config, log=print) -> pd.DataFrame:
    settings = config.stress
    train = window(labeled, None, quarter_index(settings.backtest_train_last_origin))
    test = window(
        labeled,
        quarter_index(settings.backtest_first_origin),
        quarter_index(settings.backtest_last_origin),
    )
    conditional = fit_gbm(train, CONDITIONAL_FEATURES, config)
    plain = fit_gbm(train, ALL_FEATURES, config)
    table = pd.DataFrame(
        {
            "q": test["q"].to_numpy(),
            "observed": test["event"].to_numpy(),
            "with_path": conditional.predict(test),
            "without_path": plain.predict(test),
        }
    )
    table = (
        table.groupby("q")
        .agg(
            banks=("observed", "size"),
            observed=("observed", "sum"),
            with_path=("with_path", "sum"),
            without_path=("without_path", "sum"),
        )
        .reset_index()
    )
    table["quarter"] = [label(v) for v in table["q"]]
    peak = table.loc[table["observed"].idxmax()]
    log(
        f"stress backtest at {peak['quarter']}: {peak['observed']:.0f} banks failed within "
        f"{settings.horizon_quarters} quarters, {peak['with_path']:.0f} expected with the real "
        f"path, {peak['without_path']:.0f} without it"
    )
    return table.round(2)


def run_stress(
    panel: pd.DataFrame,
    macro: dict[str, pd.DataFrame],
    known_through: date,
    config: Config,
    log=print,
) -> dict[str, pd.DataFrame]:
    horizon = config.stress.horizon_quarters
    settings = stress_settings(config)
    frame = stress_frame(panel, macro["historic"], known_through, horizon)
    labeled = frame[frame["event"].notna()].dropna(subset=list(PATH))
    history = backtest(labeled, settings, log)
    paths = scenario_paths(macro, config)
    model = fit_gbm(labeled, CONDITIONAL_FEATURES, settings)
    latest = int(panel["q"].max())
    banks = frame[frame["q"] == latest].copy()
    sizes = size_bucket(banks["ASSET"])
    summary = []
    by_size = []
    for position, row in paths.iterrows():
        shocked = banks.copy()
        for column in PATH:
            shocked[column] = np.float32(row[column])
        p = model.predict(shocked)
        low, high = simulate_total(p, config.stress.simulations, config.stress.seed + position)
        summary.append(
            {
                **row.to_dict(),
                "as_of": label(latest),
                "horizon_quarters": horizon,
                "banks": int(len(banks)),
                "expected_failures": round(float(p.sum()), 2),
                "interval_low": low,
                "interval_high": high,
                "banks_above_1pct": int((p > 0.01).sum()),
                "banks_above_5pct": int((p > 0.05).sum()),
            }
        )
        grouped = (
            pd.DataFrame({"size": sizes.to_numpy(), "p": p})
            .groupby("size")
            .agg(banks=("p", "size"), expected_failures=("p", "sum"))
            .reindex(SIZE_LABELS)
            .fillna(0)
            .reset_index()
        )
        grouped.insert(0, "scenario", row["scenario"])
        by_size.append(grouped)
        log(
            f"stress {row['scenario']}: {p.sum():.1f} failures expected within {horizon} quarters "
            f"(90% range {low:.0f} to {high:.0f}) among {len(banks):,} banks"
        )
    return {
        "stress_backtest": history,
        "stress_scenarios": pd.DataFrame(summary).round(3),
        "stress_size": pd.concat(by_size, ignore_index=True).round(3),
    }
