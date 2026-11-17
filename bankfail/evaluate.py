import numpy as np
import pandas as pd

from bankfail.config import Config, Design
from bankfail.features import (
    ALL_FEATURES,
    ASSET_QUALITY,
    CAPITAL,
    CATEGORICAL,
    CONCENTRATION,
    EARNINGS,
    FUNDING,
    GROWTH,
    LOCAL,
    LOGIT_FEATURES,
    STRUCTURE,
)
from bankfail.metrics import (
    auc,
    average_precision,
    brier,
    calibration_table,
    capture,
    cluster_bootstrap_auc,
    rounded,
    warning_quarters,
)
from bankfail.models import fit_gbm, fit_logit, texas_score
from bankfail.quarters import label, quarter_index

MODELS = ("texas", "logit", "gbm")
PROBABILITY_MODELS = ("logit", "gbm")
GROUPS = {
    "capital": CAPITAL,
    "asset quality": ASSET_QUALITY,
    "earnings": EARNINGS,
    "loan mix": CONCENTRATION,
    "growth": GROWTH,
    "size and age": STRUCTURE + CATEGORICAL,
    "funding and liquidity": FUNDING,
    "local failures": LOCAL,
}
CURVE = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)


def labeled_rows(panel: pd.DataFrame, last_complete: int) -> pd.DataFrame:
    keep = panel["event"].notna() & (panel["q"] <= last_complete)
    return panel[keep]


def window(frame: pd.DataFrame, first: int | None, last: int) -> pd.DataFrame:
    keep = frame["q"] <= last
    if first is not None:
        keep &= frame["q"] >= first
    return frame[keep]


def fit_all(train: pd.DataFrame, config: Config) -> dict:
    return {
        "logit": fit_logit(train, LOGIT_FEATURES, config),
        "gbm": fit_gbm(train, ALL_FEATURES, config),
    }


def score_all(frame: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    scores = pd.DataFrame(index=frame.index)
    scores["texas"] = texas_score(frame)
    for name in PROBABILITY_MODELS:
        scores[name] = fitted[name].predict(frame)
    return scores


def summarize(name: str, frame: pd.DataFrame, scores: pd.DataFrame, config: Config) -> list[dict]:
    y = frame["event"].to_numpy(dtype="int8")
    q = frame["q"].to_numpy()
    certs = frame["CERT"].to_numpy()
    settings = config.evaluation
    rows = []
    for model in MODELS:
        s = scores[model].to_numpy(dtype="float64")
        low, high = cluster_bootstrap_auc(y, s, certs, settings.bootstrap_reps, settings.seed)
        warn = warning_quarters(
            frame,
            s,
            settings.flag_fraction,
            settings.warning_lookback,
            config.labels.horizon_quarters,
        )
        row = {
            "design": name,
            "model": model,
            "rows": int(len(frame)),
            "positive_rows": int(y.sum()),
            "failed_banks": int(frame.loc[frame["event"] == 1, "CERT"].nunique()),
            "auc": rounded(auc(y, s)),
            "auc_low": rounded(low),
            "auc_high": rounded(high),
            "average_precision": rounded(average_precision(y, s)),
            "brier": rounded(brier(y, s), 6) if model in PROBABILITY_MODELS else float("nan"),
            "predicted_events": rounded(float(s.sum()), 1)
            if model in PROBABILITY_MODELS
            else float("nan"),
            "observed_events": int(y.sum()),
        }
        for fraction in settings.top_fractions:
            recall, precision = capture(q, y, s, fraction)
            row[f"recall_top_{fraction:g}"] = rounded(recall)
            row[f"precision_top_{fraction:g}"] = rounded(precision)
        if not warn.empty:
            row["banks_flagged_last_report"] = rounded(float(warn["flagged_at_last_report"].mean()))
            row["median_warning_quarters"] = float(warn["warning_quarters"].median())
            row["share_warned_4q"] = rounded(float((warn["warning_quarters"] >= 4).mean()))
        rows.append(row)
    return rows


def capture_curve(name: str, frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    y = frame["event"].to_numpy(dtype="int8")
    q = frame["q"].to_numpy()
    rows = []
    for model in MODELS:
        s = scores[model].to_numpy(dtype="float64")
        for fraction in CURVE:
            recall, precision = capture(q, y, s, fraction)
            rows.append(
                {
                    "design": name,
                    "model": model,
                    "fraction": fraction,
                    "recall": rounded(recall),
                    "precision": rounded(precision),
                }
            )
    return pd.DataFrame(rows)


def quarterly_counts(name: str, frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame({"q": frame["q"].to_numpy(), "observed": frame["event"].to_numpy()})
    for model in PROBABILITY_MODELS:
        data[model] = scores[model].to_numpy()
    table = data.groupby("q").agg(
        banks=("observed", "size"),
        observed=("observed", "sum"),
        logit=("logit", "sum"),
        gbm=("gbm", "sum"),
    )
    table = table.reset_index()
    table.insert(0, "design", name)
    table["quarter"] = [label(v) for v in table["q"]]
    return table.round(2)


def group_importance(
    frame: pd.DataFrame, model, reps: int, seed: int, sample_negatives: int = 60000
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positives = frame[frame["event"] == 1]
    negatives = frame[frame["event"] == 0]
    if len(negatives) > sample_negatives:
        negatives = negatives.sample(sample_negatives, random_state=seed)
    sample = pd.concat([positives, negatives])
    y = sample["event"].to_numpy(dtype="int8")
    base = auc(y, model.predict(sample))
    rows = []
    for group, columns in GROUPS.items():
        used = [c for c in columns if c in model.columns]
        drops = []
        for _ in range(max(reps, 1)):
            shuffled = sample.copy()
            order = rng.permutation(len(sample))
            shuffled[used] = sample[used].to_numpy()[order]
            drops.append(base - auc(y, model.predict(shuffled)))
        rows.append(
            {"group": group, "features": len(used), "auc_drop": rounded(float(np.mean(drops)))}
        )
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False, ignore_index=True)


def named_warnings(
    name: str, frame: pd.DataFrame, scores: pd.DataFrame, failures: pd.DataFrame, config: Config
) -> pd.DataFrame:
    s = scores["gbm"].to_numpy(dtype="float64")
    settings = config.evaluation
    warn = warning_quarters(
        frame, s, settings.flag_fraction, settings.warning_lookback, config.labels.horizon_quarters
    )
    if warn.empty:
        return warn
    info = failures.drop_duplicates("CERT", keep="last").set_index("CERT")
    warn.insert(0, "design", name)
    warn["name"] = warn["CERT"].map(info["NAME"])
    warn["place"] = warn["CERT"].map(info["CITYST"])
    warn["fail_date"] = warn["CERT"].map(info["fail_date"]).dt.date.astype("string")
    warn["assets_thousands"] = warn["CERT"].map(info["QBFASSET"])
    return warn


def run_design(
    labeled: pd.DataFrame, design: Design, failures: pd.DataFrame, config: Config, log=print
) -> dict[str, pd.DataFrame]:
    train = window(labeled, None, quarter_index(design.train_last_origin))
    test = window(
        labeled, quarter_index(design.test_first_origin), quarter_index(design.test_last_origin)
    )
    log(
        f"design {design.name}: training on {len(train):,} rows with {int(train['event'].sum()):,} "
        f"failures, testing on {len(test):,} rows"
    )
    fitted = fit_all(train, config)
    scores = score_all(test, fitted)
    calibration = []
    for model in PROBABILITY_MODELS:
        table = calibration_table(test["event"].to_numpy(), scores[model].to_numpy(), 10)
        table.insert(0, "model", model)
        table.insert(0, "design", design.name)
        calibration.append(table)
    importance = group_importance(test, fitted["gbm"], 3, config.evaluation.seed)
    importance.insert(0, "design", design.name)
    return {
        "metrics": pd.DataFrame(summarize(design.name, test, scores, config)),
        "curve": capture_curve(design.name, test, scores),
        "quarterly": quarterly_counts(design.name, test, scores),
        "calibration": pd.concat(calibration, ignore_index=True).round(6),
        "importance": importance,
        "warnings": named_warnings(design.name, test, scores, failures, config),
    }


def run_rolling(
    labeled: pd.DataFrame, config: Config, log=print
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config.evaluation
    horizon = config.labels.horizon_quarters
    last = int(labeled["q"].max())
    metrics = []
    scored = []
    for year in range(settings.rolling_first_year, settings.rolling_last_year + 1):
        first = year * 4
        train = window(labeled, None, first - horizon)
        test = window(labeled, first, min(first + 3, last))
        if test.empty or train["event"].sum() == 0:
            continue
        fitted = fit_all(train, config)
        scores = score_all(test, fitted)
        y = test["event"].to_numpy(dtype="int8")
        q = test["q"].to_numpy()
        for model in MODELS:
            s = scores[model].to_numpy(dtype="float64")
            recall, _ = capture(q, y, s, settings.flag_fraction)
            metrics.append(
                {
                    "year": year,
                    "model": model,
                    "rows": int(len(test)),
                    "positive_rows": int(y.sum()),
                    "failed_banks": int(test.loc[test["event"] == 1, "CERT"].nunique()),
                    "auc": rounded(auc(y, s)),
                    "recall_flagged": rounded(recall),
                    "predicted_events": rounded(float(s.sum()), 1)
                    if model in PROBABILITY_MODELS
                    else float("nan"),
                }
            )
        keep = test[["CERT", "q", "event", "quarters_to_event"]].copy()
        scored.append(pd.concat([keep, scores], axis=1))
        auc_gbm = next(m["auc"] for m in reversed(metrics) if m["model"] == "gbm")
        log(f"rolling {year}: {int(y.sum())} failure rows, boosting AUC {auc_gbm}")
    return pd.DataFrame(metrics), pd.concat(scored, ignore_index=True)
