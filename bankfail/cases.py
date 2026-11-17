import numpy as np
import pandas as pd

from bankfail.config import Config
from bankfail.evaluate import PROBABILITY_MODELS, fit_all, score_all, window
from bankfail.features import CREDIT_FEATURES, RUN_SCREEN
from bankfail.models import fit_gbm
from bankfail.quarters import label, quarter_end, quarter_index

SIZE_EDGES = (0.0, 1e5, 1e6, 1e7, 1e8, np.inf)
SIZE_LABELS = ("under $100M", "$100M to $1B", "$1B to $10B", "$10B to $100B", "over $100B")
THRESHOLDS = (0.005, 0.01, 0.05)


def size_bucket(assets: pd.Series) -> pd.Series:
    return pd.cut(assets, SIZE_EDGES, labels=SIZE_LABELS, right=False).astype("string")


def failure_info(failures: pd.DataFrame) -> pd.DataFrame:
    info = failures.drop_duplicates("CERT", keep="last").set_index("CERT")
    return pd.DataFrame(
        {
            "name": info["NAME"],
            "place": info["CITYST"],
            "fail_date": info["fail_date"].dt.date.astype("string"),
        }
    )


def rank_within(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("q")[column].rank(ascending=False, method="min")


def run_screen(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = [frame[c].rank(pct=True) for c in RUN_SCREEN]
    score = pieces[0] * pieces[1]
    return pd.DataFrame(
        {"run_score": score, "run_rank": score.rank(ascending=False, method="min")},
        index=frame.index,
    )


def score_origin(panel: pd.DataFrame, labeled: pd.DataFrame, origin: int, config: Config):
    train = window(labeled, None, origin - config.labels.horizon_quarters)
    banks = panel[panel["q"] == origin].copy()
    fitted = fit_all(train, config)
    scores = score_all(banks, fitted)
    scores["gbm_credit"] = fit_gbm(train, CREDIT_FEATURES, config).predict(banks)
    for column in scores.columns:
        banks[column] = scores[column]
        banks[f"{column}_rank"] = banks[column].rank(ascending=False, method="min")
    return banks.join(run_screen(banks))


def run_case(
    panel: pd.DataFrame, labeled: pd.DataFrame, failures: pd.DataFrame, config: Config, log=print
) -> dict[str, pd.DataFrame]:
    info = failure_info(failures)
    focus_rows = []
    screen = None
    for origin_day in config.case2023.score_origins:
        origin = quarter_index(origin_day)
        banks = score_origin(panel, labeled, origin, config)
        soon = banks["quarters_to_event"].between(1, 8)
        focus = banks[banks["CERT"].isin(config.case2023.certs) | soon].copy()
        focus["as_of"] = label(origin)
        focus["banks_ranked"] = len(banks)
        focus_rows.append(focus)
        screen = banks
        for cert in config.case2023.certs:
            hit = banks[banks["CERT"] == cert]
            if not hit.empty:
                row = hit.iloc[0]
                name = info["name"].get(cert, cert)
                log(
                    f"{label(origin)} {name}: boosting rank {int(row['gbm_rank'])} of "
                    f"{len(banks)}, run screen rank {int(row['run_rank'])}"
                )
    columns = ["as_of", "CERT", "banks_ranked", "ASSET", "quarters_to_event"]
    columns += ["equity_ratio", "noncurrent_loans", "roa", "deposit_growth"]
    columns += ["uninsured_deposits", "unrealized_loss_to_equity", "gbm"]
    columns += ["texas_rank", "logit_rank", "gbm_rank", "gbm_credit_rank", "run_rank"]
    case = pd.concat(focus_rows, ignore_index=True)[columns]
    case = case.join(info, on="CERT")
    case = case.sort_values(["as_of", "gbm_rank"], ignore_index=True)
    scatter = pd.DataFrame(
        {
            "CERT": screen["CERT"].to_numpy(),
            "uninsured_deposits": screen["uninsured_deposits"].to_numpy(),
            "unrealized_loss_to_equity": screen["unrealized_loss_to_equity"].to_numpy(),
            "size": size_bucket(screen["ASSET"]).to_numpy(),
            "failed_within_2y": screen["quarters_to_event"].between(1, 8).to_numpy(),
            "run_rank": screen["run_rank"].to_numpy(),
            "gbm_rank": screen["gbm_rank"].to_numpy(),
        }
    )
    return {"case": case.round(4), "screen": scatter.round(3)}


def simulate_total(probabilities: np.ndarray, simulations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    totals = []
    chunk = max(1, min(simulations, 2000))
    done = 0
    while done < simulations:
        size = min(chunk, simulations - done)
        draws = rng.random((size, len(probabilities))) < probabilities
        totals.append(draws.sum(axis=1))
        done += size
    values = np.concatenate(totals)
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def run_current(
    recent: pd.DataFrame, labeled: pd.DataFrame, failures: pd.DataFrame, config: Config, log=print
) -> dict[str, pd.DataFrame]:
    horizon = config.labels.horizon_quarters
    start = int(labeled["q"].max()) + 1
    latest = int(recent["q"].max())
    info = failure_info(failures)
    recent = recent[recent["q"] >= start].copy()
    live_models = fit_all(window(labeled, None, start - horizon), config)
    live_scores = score_all(recent, live_models)
    for column in live_scores.columns:
        recent[column] = live_scores[column]
    recent["gbm_rank"] = rank_within(recent, "gbm")
    recent["texas_rank"] = rank_within(recent, "texas")
    recent["banks_ranked"] = recent.groupby("q")["q"].transform("size")
    failed = recent[recent["quarters_to_event"] >= 1].copy()
    failed["as_of"] = [label(v) for v in failed["q"]]
    live = failed[
        ["as_of", "CERT", "banks_ranked", "quarters_to_event", "gbm", "gbm_rank", "texas_rank"]
    ].join(info, on="CERT")
    log(f"live check: {failed['CERT'].nunique()} banks failed after reports from {label(start)} on")

    models = fit_all(labeled, config)
    now = recent[recent["q"] == latest].copy()
    scores = score_all(now, models)
    now["probability"] = scores["gbm"].to_numpy()
    now["logit_probability"] = scores["logit"].to_numpy()
    now["texas_ratio"] = scores["texas"].to_numpy()
    now["size"] = size_bucket(now["ASSET"])
    p = now["probability"].to_numpy()
    low, high = simulate_total(p, config.stress.simulations, config.stress.seed)
    summary = {
        "as_of": label(latest),
        "as_of_date": quarter_end(latest).isoformat(),
        "banks": int(len(now)),
        "expected_failures": round(float(p.sum()), 2),
        "interval_low": low,
        "interval_high": high,
        "logit_expected_failures": round(float(now["logit_probability"].sum()), 2),
        "mean_probability": round(float(p.mean()), 6),
        "median_probability": round(float(np.median(p)), 6),
    }
    for threshold in THRESHOLDS:
        summary[f"banks_above_{threshold:g}"] = int((p > threshold).sum())
    by_size = (
        now.groupby("size", observed=True)
        .agg(
            banks=("probability", "size"),
            expected_failures=("probability", "sum"),
            above_1pct=("probability", lambda v: int((v > 0.01).sum())),
            assets_billions=("ASSET", lambda v: float(v.sum()) / 1e6),
        )
        .reindex(SIZE_LABELS)
        .fillna(0)
        .reset_index()
    )
    edges = np.array([0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1.0000001])
    names = ["under 0.001%", "0.001% to 0.01%", "0.01% to 0.1%", "0.1% to 0.5%"]
    names += ["0.5% to 1%", "1% to 5%", "5% or more"]
    counts = np.histogram(np.clip(p, 0, 1), bins=edges)[0]
    histogram = pd.DataFrame({"band": names, "banks": counts})
    listing = now[
        ["CERT", "NAME", "STALP", "ASSET", "probability", "logit_probability", "texas_ratio"]
    ].sort_values("probability", ascending=False)
    log(
        f"{label(latest)}: {len(now):,} banks, expected failures in the next year "
        f"{summary['expected_failures']} (90% range {low:.0f} to {high:.0f})"
    )
    return {
        "live": live.round(6),
        "current_summary": pd.DataFrame([summary]),
        "current_size": by_size.round(3),
        "current_histogram": histogram,
        "current_scores": listing,
    }


def failed_paths(scored: pd.DataFrame, failures: pd.DataFrame, lookback: int) -> pd.DataFrame:
    data = scored.copy()
    for model in ("texas",) + PROBABILITY_MODELS:
        data[f"{model}_percentile"] = data.groupby("q")[model].rank(pct=True)
    near = data[data["quarters_to_event"].between(1, lookback)]
    info = failure_info(failures)
    out = near[
        [
            "CERT",
            "q",
            "quarters_to_event",
            "gbm",
            "gbm_percentile",
            "logit_percentile",
            "texas_percentile",
        ]
    ].join(info, on="CERT")
    out["quarter"] = [label(v) for v in out["q"]]
    return out.drop(columns="q").round(5)
