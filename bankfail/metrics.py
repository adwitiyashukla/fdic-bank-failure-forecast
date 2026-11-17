import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def brier(y: np.ndarray, probability: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(probability, 0.0, 1.0)))


def flag_top(q: np.ndarray, score: np.ndarray, fraction: float) -> np.ndarray:
    frame = pd.DataFrame({"q": q, "score": score})
    rank = frame.groupby("q")["score"].rank(method="first", ascending=False)
    size = frame.groupby("q")["score"].transform("size")
    cutoff = np.ceil(size.to_numpy() * fraction)
    return rank.to_numpy() <= cutoff


def capture(
    q: np.ndarray, y: np.ndarray, score: np.ndarray, fraction: float
) -> tuple[float, float]:
    flagged = flag_top(q, score, fraction)
    hits = float(np.sum(flagged & (y == 1)))
    positives = float(np.sum(y == 1))
    recall = hits / positives if positives else float("nan")
    precision = hits / float(np.sum(flagged)) if np.sum(flagged) else float("nan")
    return recall, precision


def cluster_bootstrap_auc(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, reps: int, seed: int
) -> tuple[float, float]:
    if reps == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    codes, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(codes) + 1))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        picks = rng.integers(0, len(codes), len(codes))
        rows = np.concatenate([order[bounds[p] : bounds[p + 1]] for p in picks])
        if len(np.unique(y[rows])) == 2:
            values.append(roc_auc_score(y[rows], score[rows]))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def warning_quarters(
    frame: pd.DataFrame, score: np.ndarray, fraction: float, lookback: int, horizon: int
) -> pd.DataFrame:
    flagged = flag_top(frame["q"].to_numpy(), score, fraction)
    data = pd.DataFrame(
        {
            "CERT": frame["CERT"].to_numpy(),
            "q": frame["q"].to_numpy(),
            "ahead": frame["quarters_to_event"].to_numpy(),
            "flag": flagged,
        }
    )
    data = data[(data["ahead"] >= 1) & (data["ahead"] <= lookback)]
    rows = []
    for cert, group in data.groupby("CERT", sort=True):
        group = group.sort_values("q")
        if group["ahead"].iloc[-1] > horizon:
            continue
        fail_q = int(group["q"].iloc[-1] + group["ahead"].iloc[-1])
        streak_start = None
        for q, flag in zip(
            group["q"].to_numpy()[::-1], group["flag"].to_numpy()[::-1], strict=True
        ):
            if not flag:
                break
            streak_start = q
        continuous = 0 if streak_start is None else fail_q - int(streak_start)
        first = group.loc[group["flag"], "q"]
        earliest = 0 if first.empty else fail_q - int(first.min())
        rows.append(
            {
                "CERT": int(cert),
                "fail_q": fail_q,
                "last_report_ahead": int(group["ahead"].iloc[-1]),
                "flagged_at_last_report": bool(group["flag"].iloc[-1]),
                "warning_quarters": int(continuous),
                "first_flag_quarters": int(earliest),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y: np.ndarray, probability: np.ndarray, bins: int) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y, "p": probability})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    table = frame.groupby("bin").agg(
        rows=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )
    return table.reset_index()


def rounded(value: float, digits: int = 6) -> float:
    return value if math.isnan(value) else round(value, digits)
