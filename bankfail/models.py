from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from bankfail.config import Config
from bankfail.features import CATEGORICAL, DECREASING, INCREASING


@dataclass
class Prepared:
    columns: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    fill: np.ndarray
    flagged: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(self.columns)].to_numpy(dtype="float64")
        missing = np.isnan(values)
        clipped = np.clip(values, self.lower, self.upper)
        filled = np.where(missing, self.fill, clipped)
        indicators = missing[:, self.flagged].astype("float64")
        stacked = np.hstack([filled, indicators])
        return (stacked - self.mean) / self.scale


def prepare(frame: pd.DataFrame, columns: tuple[str, ...], lower: float, upper: float) -> Prepared:
    values = frame[list(columns)].to_numpy(dtype="float64")
    low = np.nanquantile(values, lower, axis=0)
    high = np.nanquantile(values, upper, axis=0)
    low = np.where(np.isnan(low), 0.0, low)
    high = np.where(np.isnan(high), 0.0, high)
    clipped = np.clip(values, low, high)
    fill = np.nanmedian(clipped, axis=0)
    fill = np.where(np.isnan(fill), 0.0, fill)
    missing = np.isnan(values)
    flagged = missing.mean(axis=0) > 0.001
    stacked = np.hstack([np.where(missing, fill, clipped), missing[:, flagged].astype("float64")])
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return Prepared(tuple(columns), low, high, fill, flagged, mean, scale)


@dataclass
class Logit:
    prepared: Prepared
    model: LogisticRegression

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.prepared.transform(frame))[:, 1]

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = self.prepared.transform(frame)[:, : len(self.prepared.columns)]
        weights = self.model.coef_[0][: len(self.prepared.columns)]
        return pd.DataFrame(x * weights, columns=list(self.prepared.columns), index=frame.index)


def fit_logit(frame: pd.DataFrame, columns: tuple[str, ...], config: Config) -> Logit:
    prepared = prepare(frame, columns, config.features.winsor_lower, config.features.winsor_upper)
    model = LogisticRegression(C=config.logit.c, max_iter=config.logit.max_iter)
    model.fit(prepared.transform(frame), frame["event"].to_numpy(dtype="int8"))
    return Logit(prepared, model)


@dataclass
class Boosted:
    columns: tuple[str, ...]
    model: HistGradientBoostingClassifier

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(self.columns)].to_numpy(dtype="float32")
        return self.model.predict_proba(values)[:, 1]


def monotone_signs(columns: tuple[str, ...]) -> list[int]:
    return [1 if c in INCREASING else -1 if c in DECREASING else 0 for c in columns]


def sample_rows(frame: pd.DataFrame, rate: float, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    y = frame["event"].to_numpy()
    if rate >= 1.0:
        return frame, np.ones(len(frame))
    rng = np.random.default_rng(seed)
    keep = (y == 1) | (rng.random(len(frame)) < rate)
    weights = np.where(y[keep] == 1, 1.0, 1.0 / rate)
    return frame[keep], weights


def observed(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in columns if frame[c].notna().any())


def validation_split(frame: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray] | None:
    settings = config.gbm
    q = frame["q"].to_numpy()
    y = frame["event"].to_numpy()
    last = int(q.max())
    limit = (last - int(q.min())) // 2
    span = settings.validation_quarters
    while span <= limit:
        validation = q > last - span
        fitting = q <= last - span - config.labels.horizon_quarters
        enough = settings.min_validation_failures
        if y[validation].sum() >= enough and y[fitting].sum() >= enough:
            return fitting, validation
        span += 4
    return None


def booster(settings, columns: tuple[str, ...], iterations: int, stopping: bool):
    categorical = [c in CATEGORICAL for c in columns]
    return HistGradientBoostingClassifier(
        learning_rate=settings.learning_rate,
        max_iter=iterations,
        max_leaf_nodes=settings.max_leaf_nodes,
        min_samples_leaf=settings.min_samples_leaf,
        l2_regularization=settings.l2_regularization,
        monotonic_cst=monotone_signs(columns),
        categorical_features=categorical if any(categorical) else None,
        early_stopping=stopping,
        n_iter_no_change=settings.patience,
        scoring="loss",
        random_state=settings.seed,
    )


def best_iterations(sample: pd.DataFrame, weights: np.ndarray, columns, config: Config) -> int:
    settings = config.gbm
    split = validation_split(sample, config)
    if split is None:
        return settings.fallback_iter
    fitting, validation = split
    kept = observed(sample[fitting], columns)
    x = sample[list(kept)].to_numpy(dtype="float32")
    y = sample["event"].to_numpy(dtype="int8")
    model = booster(settings, kept, settings.max_iter, True)
    model.fit(
        x[fitting],
        y[fitting],
        sample_weight=weights[fitting],
        X_val=x[validation],
        y_val=y[validation],
        sample_weight_val=weights[validation],
    )
    losses = -np.asarray(model.validation_score_)
    return max(settings.min_iter, int(np.argmin(losses)))


def fit_gbm(frame: pd.DataFrame, columns: tuple[str, ...], config: Config) -> Boosted:
    settings = config.gbm
    sample, weights = sample_rows(frame, settings.negative_sample_rate, settings.seed)
    kept = observed(sample, columns)
    iterations = best_iterations(sample, weights, kept, config)
    model = booster(settings, kept, iterations, False)
    model.fit(
        sample[list(kept)].to_numpy(dtype="float32"),
        sample["event"].to_numpy(dtype="int8"),
        sample_weight=weights,
    )
    return Boosted(kept, model)


def texas_score(frame: pd.DataFrame) -> np.ndarray:
    return frame["texas_ratio"].fillna(0.0).to_numpy(dtype="float64")
