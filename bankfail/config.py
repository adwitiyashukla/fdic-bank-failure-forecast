import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bankfail.quarters import is_quarter_end, quarter_index

EVENT_TYPES = ("FAILURE", "ASSISTANCE")


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    work_dir: Path
    results_dir: Path
    demo_dir: Path


@dataclass(frozen=True)
class Download:
    api_url: str
    fed_url: str
    first_quarter: date
    last_quarter: date
    scenario_year: int
    page_size: int
    workers: int
    attempts: int
    backoff_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class Labels:
    horizon_quarters: int
    event_types: tuple[str, ...]
    ambiguous_types: tuple[str, ...]


@dataclass(frozen=True)
class Panel:
    first_origin: date
    min_assets: float


@dataclass(frozen=True)
class Features:
    winsor_lower: float
    winsor_upper: float


@dataclass(frozen=True)
class Logit:
    c: float
    max_iter: int


@dataclass(frozen=True)
class Gbm:
    learning_rate: float
    max_iter: int
    min_iter: int
    fallback_iter: int
    patience: int
    validation_quarters: int
    min_validation_failures: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    negative_sample_rate: float
    seed: int
    threads: int


@dataclass(frozen=True)
class Design:
    name: str
    train_last_origin: date
    test_first_origin: date
    test_last_origin: date


@dataclass(frozen=True)
class Evaluation:
    top_fractions: tuple[float, ...]
    flag_fraction: float
    warning_lookback: int
    bootstrap_reps: int
    rolling_first_year: int
    rolling_last_year: int
    rolling_sample_rate: float
    seed: int
    designs: tuple[Design, ...]


@dataclass(frozen=True)
class Case2023:
    score_origins: tuple[date, ...]
    certs: tuple[int, ...]


@dataclass(frozen=True)
class Stress:
    horizon_quarters: int
    backtest_train_last_origin: date
    backtest_first_origin: date
    backtest_last_origin: date
    reference_origin: date
    simulations: int
    seed: int


@dataclass(frozen=True)
class Config:
    paths: Paths
    download: Download
    labels: Labels
    panel: Panel
    features: Features
    logit: Logit
    gbm: Gbm
    evaluation: Evaluation
    case2023: Case2023
    stress: Stress


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config is missing the [{name}] section")
    return value


def _get(section: dict, key: str, where: str, kinds: tuple[type, ...]):
    if key not in section:
        raise ValueError(f"config is missing {where}.{key}")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, kinds):
        raise ValueError(f"config value {where}.{key} has the wrong type")
    return value


def _int(section: dict, key: str, where: str, low: int, high: int | None = None) -> int:
    value = int(_get(section, key, where, (int,)))
    if value < low or (high is not None and value > high):
        bound = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise ValueError(f"config value {where}.{key} must be {bound}")
    return value


def _float(section: dict, key: str, where: str) -> float:
    value = float(_get(section, key, where, (int, float)))
    if value <= 0:
        raise ValueError(f"config value {where}.{key} must be greater than zero")
    return value


def _fraction(section: dict, key: str, where: str) -> float:
    value = float(_get(section, key, where, (int, float)))
    if not 0.0 < value < 1.0:
        raise ValueError(f"config value {where}.{key} must be between zero and one")
    return value


def _text(section: dict, key: str, where: str) -> str:
    value = _get(section, key, where, (str,)).strip()
    if not value:
        raise ValueError(f"config value {where}.{key} must not be empty")
    return value


def _url(section: dict, key: str, where: str) -> str:
    value = _text(section, key, where).rstrip("/")
    if not value.startswith("https://"):
        raise ValueError(f"config value {where}.{key} must be an https url")
    return value


def _quarter(section: dict, key: str, where: str) -> date:
    text = _text(section, key, where)
    try:
        value = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"config value {where}.{key} must be an ISO date") from error
    if not is_quarter_end(value):
        raise ValueError(f"config value {where}.{key} must be a quarter end date")
    return value


def _paths(raw: dict) -> Paths:
    section = _section(raw, "paths")
    return Paths(
        **{
            key: Path(_text(section, key, "paths"))
            for key in ("raw_dir", "work_dir", "results_dir", "demo_dir")
        }
    )


def _download(raw: dict) -> Download:
    s = _section(raw, "download")
    w = "download"
    value = Download(
        api_url=_url(s, "api_url", w),
        fed_url=_url(s, "fed_url", w),
        first_quarter=_quarter(s, "first_quarter", w),
        last_quarter=_quarter(s, "last_quarter", w),
        scenario_year=_int(s, "scenario_year", w, 2013, 2100),
        page_size=_int(s, "page_size", w, 100, 10000),
        workers=_int(s, "workers", w, 1, 16),
        attempts=_int(s, "attempts", w, 1, 20),
        backoff_seconds=_float(s, "backoff_seconds", w),
        timeout_seconds=_float(s, "timeout_seconds", w),
    )
    if value.first_quarter >= value.last_quarter:
        raise ValueError("download.first_quarter must come before download.last_quarter")
    if value.first_quarter.year < 1984:
        raise ValueError(
            "download.first_quarter cannot be before 1984, the API has no earlier data"
        )
    return value


def _types(section: dict, key: str, allow_empty: bool) -> tuple[str, ...]:
    kinds = _get(section, key, "labels", (list,))
    if (not kinds and not allow_empty) or any(kind not in EVENT_TYPES for kind in kinds):
        raise ValueError(f"labels.{key} must be a subset of {list(EVENT_TYPES)}")
    return tuple(dict.fromkeys(kinds))


def _labels(raw: dict) -> Labels:
    s = _section(raw, "labels")
    events = _types(s, "event_types", allow_empty=False)
    ambiguous = _types(s, "ambiguous_types", allow_empty=True)
    if set(events) & set(ambiguous):
        raise ValueError("labels.event_types and labels.ambiguous_types must not overlap")
    return Labels(
        horizon_quarters=_int(s, "horizon_quarters", "labels", 1, 12),
        event_types=events,
        ambiguous_types=ambiguous,
    )


def _panel(raw: dict) -> Panel:
    s = _section(raw, "panel")
    min_assets = float(_get(s, "min_assets", "panel", (int, float)))
    if min_assets < 0:
        raise ValueError("config value panel.min_assets cannot be negative")
    return Panel(first_origin=_quarter(s, "first_origin", "panel"), min_assets=min_assets)


def _features(raw: dict) -> Features:
    s = _section(raw, "features")
    lower = _fraction(s, "winsor_lower", "features")
    upper = _fraction(s, "winsor_upper", "features")
    if lower >= upper:
        raise ValueError("features.winsor_lower must be below features.winsor_upper")
    return Features(winsor_lower=lower, winsor_upper=upper)


def _rate(section: dict, key: str, where: str) -> float:
    value = float(_get(section, key, where, (int, float)))
    if not 0.0 < value <= 1.0:
        raise ValueError(f"config value {where}.{key} must be above 0 and at most 1")
    return value


def _gbm(raw: dict) -> Gbm:
    s = _section(raw, "gbm")
    rate = _rate(s, "negative_sample_rate", "gbm")
    max_iter = _int(s, "max_iter", "gbm", 1, 5000)
    min_iter = _int(s, "min_iter", "gbm", 1, 5000)
    fallback_iter = _int(s, "fallback_iter", "gbm", 1, 5000)
    if not min_iter <= fallback_iter <= max_iter:
        raise ValueError("config needs gbm.min_iter <= gbm.fallback_iter <= gbm.max_iter")
    return Gbm(
        learning_rate=_fraction(s, "learning_rate", "gbm"),
        max_iter=max_iter,
        min_iter=min_iter,
        fallback_iter=fallback_iter,
        patience=_int(s, "patience", "gbm", 1, 1000),
        validation_quarters=_int(s, "validation_quarters", "gbm", 4, 200),
        min_validation_failures=_int(s, "min_validation_failures", "gbm", 1),
        max_leaf_nodes=_int(s, "max_leaf_nodes", "gbm", 2, 1024),
        min_samples_leaf=_int(s, "min_samples_leaf", "gbm", 1),
        l2_regularization=float(_get(s, "l2_regularization", "gbm", (int, float))),
        negative_sample_rate=rate,
        seed=_int(s, "seed", "gbm", 0),
        threads=_int(s, "threads", "gbm", 1, 256),
    )


def _design(item: dict, horizon: int, last: date) -> Design:
    w = "evaluation.designs"
    design = Design(
        name=_text(item, "name", w),
        train_last_origin=_quarter(item, "train_last_origin", w),
        test_first_origin=_quarter(item, "test_first_origin", w),
        test_last_origin=_quarter(item, "test_last_origin", w),
    )
    train_end = quarter_index(design.train_last_origin) + horizon
    if quarter_index(design.test_first_origin) < train_end:
        raise ValueError(
            f"design {design.name} tests before its training labels are known, "
            "move test_first_origin later"
        )
    if design.test_last_origin < design.test_first_origin:
        raise ValueError(f"design {design.name} has test_last_origin before test_first_origin")
    if quarter_index(design.test_last_origin) + horizon > quarter_index(last):
        raise ValueError(f"design {design.name} tests origins whose outcome is not known yet")
    return design


def _evaluation(raw: dict, horizon: int, download: Download, panel: Panel) -> Evaluation:
    s = _section(raw, "evaluation")
    w = "evaluation"
    fractions = _get(s, "top_fractions", w, (list,))
    if not fractions or any(
        isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 < f < 1 for f in fractions
    ):
        raise ValueError("evaluation.top_fractions must be a list of numbers between 0 and 1")
    items = _get(s, "designs", w, (list,))
    if not items or not all(isinstance(item, dict) for item in items):
        raise ValueError("evaluation.designs must be a list of tables")
    designs = tuple(_design(item, horizon, download.last_quarter) for item in items)
    if len({d.name for d in designs}) != len(designs):
        raise ValueError("evaluation.designs names must be unique")
    first_year = _int(s, "rolling_first_year", w, 1980)
    last_year = _int(s, "rolling_last_year", w, first_year)
    if first_year - 2 < panel.first_origin.year:
        raise ValueError("evaluation.rolling_first_year needs two earlier years of panel data")
    if (last_year - 1) * 4 + 3 + horizon > quarter_index(download.last_quarter):
        raise ValueError("evaluation.rolling_last_year has no complete outcomes yet")
    return Evaluation(
        top_fractions=tuple(sorted(float(f) for f in fractions)),
        flag_fraction=_fraction(s, "flag_fraction", w),
        warning_lookback=_int(s, "warning_lookback", w, 1, 40),
        bootstrap_reps=_int(s, "bootstrap_reps", w, 0, 10000),
        rolling_first_year=first_year,
        rolling_last_year=last_year,
        rolling_sample_rate=_rate(s, "rolling_sample_rate", w),
        seed=_int(s, "seed", w, 0),
        designs=designs,
    )


def _case2023(raw: dict, panel: Panel, horizon: int) -> Case2023:
    s = _section(raw, "case2023")
    w = "case2023"
    origins = _get(s, "score_origins", w, (list,))
    if not origins:
        raise ValueError("case2023.score_origins must not be empty")
    scored = tuple(_quarter({"d": o}, "d", "case2023.score_origins") for o in origins)
    if min(quarter_index(o) for o in scored) - horizon <= quarter_index(panel.first_origin):
        raise ValueError("case2023.score_origins leave no training data before them")
    certs = _get(s, "certs", w, (list,))
    if not certs or any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in certs):
        raise ValueError("case2023.certs must be a list of positive integers")
    return Case2023(score_origins=scored, certs=tuple(certs))


def _stress(raw: dict, last: date) -> Stress:
    s = _section(raw, "stress")
    w = "stress"
    value = Stress(
        horizon_quarters=_int(s, "horizon_quarters", w, 1, 12),
        backtest_train_last_origin=_quarter(s, "backtest_train_last_origin", w),
        backtest_first_origin=_quarter(s, "backtest_first_origin", w),
        backtest_last_origin=_quarter(s, "backtest_last_origin", w),
        reference_origin=_quarter(s, "reference_origin", w),
        simulations=_int(s, "simulations", w, 100, 1000000),
        seed=_int(s, "seed", w, 0),
    )
    horizon = value.horizon_quarters
    if quarter_index(value.backtest_first_origin) < (
        quarter_index(value.backtest_train_last_origin) + horizon
    ):
        raise ValueError("stress.backtest_first_origin comes before the training labels are known")
    if value.backtest_last_origin < value.backtest_first_origin:
        raise ValueError("stress.backtest_last_origin comes before stress.backtest_first_origin")
    if quarter_index(value.backtest_last_origin) + horizon > quarter_index(last):
        raise ValueError("stress.backtest_last_origin has no complete outcomes yet")
    return value


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found at {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    download = _download(raw)
    labels = _labels(raw)
    panel = _panel(raw)
    if panel.first_origin <= download.first_quarter:
        raise ValueError("panel.first_origin must come after download.first_quarter")
    logit = _section(raw, "logit")
    return Config(
        paths=_paths(raw),
        download=download,
        labels=labels,
        panel=panel,
        features=_features(raw),
        logit=Logit(c=_float(logit, "c", "logit"), max_iter=_int(logit, "max_iter", "logit", 10)),
        gbm=_gbm(raw),
        evaluation=_evaluation(raw, labels.horizon_quarters, download, panel),
        case2023=_case2023(raw, panel, labels.horizon_quarters),
        stress=_stress(raw, download.last_quarter),
    )
