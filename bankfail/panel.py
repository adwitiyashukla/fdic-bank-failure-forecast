import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from bankfail.download import FINANCIAL_FIELDS
from bankfail.quarters import index_from_label, quarter_end, quarter_index

TEXT_FIELDS = ("NAME", "STALP", "BKCLASS", "REPDTE", "ESTYMD")
NUMERIC_FIELDS = tuple(f for f in FINANCIAL_FIELDS if f not in TEXT_FIELDS and f != "CERT")

MACRO_COLUMNS = {
    "Unemployment rate": "unemployment",
    "House Price Index (Level)": "house_prices",
    "Commercial Real Estate Price Index (Level)": "cre_prices",
}


def read_quarter(path: Path) -> pd.DataFrame:
    dtypes = {field: "float32" for field in NUMERIC_FIELDS}
    dtypes.update({field: "string" for field in TEXT_FIELDS})
    dtypes["CERT"] = "int64"
    frame = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[""])
    missing = [field for field in FINANCIAL_FIELDS if field not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    return frame


def load_financials(raw_dir: Path, first: date, last: date) -> pd.DataFrame:
    files = sorted((raw_dir / "financials").glob("*.csv.gz"))
    wanted = [
        path
        for path in files
        if quarter_index(first)
        <= quarter_index(date.fromisoformat(_iso(path)))
        <= quarter_index(last)
    ]
    if not wanted:
        raise FileNotFoundError(f"no quarter files in {raw_dir / 'financials'}, run download first")
    frame = pd.concat([read_quarter(path) for path in wanted], ignore_index=True)
    frame["q"] = _quarters(frame["REPDTE"])
    frame = frame.dropna(subset=["q"])
    frame["q"] = frame["q"].astype("int32")
    frame["CERT"] = frame["CERT"].astype("int32")
    frame["STALP"] = frame["STALP"].fillna("NA").astype("category")
    frame["BKCLASS"] = frame["BKCLASS"].fillna("NA").astype("category")
    frame["NAME"] = frame["NAME"].fillna("").astype("category")
    if frame.duplicated(["CERT", "q"]).any():
        raise ValueError("a bank appears twice in the same quarter")
    return frame.sort_values(["CERT", "q"], ignore_index=True)


def _iso(path: Path) -> str:
    stem = path.name.split(".")[0]
    return f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"


def _quarters(values: pd.Series) -> np.ndarray:
    text = values.astype("string").str.slice(0, 8)
    year = pd.to_numeric(text.str.slice(0, 4), errors="coerce").astype("float64")
    month = pd.to_numeric(text.str.slice(4, 6), errors="coerce").astype("float64")
    return (year * 4 + (month - 1) // 3).to_numpy(dtype="float64", na_value=np.nan)


def load_failures(raw_dir: Path, event_types: tuple[str, ...]) -> pd.DataFrame:
    if not event_types:
        return pd.DataFrame(
            {
                "CERT": pd.Series(dtype="int64"),
                "fail_date": pd.Series(dtype="datetime64[ns]"),
                "fail_q": pd.Series(dtype="int64"),
            }
        )
    path = raw_dir / "failures.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found, run download first")
    frame = pd.read_csv(path, dtype={"NAME": "string", "CITYST": "string", "RESTYPE": "string"})
    frame["fail_date"] = pd.to_datetime(frame["FAILDATE"], format="%m/%d/%Y", errors="coerce")
    frame = frame.dropna(subset=["fail_date", "CERT"])
    frame = frame[frame["RESTYPE"].isin(event_types)].copy()
    frame["CERT"] = frame["CERT"].astype("int64")
    frame["fail_q"] = [quarter_index(day) for day in frame["fail_date"].dt.date]
    columns = ["CERT", "NAME", "CITYST", "fail_date", "fail_q", "RESTYPE", "QBFASSET", "COST"]
    return frame[columns].sort_values(["fail_date", "CERT"], ignore_index=True)


def downloaded_on(raw_dir: Path) -> date:
    path = raw_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found, run download first")
    return date.fromisoformat(json.loads(path.read_text())["downloaded_on"])


def read_fed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in MACRO_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    out = frame.rename(columns=MACRO_COLUMNS)[list(MACRO_COLUMNS.values())].astype("float64")
    out.insert(0, "q", index_from_label(frame["Date"]).astype("int64"))
    return out.sort_values("q", ignore_index=True)


def load_macro(raw_dir: Path) -> dict[str, pd.DataFrame]:
    folder = raw_dir / "fed"
    names = ("historic", "baseline", "severely_adverse")
    missing = [name for name in names if not (folder / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"fed files {missing} not found in {folder}, run download first")
    return {name: read_fed(folder / f"{name}.csv") for name in names}


def last_complete_origin(horizon: int, known_through: date) -> int:
    q = quarter_index(known_through)
    while quarter_end(q + horizon) > known_through:
        q -= 1
    return q


def next_event_gap(panel: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    q = panel["q"].to_numpy(dtype="int64")
    if events.empty:
        return np.full(len(panel), np.nan)
    ends = {int(v): pd.Timestamp(quarter_end(int(v))) for v in np.unique(q)}
    left = pd.DataFrame(
        {
            "row": np.arange(len(panel)),
            "CERT": panel["CERT"].to_numpy(dtype="int64"),
            "report_date": pd.Series(q).map(ends).astype("datetime64[ns]").to_numpy(),
        }
    ).sort_values("report_date", kind="stable")
    right = pd.DataFrame(
        {
            "CERT": events["CERT"].to_numpy(dtype="int64"),
            "fail_date": events["fail_date"].astype("datetime64[ns]").to_numpy(),
            "fail_q": events["fail_q"].to_numpy(dtype="float64"),
        }
    ).sort_values("fail_date", kind="stable")
    matched = pd.merge_asof(
        left,
        right,
        left_on="report_date",
        right_on="fail_date",
        by="CERT",
        direction="forward",
        allow_exact_matches=False,
    ).sort_values("row")
    return matched["fail_q"].to_numpy(dtype="float64") - q


def attach_labels(
    panel: pd.DataFrame,
    failures: pd.DataFrame,
    horizon: int,
    known_through: date,
    ambiguous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    q = panel["q"].to_numpy(dtype="int64")
    ahead = next_event_gap(panel, failures)
    event = (ahead >= 1) & (ahead <= horizon)
    complete = q <= last_complete_origin(horizon, known_through)
    label = np.where(event, 1.0, np.where(complete, 0.0, np.nan))
    other = np.full(len(panel), np.nan)
    if ambiguous is not None and not ambiguous.empty:
        other = next_event_gap(panel, ambiguous)
        unclear = (other >= 1) & (other <= horizon) & ~event
        label = np.where(unclear, np.nan, label)
    out = panel.copy()
    out["quarters_to_event"] = ahead.astype("float32")
    out["quarters_to_other"] = other.astype("float32")
    out["event"] = label.astype("float32")
    return out


def matched_failures(panel: pd.DataFrame, failures: pd.DataFrame, lookback: int) -> pd.DataFrame:
    last_report = panel.groupby("CERT", observed=True)["q"].max()
    out = failures.copy()
    out["last_q"] = out["CERT"].map(last_report)
    gap = out["fail_q"] - out["last_q"]
    out["matched"] = gap.between(0, lookback)
    return out
