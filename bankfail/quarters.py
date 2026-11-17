from datetime import date

import numpy as np
import pandas as pd


def parse_date(text: str) -> date:
    return date.fromisoformat(text)


def quarter_index(day: date) -> int:
    return day.year * 4 + (day.month - 1) // 3


def quarter_end(index: int) -> date:
    year, part = divmod(int(index), 4)
    month = 3 * (part + 1)
    last = 31 if month in (3, 12) else 30
    return date(year, month, last)


def is_quarter_end(day: date) -> bool:
    return quarter_end(quarter_index(day)) == day


def quarter_ends(first: date, last: date) -> list[date]:
    return [quarter_end(q) for q in range(quarter_index(first), quarter_index(last) + 1)]


def ymd(day: date) -> str:
    return day.strftime("%Y%m%d")


def label(index: int) -> str:
    year, part = divmod(int(index), 4)
    return f"{year}Q{part + 1}"


def index_from_ymd(values: pd.Series) -> np.ndarray:
    text = values.astype(str).str.slice(0, 8)
    year = pd.to_numeric(text.str.slice(0, 4), errors="coerce")
    month = pd.to_numeric(text.str.slice(4, 6), errors="coerce")
    return (year * 4 + (month - 1) // 3).to_numpy(dtype="float64")


def index_from_label(values: pd.Series) -> np.ndarray:
    text = values.astype(str).str.strip()
    year = pd.to_numeric(text.str.slice(0, 4), errors="coerce")
    part = pd.to_numeric(text.str.slice(-1), errors="coerce")
    return (year * 4 + part - 1).to_numpy(dtype="float64")
