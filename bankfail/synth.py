import csv
import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from bankfail.config import Config
from bankfail.download import FAILURE_FIELDS, FAILURE_SIZE_FIELDS, FINANCIAL_FIELDS, write_csv
from bankfail.panel import MACRO_COLUMNS
from bankfail.quarters import label, quarter_end, quarter_ends, quarter_index, ymd

STATES = ("TX", "CA", "IL", "GA", "FL", "NY", "OH", "MN", "KS", "CO", "PA", "WI")
CHARTER_CODES = ("N", "SM", "NM", "SB", "SA")
FED_EXTRA = ("Real GDP growth", "10-year Treasury yield", "Market Volatility Index (Level)")


def crisis_level(years: np.ndarray) -> np.ndarray:
    first = np.exp(-0.5 * ((years - 1988.5) / 2.2) ** 2)
    second = np.exp(-0.5 * ((years - 2009.8) / 1.6) ** 2)
    return 1.3 * first + 1.1 * second


def macro_levels(first_q: int, last_q: int, stress: np.ndarray) -> dict[str, np.ndarray]:
    q = np.arange(first_q, last_q + 1)
    years = q / 4.0
    unemployment = 4.8 + 4.5 * stress + 0.3 * np.sin(q / 5.0)
    growth = 0.012 - 0.03 * stress
    house = 60.0 * np.exp(np.cumsum(growth))
    cre = 55.0 * np.exp(np.cumsum(growth * 1.2 - 0.002))
    return {
        "q": q,
        "years": years,
        "Unemployment rate": unemployment,
        "House Price Index (Level)": house,
        "Commercial Real Estate Price Index (Level)": cre,
    }


def fed_rows(name: str, levels: dict, rows: range) -> list[list[str]]:
    out = []
    for i in rows:
        record = [name, label(int(levels["q"][i])).replace("Q", " Q")]
        record += [f"{levels[column][i]:.2f}" for column in MACRO_COLUMNS]
        record += ["1.5", "5.0", "20.0"]
        out.append(record)
    return out


def write_fed(raw_dir: Path, config: Config) -> None:
    year = config.download.scenario_year
    jump = year * 4 - 1
    first = quarter_index(quarter_end(1976 * 4))
    years = np.arange(first, jump + 14) / 4.0
    calm = crisis_level(years)
    severe = calm.copy()
    ahead = np.arange(len(years)) > jump - first
    severe[ahead] = calm[ahead] + 1.4 * np.exp(-0.5 * ((years[ahead] - year - 1.2) / 0.8) ** 2)
    header = ["Scenario Name", "Date", *MACRO_COLUMNS, *FED_EXTRA]
    folder = raw_dir / "fed"
    folder.mkdir(parents=True, exist_ok=True)
    base = macro_levels(first, jump + 13, calm)
    stressed = macro_levels(first, jump + 13, severe)
    history = range(0, jump - first + 1)
    future = range(jump - first + 1, jump - first + 14)
    files = {
        "historic": fed_rows("Actual", base, history),
        "baseline": fed_rows("Supervisory Baseline", base, future),
        "severely_adverse": fed_rows("Supervisory Severely Adverse", stressed, future),
    }
    for name, rows in files.items():
        with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)


def number(value: float) -> str:
    return "" if np.isnan(value) else f"{value:.4f}"


def bank_rows(quarter, state: dict, rng: np.random.Generator) -> list[list[str]]:
    n = len(state["cert"])
    h = state["health"]
    assets = state["assets"]
    loans = assets * np.clip(0.62 - 0.03 * h + rng.normal(0, 0.03, n), 0.2, 0.9)
    equity = assets * np.clip(0.095 + 0.022 * h + rng.normal(0, 0.006, n), -0.05, 0.3)
    noncurrent = loans * np.clip(np.exp(-4.3 - 0.9 * h + rng.normal(0, 0.25, n)), 0, 0.4)
    oreo = assets * np.clip(np.exp(-6.0 - 0.9 * h + rng.normal(0, 0.3, n)), 0, 0.1)
    reserve = loans * np.clip(0.012 + 0.2 * noncurrent / np.maximum(loans, 1), 0.004, 0.08)
    deposits = assets * np.clip(0.84 + 0.02 * h, 0.5, 0.95)
    roa = 1.0 + 0.7 * h + rng.normal(0, 0.25, n)
    year = quarter.year
    values = {
        "ASSET": assets,
        "LIAB": assets - equity,
        "EQ": equity,
        "INTAN": assets * 0.004,
        "ASSTLT": assets * np.clip(0.25 - 0.02 * h, 0.05, 0.6),
        "DEP": deposits,
        "DEPDOM": deposits,
        "DEPINS": deposits * 0.7,
        "DEPUNINS": deposits * np.clip(0.3 + rng.normal(0, 0.08, n), 0.02, 0.95),
        "COREDEP": deposits * np.clip(0.8 + 0.03 * h, 0.3, 0.99),
        "BRO": deposits * np.clip(0.03 - 0.015 * h, 0, 0.4),
        "OTHBFHLB": assets * 0.03,
        "OTHBOR": assets * np.clip(0.05 - 0.01 * h, 0, 0.3),
        "VOLIAB": assets * np.clip(0.15 - 0.02 * h, 0.01, 0.6),
        "CHBAL": assets * np.clip(0.06 + 0.01 * h, 0.005, 0.4),
        "SC": assets * 0.22,
        "SCHA": assets * 0.05,
        "SCHF": assets * 0.05 * (1 - np.clip(rng.normal(0.02, 0.03, n), -0.05, 0.3)),
        "SCAA": assets * 0.17,
        "SCAF": assets * 0.17 * (1 - np.clip(rng.normal(0.01, 0.02, n), -0.05, 0.3)),
        "LNLS": loans,
        "LNLSNET": loans - reserve,
        "LNATRES": reserve,
        "LNRE": loans * 0.6,
        "LNRECONS": loans * np.clip(0.08 - 0.03 * h, 0, 0.5),
        "LNRENRES": loans * 0.2,
        "LNREMULT": loans * 0.04,
        "LNRERES": loans * 0.25,
        "LNREAG": loans * 0.03,
        "LNAG": loans * 0.05,
        "LNCI": loans * 0.18,
        "LNCON": loans * 0.07,
        "ORE": oreo,
        "NCLNLS": noncurrent,
        "NAASSET": noncurrent * 0.7,
        "P3ASSET": assets * np.clip(np.exp(-4.8 - 0.6 * h), 0, 0.1),
        "P9ASSET": noncurrent * 0.3,
        "NTLNLS": loans * np.clip(0.004 - 0.004 * h, 0, 0.1),
        "NETINC": assets * roa / 100,
        "ELNATR": assets * np.clip(0.003 - 0.004 * h, 0, 0.05),
        "INTINC": assets * 0.05,
        "EINTEXP": assets * 0.02,
        "NONII": assets * 0.008,
        "NONIX": assets * 0.028,
        "RBC1AAJ": equity / assets * 100 + 0.3,
        "RBCRWAJ": equity / assets * 140,
        "ROA": roa,
        "NIMY": 3.8 + 0.3 * h + rng.normal(0, 0.3, n),
        "EEFFR": 62 - 8 * h + rng.normal(0, 5, n),
        "INTINCY": 6.0 - 0.3 * h,
        "INTEXPY": 2.4 - 0.3 * h,
        "NONIIAY": 0.8 + rng.normal(0, 0.2, n),
        "ELNATRY": np.clip(0.3 - 0.4 * h, 0, 5),
        "NCLNLSR": noncurrent / loans * 100,
        "NTLNLSR": np.clip(0.4 - 0.4 * h, 0, 10),
        "LNATRESR": reserve / loans * 100,
        "LNLSDEPR": loans / deposits * 100,
        "NPERFV": (noncurrent + oreo) / assets * 100,
    }
    if year < 2001:
        values["P3ASSET"] = np.full(n, np.nan)
    if year < 1994:
        for field in ("SCHA", "SCHF", "SCAA", "SCAF"):
            values[field] = np.full(n, np.nan)
    rows = []
    for i in range(n):
        record = []
        for field in FINANCIAL_FIELDS:
            if field == "CERT":
                record.append(str(state["cert"][i]))
            elif field == "REPDTE":
                record.append(ymd(quarter))
            elif field == "NAME":
                record.append(f"SYNTHETIC BANK {state['cert'][i]}")
            elif field == "STALP":
                record.append(state["state"][i])
            elif field == "BKCLASS":
                record.append(state["charter"][i])
            elif field == "ESTYMD":
                record.append(state["founded"][i])
            elif field in ("MUTUAL", "SUBCHAPS"):
                record.append(str(int(state["flags"][i])))
            elif field == "OFFDOM":
                record.append(str(int(state["offices"][i])))
            elif field == "RSSDHCR":
                record.append(str(int(state["holding"][i])))
            elif field == "SPECGRP":
                record.append("4")
            else:
                record.append(number(float(values[field][i])))
        rows.append(record)
    return rows


def write_synthetic(config: Config, banks: int, seed: int = 3, log=print) -> dict:
    rng = np.random.default_rng(seed)
    raw = config.paths.raw_dir
    manifest_path = raw / "manifest.json"
    if manifest_path.is_file() and not json.loads(manifest_path.read_text()).get("synthetic"):
        raise RuntimeError(f"{raw} holds downloaded FDIC data, refusing to overwrite it")
    quarters = quarter_ends(config.download.first_quarter, config.download.last_quarter)
    total = banks + banks // 3
    state = {
        "cert": np.arange(10001, 10001 + total),
        "state": rng.choice(STATES, total),
        "charter": rng.choice(CHARTER_CODES, total),
        "founded": [f"{int(y)}0101" for y in rng.integers(1890, 1984, total)],
        "flags": rng.random(total) < 0.1,
        "offices": rng.integers(1, 40, total),
        "holding": np.where(rng.random(total) < 0.7, rng.integers(1000, 9999, total), 0),
        "assets": np.exp(rng.normal(11.5, 1.3, total)),
        "health": rng.normal(0.0, 0.8, total),
        "mean": rng.normal(0.0, 0.6, total),
        "beta": rng.gamma(2.0, 0.5, total),
        "start": np.where(np.arange(total) < banks, 0, rng.integers(1, len(quarters) - 8, total)),
    }
    alive = np.ones(total, dtype=bool)
    failures = []
    counts = {}
    for position, quarter in enumerate(quarters):
        stress = crisis_level(np.array([quarter_index(quarter) / 4.0]))[0]
        h = state["health"]
        state["health"] = (
            0.85 * h
            + 0.15 * state["mean"]
            - 0.35 * state["beta"] * stress
            + rng.normal(0, 0.25, total)
        )
        state["assets"] = state["assets"] * np.exp(0.01 + 0.01 * state["health"] / 4)
        active = alive & (state["start"] <= position)
        merged = active & (rng.random(total) < 0.004)
        alive &= ~merged
        active &= ~merged
        index = np.flatnonzero(active)
        view = {
            key: (value[index] if isinstance(value, np.ndarray) else [value[i] for i in index])
            for key, value in state.items()
        }
        rows = bank_rows(quarter, view, rng)
        write_csv(raw / "financials" / f"{ymd(quarter)}.csv.gz", FINANCIAL_FIELDS, rows)
        counts[ymd(quarter)] = len(rows)
        hazard = 1 / (1 + np.exp(-(-8.2 - 2.2 * state["health"][index])))
        failing = index[rng.random(len(index)) < hazard]
        for i in failing:
            alive[i] = False
            day = quarter + timedelta(days=int(rng.integers(20, 170)))
            kind = "ASSISTANCE" if rng.random() < 0.05 else "FAILURE"
            failures.append(
                [
                    str(len(failures) + 1),
                    str(state["cert"][i]),
                    f"SYNTHETIC BANK {state['cert'][i]}",
                    f"TOWN, {state['state'][i]}",
                    f"{day.month}/{day.day}/{day.year}",
                    kind,
                    "PA",
                    "DIF",
                    state["charter"][i],
                    f"{state['assets'][i]:.0f}",
                    f"{state['assets'][i] * 0.8:.0f}",
                    f"{state['assets'][i] * 0.1:.1f}",
                ]
            )
    write_csv(raw / "failures.csv", FAILURE_FIELDS + FAILURE_SIZE_FIELDS, failures)
    write_fed(raw, config)
    known = quarter_end(quarter_index(config.download.last_quarter)) + timedelta(days=70)
    manifest = {
        "downloaded_on": known.isoformat(),
        "quarters": counts,
        "financial_rows": sum(counts.values()),
        "failure_rows": len(failures),
        "synthetic": True,
    }
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"synthetic data: {manifest['financial_rows']:,} bank quarters, {len(failures):,} failures")
    return manifest
