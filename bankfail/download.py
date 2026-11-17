import csv
import gzip
import hashlib
import http.client
import io
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path

from bankfail.config import Config, Download
from bankfail.quarters import quarter_ends, ymd

IDENTITY_FIELDS = ("CERT", "REPDTE", "NAME", "STALP", "BKCLASS", "ESTYMD", "MUTUAL", "SUBCHAPS")
STRUCTURE_FIELDS = ("OFFDOM", "RSSDHCR", "SPECGRP", "ASSET", "LIAB", "EQ", "INTAN", "ASSTLT")
FUNDING_FIELDS = ("DEP", "DEPDOM", "DEPINS", "DEPUNINS", "COREDEP", "BRO", "OTHBFHLB", "OTHBOR")
LIQUIDITY_FIELDS = ("VOLIAB", "CHBAL", "SC", "SCHA", "SCHF", "SCAA", "SCAF")
LOAN_FIELDS = ("LNLS", "LNLSNET", "LNATRES", "LNRE", "LNRECONS", "LNRENRES", "LNREMULT")
LOAN_MIX_FIELDS = ("LNRERES", "LNREAG", "LNAG", "LNCI", "LNCON")
QUALITY_FIELDS = ("ORE", "NCLNLS", "NAASSET", "P3ASSET", "P9ASSET", "NTLNLS")
INCOME_FIELDS = ("NETINC", "ELNATR", "INTINC", "EINTEXP", "NONII", "NONIX")
RATIO_FIELDS = ("RBC1AAJ", "RBCRWAJ", "ROA", "NIMY", "EEFFR", "INTINCY", "INTEXPY", "NONIIAY")
QUALITY_RATIO_FIELDS = ("ELNATRY", "NCLNLSR", "NTLNLSR", "LNATRESR", "LNLSDEPR", "NPERFV")

FINANCIAL_FIELDS = (
    IDENTITY_FIELDS
    + STRUCTURE_FIELDS
    + FUNDING_FIELDS
    + LIQUIDITY_FIELDS
    + LOAN_FIELDS
    + LOAN_MIX_FIELDS
    + QUALITY_FIELDS
    + INCOME_FIELDS
    + RATIO_FIELDS
    + QUALITY_RATIO_FIELDS
)

FAILURE_FIELDS = ("ID", "CERT", "NAME", "CITYST", "FAILDATE", "RESTYPE", "RESTYPE1", "SAVR")
FAILURE_SIZE_FIELDS = ("CHCLASS", "QBFASSET", "QBFDEP", "COST")

FED_FILES = {
    "historic": "{year}_Final_Historic_Domestic.csv",
    "baseline": "{year}_Final_Supervisory_Baseline_Domestic.csv",
    "severely_adverse": "{year}_Final_Supervisory_Severely_Adverse_Domestic.csv",
}

FED_COLUMNS = (
    "Scenario Name",
    "Date",
    "Unemployment rate",
    "House Price Index (Level)",
    "Commercial Real Estate Price Index (Level)",
)

USER_AGENT = "fdic-bank-failure-forecast/1.0"
TRANSIENT = (OSError, http.client.HTTPException, ValueError)

Opener = Callable[[urllib.request.Request, float], bytes]


def read_url(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def api_url(base: str, endpoint: str, params: dict) -> str:
    return f"{base}/{endpoint}?{urllib.parse.urlencode(params, safe=':[]*,')}"


def fetch(url: str, settings: Download, opener: Opener = read_url) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = ""
    for attempt in range(1, settings.attempts + 1):
        try:
            payload = opener(request, settings.timeout_seconds)
            if not payload:
                raise ValueError("empty response")
            return payload
        except TRANSIENT as error:
            last = str(error) or error.__class__.__name__
            if attempt < settings.attempts:
                time.sleep(settings.backoff_seconds * attempt)
    raise RuntimeError(f"could not fetch {url} after {settings.attempts} attempts: {last}")


def parse_csv(payload: bytes) -> tuple[list[str], list[list[str]]]:
    text = payload.decode("utf-8-sig")
    if text.lstrip().startswith(("{", "<")):
        raise ValueError("expected csv but got another format")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("csv response has no header")
    return rows[0], [row for row in rows[1:] if row]


def project(header: list[str], rows: list[list[str]], fields: tuple[str, ...]) -> list[list[str]]:
    positions = [header.index(field) if field in header else -1 for field in fields]
    width = len(header)
    out = []
    for row in rows:
        if len(row) != width:
            raise ValueError(f"csv row has {len(row)} values, header has {width}")
        out.append([row[p] if p >= 0 else "" for p in positions])
    return out


def count_records(settings: Download, endpoint: str, filters: str, opener: Opener) -> int:
    params = {"fields": "CERT", "limit": 1}
    if filters:
        params["filters"] = filters
    body = json.loads(fetch(api_url(settings.api_url, endpoint, params), settings, opener))
    total = body.get("meta", {}).get("total")
    if not isinstance(total, int) or total < 0:
        raise ValueError(f"no record count in the {endpoint} response")
    return total


def fetch_table(
    settings: Download,
    endpoint: str,
    filters: str,
    fields: tuple[str, ...],
    sort_by: str,
    opener: Opener,
) -> list[list[str]]:
    total = count_records(settings, endpoint, filters, opener)
    rows: list[list[str]] = []
    offset = 0
    while offset < total:
        params = {
            "fields": ",".join(fields),
            "limit": settings.page_size,
            "offset": offset,
            "sort_by": sort_by,
            "sort_order": "ASC",
            "format": "csv",
        }
        if filters:
            params["filters"] = filters
        header, page = parse_csv(
            fetch(api_url(settings.api_url, endpoint, params), settings, opener)
        )
        if not page:
            raise ValueError(f"{endpoint} returned an empty page at offset {offset}")
        rows.extend(project(header, page, fields))
        offset += settings.page_size
    if len(rows) != total:
        raise ValueError(f"{endpoint} returned {len(rows)} rows, expected {total}")
    return rows


def write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temp, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temp, path)


def quarter_path(raw_dir: Path, quarter: date) -> Path:
    return raw_dir / "financials" / f"{ymd(quarter)}.csv.gz"


def stored_header(path: Path) -> tuple[str, ...]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return tuple(next(csv.reader(handle), []))
    except (OSError, EOFError, ValueError):
        return ()


def download_quarter(settings: Download, raw_dir: Path, quarter: date, opener: Opener) -> int:
    path = quarter_path(raw_dir, quarter)
    if stored_header(path) == FINANCIAL_FIELDS:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return sum(1 for _ in handle) - 1
    last = ""
    for attempt in range(1, settings.attempts + 1):
        try:
            rows = fetch_table(
                settings, "financials", f"REPDTE:{ymd(quarter)}", FINANCIAL_FIELDS, "CERT", opener
            )
            certs = [row[0] for row in rows]
            if len(set(certs)) != len(certs):
                raise ValueError("duplicate banks in one quarter")
            if any(row[1] != ymd(quarter) for row in rows):
                raise ValueError("rows from another quarter came back")
            write_csv(path, FINANCIAL_FIELDS, rows)
            return len(rows)
        except (ValueError, RuntimeError) as error:
            last = str(error)
            if attempt < settings.attempts:
                time.sleep(settings.backoff_seconds * attempt)
    raise RuntimeError(f"quarter {quarter} failed after {settings.attempts} attempts: {last}")


def download_failures(settings: Download, raw_dir: Path, opener: Opener) -> int:
    fields = FAILURE_FIELDS + FAILURE_SIZE_FIELDS
    rows = fetch_table(settings, "failures", "", fields, "FAILDATE", opener)
    ids = [row[0] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("the failures list came back with duplicate records")
    write_csv(raw_dir / "failures.csv", fields, rows)
    return len(rows)


def download_fed(settings: Download, raw_dir: Path, opener: Opener) -> dict[str, str]:
    digests = {}
    for name, pattern in FED_FILES.items():
        filename = pattern.format(year=settings.scenario_year)
        payload = fetch(f"{settings.fed_url}/{filename}", settings, opener)
        header, rows = parse_csv(payload)
        missing = [column for column in FED_COLUMNS if column not in header]
        if missing or not rows:
            raise RuntimeError(f"{filename} is missing columns {missing} or has no rows")
        target = raw_dir / "fed" / f"{name}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def download_all(config: Config, opener: Opener = read_url, log=print) -> dict:
    settings = config.download
    raw_dir = config.paths.raw_dir
    quarters = quarter_ends(settings.first_quarter, settings.last_quarter)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=settings.workers) as pool:
        futures = {pool.submit(download_quarter, settings, raw_dir, q, opener): q for q in quarters}
        for done, future in enumerate(as_completed(futures), start=1):
            quarter = futures[future]
            counts[ymd(quarter)] = future.result()
            log(f"financials {quarter} {counts[ymd(quarter)]} banks ({done}/{len(quarters)})")
    failures = download_failures(settings, raw_dir, opener)
    log(f"failures {failures} rows")
    fed = download_fed(settings, raw_dir, opener)
    log(f"fed scenario files {', '.join(sorted(fed))}")
    manifest = {
        "downloaded_on": datetime.now(UTC).date().isoformat(),
        "quarters": dict(sorted(counts.items())),
        "financial_rows": sum(counts.values()),
        "failure_rows": failures,
        "fed_sha256": fed,
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
