import csv
import gzip
import json
from dataclasses import replace

import pytest

from bankfail import download
from bankfail.config import load_config
from tests.fakeapi import FakeApi

FED_HEADER = (
    "Scenario Name,Date,Real GDP growth,Unemployment rate,"
    "House Price Index (Level),Commercial Real Estate Price Index (Level)\n"
)


def fed_files(year):
    body = FED_HEADER + "Actual,2025 Q4,1.0,4.5,323.4,305.8\n"
    return {pattern.format(year=year): body.encode() for pattern in download.FED_FILES.values()}


def bank(cert, date, name="BANK"):
    return {
        "CERT": cert,
        "REPDTE": date,
        "NAME": name,
        "ASSET": 1000 + cert,
        "ID": f"{cert}_{date}",
    }


@pytest.fixture
def config(tmp_path, config_path):
    base = load_config(config_path)
    settings = replace(
        base.download,
        api_url="https://api.example/banks",
        fed_url="https://fed.example/files",
        first_quarter=base.download.first_quarter.replace(year=2024, month=3, day=31),
        last_quarter=base.download.last_quarter.replace(year=2024, month=6, day=30),
        page_size=100,
        workers=2,
        backoff_seconds=0.001,
    )
    return replace(base, download=settings, paths=replace(base.paths, raw_dir=tmp_path / "raw"))


def make_api(config, banks_per_quarter=250):
    rows = []
    for date in ("20240331", "20240630"):
        rows.extend(
            bank(cert, date, name=f'BANK "{cert}", NA') for cert in range(1, banks_per_quarter + 1)
        )
    failures = [
        {
            "ID": str(i),
            "CERT": 1000 + i,
            "NAME": f"FAILED {i}",
            "FAILDATE": "1/17/2025",
            "RESTYPE": "FAILURE",
            "COST": 10.5,
        }
        for i in range(1, 6)
    ]
    return FakeApi(rows, failures, fed_files(config.download.scenario_year))


def test_download_all_pages_through_every_quarter(config):
    api = make_api(config)
    manifest = download.download_all(config, opener=api, log=lambda message: None)
    assert manifest["quarters"] == {"20240331": 250, "20240630": 250}
    assert manifest["failure_rows"] == 5
    path = config.paths.raw_dir / "financials" / "20240630.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == download.FINANCIAL_FIELDS
    assert [int(row[0]) for row in rows[1:]] == list(range(1, 251))
    assert rows[5][2] == 'BANK "5", NA'
    saved = json.loads((config.paths.raw_dir / "manifest.json").read_text())
    assert saved["financial_rows"] == 500


def test_existing_quarter_files_are_not_fetched_again(config):
    api = make_api(config)
    download.download_all(config, opener=api, log=lambda message: None)
    first = api.calls
    download.download_all(config, opener=api, log=lambda message: None)
    assert api.calls - first == 1 + 1 + len(download.FED_FILES)


def test_transient_errors_are_retried(config):
    api = make_api(config)
    api.fail_first = 2
    rows = download.fetch_table(
        config.download, "financials", "REPDTE:20240331", ("CERT",), "CERT", api
    )
    assert len(rows) == 250


def test_fetch_gives_up_with_a_readable_error(config):
    def broken(request, timeout):
        raise OSError("network down")

    with pytest.raises(RuntimeError, match="after 5 attempts: network down"):
        download.fetch("https://api.example/banks/failures", config.download, broken)


def test_short_download_is_rejected(config):
    api = make_api(config)
    original = api._rows

    def shrinking(endpoint, filters):
        rows = original(endpoint, filters)
        return rows if api.calls % 2 else rows[:-1]

    api._rows = shrinking
    with pytest.raises(RuntimeError, match="failed after 5 attempts"):
        download.download_quarter(
            config.download, config.paths.raw_dir, config.download.first_quarter, api
        )
    assert not list((config.paths.raw_dir / "financials").glob("*.gz"))


def test_json_error_page_is_not_saved_as_csv():
    with pytest.raises(ValueError, match="expected csv"):
        download.parse_csv(b'{"error": "bad request"}')


def test_ragged_rows_are_rejected():
    with pytest.raises(ValueError, match="values"):
        download.project(["CERT", "NAME"], [["1"]], ("CERT",))


def test_missing_fed_columns_fail(config):
    api = make_api(config)
    name = download.FED_FILES["historic"].format(year=config.download.scenario_year)
    api.fed_files[name] = b"Scenario Name,Date\nActual,2025 Q4\n"
    with pytest.raises(RuntimeError, match="missing columns"):
        download.download_fed(config.download, config.paths.raw_dir, api)


def test_api_url_keeps_filter_syntax_readable():
    url = download.api_url(
        "https://x/banks", "financials", {"filters": "CERT:[1 TO *]", "limit": 5}
    )
    assert "filters=CERT:[1+TO+*]" in url
