import pytest

from bankfail.config import load_config


def edited(tmp_path, config_path, old, new):
    text = config_path.read_text()
    assert old in text
    path = tmp_path / "config.toml"
    path.write_text(text.replace(old, new))
    return path


def test_shipped_config_loads(config_path):
    config = load_config(config_path)
    assert config.labels.horizon_quarters == 4
    assert config.labels.event_types == ("FAILURE",)
    assert [design.name for design in config.evaluation.designs] == ["crisis", "recent"]
    assert config.gbm.min_iter <= config.gbm.fallback_iter <= config.gbm.max_iter


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("horizon_quarters = 4", 'horizon_quarters = "4"', "wrong type"),
        ('first_quarter = "1984-03-31"', 'first_quarter = "1984-03-30"', "quarter end"),
        ('first_quarter = "1984-03-31"', 'first_quarter = "1983-12-31"', "before 1984"),
        ('test_first_origin = "2006-12-31"', 'test_first_origin = "2006-06-30"', "labels"),
        ('test_last_origin = "2025-06-30"', 'test_last_origin = "2025-09-30"', "not known yet"),
        ('ambiguous_types = ["ASSISTANCE"]', 'ambiguous_types = ["FAILURE"]', "overlap"),
        ('event_types = ["FAILURE"]', 'event_types = ["CLOSED"]', "subset"),
        ('api_url = "https://api.fdic.gov/banks"', 'api_url = "http://x.gov/banks"', "https"),
        ("winsor_lower = 0.005", "winsor_lower = 0.999", "below"),
        ("min_iter = 50", "min_iter = 500", "fallback_iter"),
        ("rolling_first_year = 1990", "rolling_first_year = 1985", "two earlier years"),
        ("rolling_sample_rate = 0.25", "rolling_sample_rate = 1.5", "at most 1"),
        ("negative_sample_rate = 1.0", "negative_sample_rate = 0", "above 0"),
        ("threads = 4", "threads = 0", "gbm.threads must be between"),
        ('space_id = "adwitiyashukla/fdic-bank-failure-forecast"', 'space_id = "demo"', "owner"),
        ("certs = [24735, 57053, 59017]", "certs = [24735, true]", "positive integers"),
        ("simulations = 20000", "simulations = 5", "between"),
    ],
)
def test_bad_values_are_rejected(tmp_path, config_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        load_config(edited(tmp_path, config_path, old, new))


def test_missing_section_is_named(tmp_path, config_path):
    path = edited(tmp_path, config_path, "[stress]", "[stresss]")
    with pytest.raises(ValueError, match=r"missing the \[stress\] section"):
        load_config(path)


def test_missing_key_is_named(tmp_path, config_path):
    path = edited(tmp_path, config_path, "page_size = 10000\n", "")
    with pytest.raises(ValueError, match="download.page_size"):
        load_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
