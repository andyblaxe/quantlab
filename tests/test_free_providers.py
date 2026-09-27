"""Parsers for free providers, tested against recorded response shapes (no network)."""

import pandas as pd
import pytest

from quantlab.config import Settings
from quantlab.data.providers.base import ProviderNotConfigured
from quantlab.data.providers.free import CboeIndexProvider, FredProvider, StooqProvider, TiingoProvider
from quantlab.data.validation import validate_actions, validate_bars
from quantlab.provenance import DataFlag

STOOQ = """Date,Open,High,Low,Close,Volume
2024-11-27,600.46,600.85,597.45,598.83,34000000
2024-11-28,1,1,1,1,1
2024-11-29,599.66,603.35,599.38,602.55,30100000
"""


def test_stooq_parse_drops_non_sessions_and_timestamps_close():
    df = StooqProvider.parse(STOOQ, "SPY")
    assert list(df["session"].dt.strftime("%Y-%m-%d")) == ["2024-11-27", "2024-11-29"]  # Thanksgiving row dropped
    assert df["available_at"].iloc[1] == pd.Timestamp("2024-11-29 18:15", tz="UTC")  # early close
    assert validate_bars(df).ok
    assert DataFlag.VENDOR_ADJUSTED in StooqProvider.info.flags


def test_stooq_rejects_error_pages():
    with pytest.raises(ValueError):
        StooqProvider.parse("No data", "XXX")


def test_tiingo_parse_extracts_actions():
    recs = [
        {"date": "2020-08-28T00:00:00.000Z", "open": 500.0, "high": 505.0, "low": 495.0, "close": 499.23,
         "volume": 1e6, "divCash": 0.0, "splitFactor": 1.0},
        {"date": "2020-08-31T00:00:00.000Z", "open": 127.58, "high": 131.0, "low": 126.0, "close": 129.04,
         "volume": 4e6, "divCash": 0.0, "splitFactor": 4.0},
        {"date": "2020-09-01T00:00:00.000Z", "open": 132.76, "high": 134.8, "low": 130.53, "close": 134.18,
         "volume": 1.5e6, "divCash": 0.2, "splitFactor": 1.0},
    ]
    bars, acts = TiingoProvider.parse(recs, "AAPL")
    assert len(bars) == 3 and validate_bars(bars, acts).ok
    assert acts["action"].tolist() == ["split", "dividend"] and acts["value"].tolist() == [4.0, 0.2]
    assert validate_actions(acts).ok


def test_fred_parse_vintages_and_missing_markers():
    payload = {"observations": [
        {"realtime_start": "2024-04-25", "realtime_end": "2024-05-29", "date": "2024-01-01", "value": "1.6"},
        {"realtime_start": "2024-05-30", "realtime_end": "9999-12-31", "date": "2024-01-01", "value": "1.3"},
        {"realtime_start": "2024-07-25", "realtime_end": "9999-12-31", "date": "2024-04-01", "value": "."},
    ]}
    v = FredProvider.parse(payload, "GDPC1", vintages=True)
    assert len(v) == 2  # "." is missing, not zero
    assert v["available_at"].iloc[0] > pd.Timestamp("2024-04-25 12:00", tz="UTC")
    nv = FredProvider.parse(payload, "GDPC1", vintages=False)
    assert (nv["available_at"] > nv["observation_date"].dt.tz_localize("UTC")).all()


def test_cboe_parse():
    text = "DATE,OPEN,HIGH,LOW,CLOSE\n11/27/2024,14.1,14.5,13.9,14.10\n11/29/2024,14.0,14.2,13.4,13.51\n"
    df = CboeIndexProvider.parse(text, "VIX")
    assert df["value"].tolist() == [14.10, 13.51]
    assert df["available_at"].iloc[1] == pd.Timestamp("2024-11-29 18:30", tz="UTC")


def test_missing_credentials_raise_not_configured(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    from quantlab import config
    config.get_settings.cache_clear()
    monkeypatch.setattr(config, "get_settings", lambda: Settings(_env_file=None))
    import quantlab.data.providers.free as free
    monkeypatch.setattr(free, "get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(ProviderNotConfigured):
        FredProvider().get_macro_series("DGS10", "2020-01-01", "2020-12-31")


def test_blank_env_values_mean_unset(monkeypatch):
    monkeypatch.setenv("NORGATE_EXPORT_DIR", "")
    monkeypatch.setenv("FRED_API_KEY", " ")
    s = Settings(_env_file=None)
    assert s.norgate_export_dir is None and s.fred_api_key is None
    assert s.live_trading_enabled is False
