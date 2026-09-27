"""EDGAR 8-K Item 2.02 parser, tested on recorded filing shapes (no network).

The fixtures mirror real cases seen in EDGAR: an intraday pre-announcement the day before the
release (AAPL, Jan 2005), a second Item 2.02 filed with a late 10-Q (MSFT, Nov 2004) and an
issuer whose reporting lag drifted over the years (AAPL, ~18 days in 2006 vs ~31 later).
"""

import pandas as pd
import pytest

from quantlab.config import Settings
from quantlab.data.providers.base import ProviderNotConfigured
from quantlab.data.providers.edgar import EdgarProvider, classify_timing
from quantlab.data.validation import validate_generic


def _page(rows):
    cols = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "items"]
    return {c: [r[i] for r in rows] for i, c in enumerate(cols)}


def _et(s):
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _events(rows):
    return EdgarProvider.parse_earnings(EdgarProvider.parse_filings([_page(rows)]), "TEST", 42)


def _quarter(n, pe, rel_et, q_filed, form="10-Q"):
    """A clean quarter: one Item 2.02 release and its periodic report."""
    return [(f"r{n}", rel_et[:10], rel_et[:10], _et(rel_et), "8-K", "2.02,9.01"),
            (f"q{n}", q_filed, pe, _et(q_filed + " 17:00"), form, "")]


def test_timing_classification_uses_the_real_calendar():
    ts = lambda s: pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")
    assert classify_timing(ts("2024-10-31 07:00")) == "BMO"
    assert classify_timing(ts("2024-10-31 12:00")) == "DURING"
    assert classify_timing(ts("2024-10-31 16:30")) == "AMC"
    assert classify_timing(ts("2024-11-29 13:30")) == "AMC"  # early close at 13:00
    assert classify_timing(ts("2024-11-02 10:00")) == "BMO"  # Saturday → before next open


def test_pre_announcement_is_dropped_and_release_kept():
    rows = (_quarter(0, "2004-09-25", "2004-10-13 16:31", "2004-12-03", "10-K")
            + [("pre", "2005-01-11", "2005-01-11", _et("2005-01-11 13:39"), "8-K", "2.02")]
            + _quarter(1, "2004-12-25", "2005-01-12 16:26", "2005-02-02")
            + _quarter(2, "2005-03-26", "2005-04-13 16:28", "2005-05-04"))
    ev, counts = _events(rows)
    q = ev.set_index("fiscal_period")
    assert q.loc["2004-12-25", "accession"] == "r1"
    assert counts["unmatched"] == 1 and counts["matched"] == 3
    assert (ev["timing"] == "AMC").all()
    assert validate_generic(ev, "earnings_events").ok


def test_second_item_202_with_late_10q_is_not_the_release():
    rows = (_quarter(0, "2004-06-30", "2004-07-22 16:10", "2004-08-27", "10-K")
            + [("r1", "2004-10-21", "2004-10-21", _et("2004-10-21 16:37"), "8-K", "2.02"),
               ("late", "2004-11-08", "2004-11-08", _et("2004-11-08 17:11"), "8-K", "2.02"),
               ("q1", "2004-11-08", "2004-09-30", _et("2004-11-08 19:47"), "10-Q", "")]
            + _quarter(2, "2004-12-31", "2005-01-27 16:12", "2005-02-07"))
    ev, _ = _events(rows)
    assert ev.set_index("fiscal_period").loc["2004-09-30", "accession"] == "r1"


def _regular_quarters(first_pe, n, lag, tag):
    """n clean quarters with a fixed reporting lag (days after period end), 10-Q two days later."""
    out = []
    for i, pe in enumerate(pd.date_range(first_pe, periods=n, freq="QE")):
        rel = pe + pd.Timedelta(days=lag)
        out += _quarter(f"{tag}{i}", pe.strftime("%Y-%m-%d"), rel.strftime("%Y-%m-%d") + " 16:30",
                        (rel + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    return out


def test_usual_lag_is_local_so_drifting_issuers_match():
    # lag ~18 days around 2006, ~31 days in recent years (more recent quarters, as in real histories)
    rows = (_regular_quarters("2005-03-31", 5, 18, "e") + _regular_quarters("2006-12-31", 2, 18, "f")
            + _regular_quarters("2016-03-31", 11, 31, "l") + [
        # 2006 Q2: the release (lag 18), then a restatement-related 2.02 (lag 33); 10-Q filed months late
        ("jul19", "2006-07-19", "2006-07-19", _et("2006-07-19 16:29"), "8-K", "2.02,9.01"),
        ("aug03", "2006-08-03", "2006-08-03", _et("2006-08-03 17:52"), "8-K", "2.02,4.02,9.01"),
        ("q06", "2006-12-29", "2006-07-01", _et("2006-12-29 16:00"), "10-Q", ""),
        # 2018 Q4: revenue warning (lag 4) before the release (lag 31)
        ("warn", "2019-01-02", "2019-01-02", _et("2019-01-02 16:30"), "8-K", "2.02,9.01"),
        ("jan31", "2019-01-31", "2019-01-31", _et("2019-01-31 16:30"), "8-K", "2.02,9.01"),
        ("q18", "2019-02-01", "2018-12-31", _et("2019-02-01 17:00"), "10-Q", ""),
    ])
    q = _events(rows)[0].set_index("fiscal_period")
    assert q.loc["2006-07-01", "accession"] == "jul19"
    assert q.loc["2018-12-31", "accession"] == "jan31"


def test_amendments_and_pre_2004_filings_are_ignored():
    rows = (_quarter(0, "2004-09-30", "2004-10-20 07:25", "2004-11-05")
            + [("amend", "2004-10-21", "2004-10-21", _et("2004-10-21 07:00"), "8-K/A", "2.02"),
               ("old", "2004-04-15", "2004-04-15", _et("2004-04-15 07:00"), "8-K", "2.02"),
               ("q-1", "2004-05-07", "2004-03-31", _et("2004-05-07 17:00"), "10-Q", "")])
    ev, counts = _events(rows)
    assert ev["accession"].tolist() == ["r0"] and counts["item_202_filings"] == 1
    assert ev["timing"].iloc[0] == "BMO"


def test_parse_filings_merges_pages_and_ticker_map_normalises_classes():
    rows = _quarter(0, "2024-09-30", "2024-10-30 16:05", "2024-10-31")
    df = EdgarProvider.parse_filings([_page(rows), _page(rows[:1])])
    assert len(df) == 2 and df["acceptance"].dt.tz is not None
    m = EdgarProvider.parse_ticker_map({"0": {"cik_str": 1067983, "ticker": "BRK.B", "title": "x"}})
    assert m == {"BRK-B": 1067983}


def test_missing_user_agent_raises_not_configured(monkeypatch):
    import quantlab.data.providers.edgar as edgar
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    monkeypatch.setattr(edgar, "get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(ProviderNotConfigured):
        EdgarProvider().cik_map()
