"""Security master: permanent identity with dated tickers (reuse, renames) and lifecycle."""

import pandas as pd
import pytest

from quantlab.data.security_master import SecurityMaster, UnknownIdentifier

D = pd.Timestamp
UTC = lambda s: pd.Timestamp(s, tz="UTC")


def _master(extra_ids=()):
    sec = pd.DataFrame([
        ("T:DELL_OLD", "stock", D("1988-08-17"), D("2013-10-29"), "acquisition", 13.88),
        ("T:DELL_NEW", "stock", D("2018-12-28"), pd.NaT, None, None),
        ("T:BKNG", "stock", D("1999-03-31"), pd.NaT, None, None),
        ("T:LEH", "stock", D("1994-05-02"), D("2008-09-17"), "bankruptcy", None),
    ], columns=["security_id", "asset_type", "first_session", "last_session", "delisting_reason", "delisting_value"])
    sec["available_at"], sec["source"] = UTC("2026-01-01"), "test"
    ids = pd.DataFrame([
        ("T:DELL_OLD", "ticker", "DELL", D("1988-08-17"), D("2013-10-29")),
        ("T:DELL_NEW", "ticker", "DELL", D("2018-12-28"), pd.NaT),
        ("T:DELL_NEW", "ticker", "DVMT", D("2016-09-07"), D("2018-12-27")),
        ("T:BKNG", "ticker", "PCLN", D("1999-03-31"), D("2018-02-26")),
        ("T:BKNG", "ticker", "BKNG", D("2018-02-27"), pd.NaT),
        ("T:BKNG", "name", "Priceline Group Inc", D("1999-03-31"), D("2018-02-26")),
        ("T:BKNG", "name", "Booking Holdings Inc", D("2018-02-27"), pd.NaT),
        ("T:LEH", "ticker", "LEH", D("1994-05-02"), D("2008-09-17")),
        *extra_ids,
    ], columns=["security_id", "id_type", "value", "valid_from", "valid_to"])
    ids["available_at"], ids["source"] = UTC("2026-01-01"), "test"
    ev = pd.DataFrame([("T:LEH", "bankruptcy", D("2008-09-15")), ("T:LEH", "halt", D("2008-09-16")),
                       ("T:DELL_OLD", "acquisition", D("2013-10-29"))],
                      columns=["security_id", "event_type", "event_date"])
    ev["available_at"], ev["source"] = UTC("2026-01-01"), "test"
    return SecurityMaster(sec, ids, ev)


def test_reused_ticker_resolves_by_date():
    m = _master()
    assert m.resolve("DELL", "2010-06-01") == "T:DELL_OLD"
    assert m.resolve("DELL", "2024-06-01") == "T:DELL_NEW"
    with pytest.raises(UnknownIdentifier):
        m.resolve("DELL", "2015-06-01")  # nobody traded as DELL then


def test_renamed_security_keeps_identity_and_history():
    m = _master()
    assert m.resolve("PCLN", "2010-01-04") == m.resolve("BKNG", "2024-01-02") == "T:BKNG"
    assert m.attribute("T:BKNG", "2010-01-04") == "PCLN"
    assert m.attribute("T:BKNG", "2010-01-04", "name") == "Priceline Group Inc"
    assert m.history("T:BKNG")["value"].tolist() == ["PCLN", "BKNG"]


def test_delisted_securities_stay_in_the_historical_universe():
    m = _master()
    assert "T:LEH" in m.listed_on("2008-06-02") and "T:LEH" not in m.listed_on("2009-01-02")
    assert m.delisting("T:LEH")["delisting_reason"] == "bankruptcy"
    idx = pd.DatetimeIndex(["2008-09-12", "2008-09-15", "2008-09-16", "2008-09-17", "2008-09-18"])
    vol = pd.DataFrame({"T:LEH": [1e6, 5e6, 3e6, 2e6, 0], "T:BKNG": [1, 1, 1, 0, 1.0]}, index=idx)
    t = m.tradable_mask(vol)
    assert t["T:LEH"].tolist() == [True, True, False, False, False]  # halted, then delisted
    assert t["T:BKNG"].tolist() == [True, True, True, False, True]  # zero volume ⇒ not tradable


def test_simultaneous_ticker_reuse_is_rejected():
    with pytest.raises(ValueError, match="two securities at once"):
        _master([("T:LEH", "ticker", "DELL", pd.Timestamp("2000-01-03"), pd.Timestamp("2001-01-02"))])


def test_coverage_reports_what_is_missing():
    c = _master().coverage()
    assert c["delisted"] == 2 and c["ticker_history"] and c["name_history"] and not c["exchange_history"]
    assert c["delisting_reason_coverage"] == 1.0 and "bankruptcy" in c["lifecycle_event_types"]
