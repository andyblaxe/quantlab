import numpy as np
import pandas as pd
import pytest

from quantlab.data.adjust import build_panel
from quantlab.data.pit import LookAheadError, PointInTimeFrame, asof_series
from quantlab.data.providers.base import CapabilityNotSupported
from quantlab.data.providers.synthetic import SyntheticProvider
from quantlab.data.schemas import BARS_DAILY
from quantlab.data.store import DataStore, DataValidationError
from quantlab.data.universe import MembershipUniverse, StaticUniverse
from quantlab.data.validation import validate_actions, validate_bars
from quantlab.provenance import DataFlag, DataLabel, Dataset, Provenance


def _bars(rows):
    df = pd.DataFrame(rows, columns=["symbol", "session", "open", "high", "low", "close", "volume"])
    df["session"] = pd.to_datetime(df["session"])
    from quantlab.calendar import get_calendar
    df["available_at"] = get_calendar().daily_bar_available_at(pd.DatetimeIndex(df["session"]))
    df["source"] = "test"
    return df


def test_synthetic_bars_are_valid_and_labeled(small_market):
    ds = small_market.bars()
    assert ds.label == DataLabel.SIMULATED
    assert BARS_DAILY.validate(ds.frame) == []
    rep = validate_bars(ds.frame, small_market.corporate_actions().frame)
    assert rep.ok, rep.summary()
    assert validate_actions(small_market.corporate_actions().frame).ok


def test_synthetic_has_splits_and_dividends(small_market):
    acts = small_market.corporate_actions().frame
    assert (acts["action"] == "split").any(), "fixture should exercise split handling"
    assert (acts["action"] == "dividend").any()


def test_tri_reproduces_true_total_returns_through_splits(small_market, small_panel):
    truth = small_market.truth["total_return"]
    for i, sym in enumerate(small_market.symbols):
        got = np.log1p(small_panel["ret"][sym].to_numpy()[1:])
        assert np.allclose(got, truth[1:, i], atol=1e-10)


def test_adjusted_prices_continuous_across_split(small_market, small_panel):
    acts = small_market.corporate_actions().frame
    split = acts[acts["action"] == "split"].iloc[0]
    sym, ex = split["symbol"], split["ex_date"]
    raw = small_panel["raw_close"][sym]
    adj = small_panel["close"][sym]
    i = raw.index.get_loc(ex)
    raw_move = raw.iloc[i] / raw.iloc[i - 1] - 1
    adj_move = adj.iloc[i] / adj.iloc[i - 1] - 1
    assert raw_move < -0.3  # raw tape shows the mechanical split drop
    assert abs(adj_move) < 0.3  # adjusted series does not


def test_adjustment_is_point_in_time(small_market):
    """Adjusted history up to t must not change when later data (incl. later splits) arrives."""
    bars, acts = small_market.bars().frame, small_market.corporate_actions().frame
    full = build_panel(bars, acts)
    cut = pd.Timestamp("2018-06-29")
    part = build_panel(bars[bars["session"] <= cut], acts[acts["ex_date"] <= cut])
    a = part["close"]
    b = full["close"].loc[a.index]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_validation_catches_impossible_bars():
    df = _bars([
        ("A", "2024-01-02", 10, 11, 9, 10.5, 100),
        ("A", "2024-01-03", 10, 9.5, 9, 10.5, 100),  # high below close
        ("A", "2024-01-04", -1, 11, 9, 10.5, 100),  # negative price
    ])
    rep = validate_bars(df)
    checks = {i.check for i in rep.errors}
    assert {"ohlc_inconsistent", "non_positive_or_missing_price"} <= checks


def test_validation_catches_early_availability_and_non_sessions():
    df = _bars([("A", "2024-01-02", 10, 11, 9, 10.5, 100), ("A", "2024-01-03", 10, 11, 9, 10.5, 100)])
    df.loc[1, "available_at"] = pd.Timestamp("2024-01-03 15:00", tz="UTC")  # before the close
    rep = validate_bars(df)
    assert "available_before_close" in {i.check for i in rep.errors}
    df2 = _bars([("A", "2024-01-02", 10, 11, 9, 10.5, 100)])
    df2.loc[0, "session"] = pd.Timestamp("2024-01-06")  # Saturday
    assert "non_trading_session" in {i.check for i in validate_bars(df2).errors}


def test_validation_warns_on_suspicious_but_possible_data():
    rows = [("A", d, 10, 11, 9, 10, 100) for d in pd.bdate_range("2024-01-02", "2024-01-12")
            if d != pd.Timestamp("2024-01-15")]
    rows.append(("A", "2024-01-16", 20, 25, 19, 24, 100))  # +140% jump, no action; also a gap day
    rep = validate_bars(_bars(rows))
    w = {i.check for i in rep.warnings}
    assert "extreme_move_without_action" in w
    assert "stale_prices" in w
    assert rep.ok  # warnings do not block


def test_store_is_write_once_and_verifies_integrity(tmp_path, small_market):
    st = DataStore(tmp_path)
    ds = small_market.bars()
    h1 = st.save("bars_daily", ds, actions=small_market.corporate_actions().frame)
    h2 = st.save("bars_daily", ds, actions=small_market.corporate_actions().frame)
    assert h1 == h2
    loaded = st.load("bars_daily", h1)
    assert loaded.label == DataLabel.SIMULATED and len(loaded.frame) == len(ds.frame)
    man = st.manifest("bars_daily", h1)
    assert man["provenance"]["label"] == "SIMULATED" and man["validation"]["ok"]
    n = st.query("SELECT count(*) AS n FROM b", b="bars_daily")["n"].iloc[0]
    assert n == len(ds.frame)
    # tamper with the stored file → load fails
    pq = tmp_path / "bars_daily" / f"{h1}.parquet"
    df = pd.read_parquet(pq)
    df.loc[0, "close"] *= 1.01
    df.to_parquet(pq, index=False)
    with pytest.raises(IOError):
        st.load("bars_daily", h1)


def test_store_roundtrip_survives_dtype_changes(tmp_path):
    # object string columns come back from Parquet as str under pandas 3; the hash must still match
    df = pd.DataFrame({"symbol": pd.Series(["A", "B"], dtype=object),
                       "ex_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                       "action": pd.Series(["dividend", "split"], dtype=object),
                       "value": [0.5, 2.0],
                       "available_at": pd.to_datetime(["2024-01-02 14:30", "2024-01-03 14:30"], utc=True),
                       "source": pd.Series(["test", "test"], dtype=object)})
    st = DataStore(tmp_path)
    h = st.save("corporate_actions", Dataset("corporate_actions", df, Provenance(DataLabel.REAL, "test")))
    assert len(st.load("corporate_actions", h).frame) == 2


def test_store_refuses_invalid_data_unless_forced(tmp_path):
    df = _bars([("A", "2024-01-02", 10, 9, 9, 10.5, 100)])
    ds = Dataset("bad", df, Provenance(DataLabel.REAL, "test"))
    st = DataStore(tmp_path)
    with pytest.raises(DataValidationError):
        st.save("bars_daily", ds)
    h = st.save("bars_daily", ds, force=True)
    assert st.manifest("bars_daily", h)["forced"] is True


def test_pit_frame_returns_latest_known_vintage():
    t = lambda s: pd.Timestamp(s, tz="UTC")
    df = pd.DataFrame({
        "series_id": ["GDP"] * 3,
        "observation_date": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-06-30"]),
        "value": [1.0, 1.5, 2.0],
        "available_at": [t("2024-04-25"), t("2024-05-30"), t("2024-07-25")],
    })
    pit = PointInTimeFrame(df, key=("series_id", "observation_date"))
    assert pit.as_of(t("2024-04-01")).empty
    assert pit.as_of(t("2024-05-01"))["value"].tolist() == [1.0]
    assert pit.as_of(t("2024-06-01"))["value"].tolist() == [1.5]  # revision visible only after release
    with pytest.raises(ValueError):
        pit.as_of("2024-06-01")  # naive timestamp is ambiguous
    with pytest.raises(LookAheadError):
        pit.assert_available(df, t("2024-06-01"))


def test_asof_series_never_uses_future_values():
    t = lambda s: pd.Timestamp(s, tz="UTC")
    df = pd.DataFrame({
        "observation_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
        "value": [10.0, 11.0, 99.0],  # late revision of the older observation
        "available_at": [t("2024-01-02 21:30"), t("2024-01-03 21:30"), t("2024-01-05 12:00")],
    })
    decisions = pd.DatetimeIndex([t("2024-01-02 21:00"), t("2024-01-02 22:00"), t("2024-01-04 00:00"),
                                  t("2024-01-06 00:00")])
    out = asof_series(df, decisions)
    assert np.isnan(out["value"].iloc[0])  # nothing known yet
    assert out["value"].iloc[1] == 10.0
    assert out["value"].iloc[2] == 11.0
    assert out["value"].iloc[3] == 11.0  # old-period revision must not displace the newer period
    assert (out["value_available_at"].dropna() <= decisions[1:]).all()


def test_membership_universe_is_point_in_time():
    t = lambda s: pd.Timestamp(s, tz="UTC")
    mem = pd.DataFrame({
        "index_id": ["IDX"] * 2, "symbol": ["OLD", "NEW"],
        "start_date": pd.to_datetime(["2000-01-01", "2020-06-22"]),
        "end_date": pd.to_datetime(["2020-06-22", None]),
        "available_at": [t("2000-01-01"), t("2020-06-12 22:00")], "source": "test",
    })
    u = MembershipUniverse("IDX", mem)
    assert u.members("2020-06-19") == ["OLD"]
    assert u.members("2020-06-22") == ["NEW"]
    idx = pd.bdate_range("2020-06-15", "2020-06-26")
    m = u.mask(idx, pd.Index(["OLD", "NEW"]))
    assert m.loc["2020-06-19", "OLD"] and not m.loc["2020-06-19", "NEW"]
    assert m.loc["2020-06-22", "NEW"] and not m.loc["2020-06-22", "OLD"]


def test_static_universe_carries_survivorship_flag():
    u = StaticUniverse("big_caps_today", ["AAPL"], flags=frozenset({DataFlag.SURVIVORSHIP_RISK}))
    assert DataFlag.SURVIVORSHIP_RISK in u.flags


def test_provider_capabilities_are_explicit(small_market):
    p = SyntheticProvider(small_market)
    assert "get_daily_bars" in p.capabilities()
    assert "get_option_quotes_eod" not in p.capabilities()
    with pytest.raises(CapabilityNotSupported):
        p.get_option_quotes_eod("SYN000", "2020-01-01", "2020-12-31")
    ds = p.get_daily_bars(["SYN000"], "2020-01-01", "2020-01-31")
    assert set(ds.frame["symbol"]) == {"SYN000"} and ds.label == DataLabel.SIMULATED
