import pandas as pd
import pytest

from quantlab.calendar import get_calendar
from quantlab.provenance import DataFlag, DataLabel, Dataset, Provenance, combine_labels, frame_hash


def test_simulated_is_contagious():
    assert combine_labels([DataLabel.REAL, DataLabel.REAL]) == DataLabel.DERIVED
    assert combine_labels([DataLabel.REAL, DataLabel.SIMULATED]) == DataLabel.SIMULATED
    assert combine_labels([DataLabel.DERIVED, DataLabel.SIMULATED]) == DataLabel.SIMULATED
    with pytest.raises(ValueError):
        combine_labels([])


def test_flags_propagate_on_derivation():
    a = Provenance(DataLabel.REAL, "x", frozenset({DataFlag.SURVIVORSHIP_RISK}))
    b = Provenance(DataLabel.REAL, "y", frozenset({DataFlag.VENDOR_ADJUSTED}))
    d = Provenance.derive("feat", [a, b])
    assert d.label == DataLabel.DERIVED
    assert d.flags == {DataFlag.SURVIVORSHIP_RISK, DataFlag.VENDOR_ADJUSTED}
    assert Provenance.from_dict(d.to_dict()) == d


def test_frame_hash_detects_any_change():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    h = frame_hash(df)
    assert frame_hash(df.copy()) == h
    df2 = df.copy()
    df2.loc[1, "a"] = 2.0000001
    assert frame_hash(df2) != h
    ds1 = Dataset("n", df, Provenance(DataLabel.REAL, "s"))
    ds2 = Dataset("n", df, Provenance(DataLabel.SIMULATED, "s"))
    assert ds1.content_hash != ds2.content_hash  # label is part of the version identity


def test_close_availability_respects_early_close_and_delay():
    cal = get_calendar()
    s = cal.sessions("2024-11-27", "2024-11-29")
    av = cal.daily_bar_available_at(s, delay_minutes=15)
    # regular close 16:00 ET = 21:00 UTC (EST); day after Thanksgiving closes 13:00 ET
    assert av[0] == pd.Timestamp("2024-11-27 21:15", tz="UTC")
    assert av[1] == pd.Timestamp("2024-11-29 18:15", tz="UTC")
    assert cal.is_early_close("2024-11-29")
    assert not cal.is_session("2024-11-28")  # Thanksgiving


def test_availability_rejects_non_sessions():
    with pytest.raises(ValueError):
        get_calendar().daily_bar_available_at(pd.DatetimeIndex(["2024-11-28"]))
