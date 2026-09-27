import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

import quantlab.features as F
from quantlab.data.pit import LookAheadError
from quantlab.features.base import FeatureDef, assert_no_lookahead, check_no_lookahead, get_feature, parse_spec
from quantlab.research.labels import forward_return


def _spec_for(fam):
    return fam + ("(market=SYNMKT)" if fam in ("beta", "corr", "rel_ret") else "")


@pytest.mark.parametrize("family", sorted(F.FAMILIES))
def test_every_feature_passes_truncation_test(family, small_panel):
    f = get_feature(_spec_for(family))
    assert check_no_lookahead(f, small_panel, n_checks=6) == []


@pytest.mark.parametrize("bad", ["centered", "shift_back", "full_sample_z"])
def test_truncation_test_catches_leaky_features(bad, small_panel):
    fns = {
        "centered": lambda p: p["close"].rolling(5, center=True).mean(),
        "shift_back": lambda p: p["close"].shift(-1) / p["close"] - 1,
        "full_sample_z": lambda p: (p["ret"] - p["ret"].mean()) / p["ret"].std(),
    }
    f = FeatureDef(bad, bad, {}, fns[bad], 5, "test", "deliberately leaky")
    with pytest.raises(LookAheadError):
        assert_no_lookahead(f, small_panel, n_checks=6)


def test_spec_parsing_and_canonical_names():
    assert parse_spec("sma_ratio(n=200)") == ("sma_ratio", {"n": 200})
    assert parse_spec("bollinger(n=20, k=2.5)") == ("bollinger", {"n": 20, "k": 2.5})
    assert get_feature("ma_cross(slow=200,fast=50)").name == "ma_cross(fast=50,slow=200)"
    with pytest.raises(KeyError):
        get_feature("not_a_feature")
    with pytest.raises(ValueError):
        get_feature("ret(bogus=1)")


def test_known_values(small_panel):
    c = small_panel["close"]["SYN000"]
    r5 = get_feature("ret(h=5)").compute(small_panel)["SYN000"]
    assert np.isclose(r5.iloc[10], c.iloc[10] / c.iloc[5] - 1)
    sma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    bb = get_feature("bollinger(n=20,k=2.0)").compute(small_panel)["SYN000"]
    assert np.isclose(bb.iloc[50], (c.iloc[50] - sma.iloc[50]) / (2 * sd.iloc[50]))
    rsi = get_feature("rsi(n=14)").compute(small_panel)
    assert rsi.stack().dropna().between(0, 100).all()


def test_rsi_extremes():
    idx = pd.bdate_range("2020-01-01", periods=60)
    up = pd.DataFrame({"X": np.linspace(10, 20, 60)}, index=idx)
    p = {"close": up}
    assert get_feature("rsi(n=14)").compute(p)["X"].iloc[-1] == pytest.approx(100.0)


def test_features_package_never_imports_labels():
    root = pathlib.Path(F.__file__).parent
    for py in root.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None) or ""
                names = [a.name for a in node.names]
                assert "labels" not in mod and not any("labels" in n for n in names), f"{py.name} imports labels"


def test_forward_return_alignment(small_panel):
    fr = forward_return(small_panel, 3, "next_open")["SYN000"]
    o = small_panel["open"]["SYN000"]
    assert np.isclose(fr.iloc[10], o.iloc[14] / o.iloc[11] - 1)
    assert fr.iloc[-4:].isna().all()  # exits beyond data are NaN, not extrapolated
    fc = forward_return(small_panel, 1, "next_close")["SYN000"]
    c = small_panel["close"]["SYN000"]
    assert np.isclose(fc.iloc[10], c.iloc[12] / c.iloc[11] - 1)
