import numpy as np
import pandas as pd
import pytest

from quantlab.config import Settings
from quantlab.paper.broker import LiveTradingDisabled, Order, PaperBroker, Quote, assert_live_trading_permitted
from quantlab.paper.forward import (
    ForwardTester, NotPaperTradable, compare_with_history, cusum_degradation, forward_results, live_eligibility,
)
from quantlab.research.catalog import SignalStatus, transition, upsert_record
from quantlab.risk.gate import RiskGate

T0 = pd.Timestamp("2026-01-05 15:00", tz="UTC")


def _broker(cash=10_000):
    b = PaperBroker(starting_cash=cash, slippage_bps=2.0)
    b.set_quote(Quote("SPY", 99.98, 100.02, T0, source="test"))
    return b


def test_live_trading_lock(tmp_path, monkeypatch):
    with pytest.raises(LiveTradingDisabled):
        assert_live_trading_permitted(Settings(_env_file=None))
    monkeypatch.setenv("QUANTLAB_LIVE_TRADING_ENABLED", "true")
    with pytest.raises(LiveTradingDisabled, match="safety-review"):
        assert_live_trading_permitted(Settings(_env_file=None))
    f = tmp_path / "review.txt"
    f.write_text("looks fine")
    monkeypatch.setenv("QUANTLAB_LIVE_SAFETY_REVIEW_FILE", str(f))
    with pytest.raises(LiveTradingDisabled, match="APPROVED-BY"):
        assert_live_trading_permitted(Settings(_env_file=None))


def test_paper_broker_fills_conservatively():
    b = _broker()
    f = b.submit(Order("SPY", 10))
    assert f.status == "FILLED" and f.price == pytest.approx(100.02 * 1.0002)  # ask + slippage, never mid
    s = b.submit(Order("SPY", -10))
    assert s.price == pytest.approx(99.98 * 0.9998)
    assert b.cash() < 10_000  # round trip costs money
    assert b.submit(Order("QQQ", 1)).reason == "no quote"
    b.now = T0 + pd.Timedelta(hours=1)
    assert b.submit(Order("SPY", 1)).reason == "stale quote"
    b.now = T0
    assert b.submit(Order("SPY", 1_000)).reason == "insufficient cash"


def _paper_signal(reg):
    upsert_record(reg, "SIG-000001", {"Name": "x", "Hypothesis_ID": "H-000001"}, "c")
    transition(reg, "SIG-000001", SignalStatus.VALIDATING, "v")
    transition(reg, "SIG-000001", SignalStatus.ACCEPTED, "a")


def test_forward_tester_requires_paper_status_and_risk_gate(registry):
    _paper_signal(registry)
    ft = ForwardTester(registry, _broker(), RiskGate())
    with pytest.raises(NotPaperTradable):
        ft.open("SIG-000001", "SPY", 10, "signal fired", {}, {})
    transition(registry, "SIG-000001", SignalStatus.PAPER_TRADING, "approved", approved_by="owner")
    big = ft.open("SIG-000001", "SPY", 100, "signal fired", {"vix": 15}, {"mean": 0.002})
    assert big["status"] == "FILLED" and big["quantity"] == pytest.approx(20, rel=1e-3)  # gate cut 100 → ~20 (20% cap at the ask)
    rec = registry.get("paper_trades", big["id"])["payload"]
    for key in ("timestamp", "signal_id", "market_state", "recommended_quantity", "quote", "fill", "position_size",
                "reason", "expected", "slippage_vs_mid", "strategy_version", "risk_decision"):
        assert key in rec
    ft.broker.set_quote(Quote("SPY", 101.0, 101.04, T0 + pd.Timedelta(days=1)))
    ft.broker.now = T0 + pd.Timedelta(days=1)
    cl = ft.close(big["id"], "holding period over")
    assert cl["realized_net"] > 0 and len(forward_results(registry, "SIG-000001")) == 1


def test_forward_comparison_and_degradation():
    rng = np.random.default_rng(0)
    hist = pd.Series(rng.normal(0.002, 0.01, 500))
    good = compare_with_history(pd.Series(rng.normal(0.002, 0.01, 40)), hist)
    bad = compare_with_history(pd.Series(rng.normal(-0.006, 0.01, 40)), hist)
    assert not good["below_5th_percentile"] and bad["below_5th_percentile"]
    assert cusum_degradation(pd.Series(rng.normal(-0.01, 0.01, 60)), 0.002, 0.01)["recommendation"] == "DEGRADED"
    assert cusum_degradation(pd.Series(rng.normal(0.002, 0.01, 60)), 0.002, 0.01)["alarm"] is False


def test_live_eligibility_needs_forward_evidence(registry):
    _paper_signal(registry)
    res = live_eligibility(registry, "SIG-000001", pd.Series(np.full(100, 0.001)), modelled_slippage=0.0003)
    assert res["eligible"] is False and res["paper_trades"]["value"] == 0
