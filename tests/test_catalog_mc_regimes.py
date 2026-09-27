import numpy as np
import pandas as pd
import pytest

from quantlab.montecarlo.simulate import MonteCarloConfig, monte_carlo
from quantlab.research.catalog import (
    InvalidTransition, SignalStatus, all_signals, latest_record, transition, upsert_record,
)
from quantlab.research.regimes import performance_by_regime, regime_labels


def test_catalog_versions_and_lifecycle(registry):
    upsert_record(registry, "SIG-1", {"Name": "x", "Sharpe": 0.5}, "created")
    assert latest_record(registry, "SIG-1")["Status"] == "EXPERIMENTAL"
    upsert_record(registry, "SIG-1", {"Sharpe": 0.4}, "re-estimated")
    rec = latest_record(registry, "SIG-1")
    assert rec["Sharpe"] == 0.4 and rec["Name"] == "x" and rec["_version"] == 2
    assert len(registry.find("signal_records", signal_id="SIG-1")) == 2  # history preserved
    with pytest.raises(InvalidTransition):
        transition(registry, "SIG-1", SignalStatus.ACCEPTED, "skip validation")
    transition(registry, "SIG-1", SignalStatus.VALIDATING, "screened")
    transition(registry, "SIG-1", SignalStatus.ACCEPTED, "validated")
    with pytest.raises(InvalidTransition, match="approval"):
        transition(registry, "SIG-1", SignalStatus.PAPER_TRADING, "promote")
    transition(registry, "SIG-1", SignalStatus.PAPER_TRADING, "promote", approved_by="owner")
    with pytest.raises(ValueError):
        upsert_record(registry, "SIG-1", {"Status": "LIVE_ELIGIBLE"}, "sneaky")
    transition(registry, "SIG-1", SignalStatus.RETIRED, "decayed")
    with pytest.raises(InvalidTransition):
        transition(registry, "SIG-1", SignalStatus.ACCEPTED, "resurrect")  # terminal
    assert [s["Signal_ID"] for s in all_signals(registry)] == ["SIG-1"]


def test_rejected_is_terminal(registry):
    upsert_record(registry, "SIG-2", {"Name": "y"}, "c")
    transition(registry, "SIG-2", SignalStatus.REJECTED, "failed")
    for s in SignalStatus:
        if s != SignalStatus.REJECTED:
            with pytest.raises(InvalidTransition):
                transition(registry, "SIG-2", s, "retry")


def test_monte_carlo_basic_properties():
    rng = np.random.default_rng(0)
    good = rng.normal(0.01, 0.02, 300)
    out = monte_carlo(good, MonteCarloConfig(n_paths=300, position_fraction=0.5))  # ~+0.5%/trade × 300
    h = out["historical"]["10000"]
    assert h["p_double"] > 0.5 and h["p_ruin"] == 0.0
    q = h["ending_capital_quantiles"]
    assert q["p05"] <= q["p50"] <= q["p95"]
    assert out["stressed"]["10000"]["ending_capital_quantiles"]["p50"] < q["p50"]  # edge decay hurts
    bad = rng.normal(-0.02, 0.05, 300)
    hb = monte_carlo(bad, MonteCarloConfig(n_paths=300, position_fraction=0.5))["historical"]["10000"]
    assert hb["p_ruin"] > 0.5 and hb["p_loss_50"] > 0.5
    assert monte_carlo(good[:5])["status"] == "INSUFFICIENT_DATA"


def test_monte_carlo_small_accounts_suffer_fixed_costs_and_min_size():
    r = np.random.default_rng(1).normal(0.004, 0.03, 400)
    cfg = MonteCarloConfig(n_paths=300, position_fraction=0.2, fixed_cost_per_trade=1.0, min_position_value=50.0)
    out = monte_carlo(r, cfg)["historical"]
    assert out["100"]["avg_trades_skipped"] == 400  # $20 positions below a $50 minimum: no trades possible
    assert out["1000"]["ending_capital_quantiles"]["p50"] < 1000 < out["100000"]["ending_capital_quantiles"]["p50"] / 100


def test_monte_carlo_reproducible():
    r = np.random.default_rng(2).normal(0.003, 0.02, 200)
    a = monte_carlo(r, MonteCarloConfig(n_paths=200, seed=7))
    b = monte_carlo(r, MonteCarloConfig(n_paths=200, seed=7))
    assert a == b


def test_regime_labels_are_point_in_time(small_panel):
    full = regime_labels(small_panel, "SYNMKT")
    cut = small_panel["close"].index[900]
    part = regime_labels({k: v.loc[:cut] for k, v in small_panel.items()}, "SYNMKT")
    pd.testing.assert_frame_equal(full.loc[:cut], part)
    assert set(full["trend"].dropna()) <= {"BULL", "BEAR"}


def test_performance_by_regime_flags_small_cells(small_panel):
    labels = regime_labels(small_panel, "SYNMKT")
    idx = small_panel["close"].index[300:320]
    trades = pd.DataFrame({"signal_session": idx, "net_ret": np.linspace(-0.01, 0.01, 20)})
    out = performance_by_regime(trades, labels, min_n=30)
    assert all(v["small_sample"] for v in out["year"].values())
