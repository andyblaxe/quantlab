import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio.combine import SignalInput, combine_signals, effective_number, fractional_kelly_portfolio
from quantlab.risk.gate import ProposedTrade, RiskGate, RiskLimits
from quantlab.risk.metrics import (
    PortfolioState, Position, cornish_fisher_var, historical_var_es, parametric_var_es, portfolio_vol, shrunk_covariance,
)
from quantlab.risk.sizing import (
    SizingPolicy, growth_rate, kelly_empirical, kelly_gaussian, recommended_fraction, ruin_probability_gaussian,
)


def test_var_es_known_values():
    r = np.concatenate([np.full(94, 0.01), np.full(6, -0.10)])
    h = historical_var_es(r, 0.95)
    assert h["var"] == pytest.approx(0.10) and h["es"] == pytest.approx(0.10)
    p = parametric_var_es(0.0, 0.01, 0.95)
    assert p["var"] == pytest.approx(0.01645, abs=1e-4) and p["es"] > p["var"]
    assert cornish_fisher_var(0, 0.01, -1.0, 3.0) > p["var"]  # negative skew / fat tails raise VaR


def test_shrunk_covariance_is_psd_and_vol():
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.normal(0, 0.01, (60, 8)))
    C = shrunk_covariance(R)
    assert np.all(np.linalg.eigvalsh(C.to_numpy()) > 0)
    w = pd.Series(1 / 8, index=R.columns)
    assert 0 < portfolio_vol(w, C) < 0.2


def test_kelly_is_fractional_uncertainty_aware_and_capped():
    rng = np.random.default_rng(1)
    r = rng.normal(0.01, 0.05, 500)
    assert kelly_empirical(r) == pytest.approx(kelly_gaussian(0.01, 0.05), rel=0.35)
    rec = recommended_fraction(r)
    assert 0 < rec["fraction"] <= 0.20 and rec["lower_bound"] < rec["mean"]
    assert recommended_fraction(r[:10])["fraction"] == 0.0  # too few trades
    assert recommended_fraction(rng.normal(0.0, 0.05, 500))["fraction"] == 0.0  # no edge → no trade
    # full Kelly beyond the optimum reduces growth — fractional is the conservative side
    fk = kelly_empirical(r)
    assert growth_rate(r, fk) > growth_rate(r, 2.5 * fk)
    assert growth_rate(r, 0.25 * fk) > 0


def test_ruin_probability_monotone_in_leverage():
    lo = ruin_probability_gaussian(0.01, 0.05, 0.5)
    hi = ruin_probability_gaussian(0.01, 0.05, 6.0)
    assert lo < hi <= 1.0
    assert ruin_probability_gaussian(0.01, 0.05, 10.0) == 1.0  # beyond 2x Kelly: negative log drift


def _state(equity=10_000, cash=10_000, positions=(), peak=None):
    return PortfolioState(equity=equity, cash=cash, positions=list(positions), peak_equity=peak or equity)


def test_gate_vetoes_regardless_of_signal():
    gate = RiskGate()
    # model loves it, but it is 3x the per-position cap → reduced
    d = gate.evaluate(ProposedTrade("SPY", 60, 100.0, kind="etf", signal_id="SIG-1"), _state())
    assert d.decision == "REDUCE" and d.approved_quantity == pytest.approx(20)
    # drawdown halt: no new risk at all
    d = gate.evaluate(ProposedTrade("SPY", 1, 100.0), _state(equity=7_000, cash=7_000, peak=10_000))
    assert d.decision == "REJECT" and "drawdown" in d.reasons[0]
    # shorting disabled
    assert gate.evaluate(ProposedTrade("SPY", -5, 100.0), _state()).decision == "REJECT"
    # naked short option
    t = ProposedTrade("SPY", -1, 2.0, kind="option", multiplier=100, is_short_option=True, defined_risk=False, max_loss=None)
    assert gate.evaluate(t, _state()).decision == "REJECT"
    # option without computed max loss
    assert gate.evaluate(ProposedTrade("SPY", 1, 2.0, kind="option", multiplier=100), _state()).decision == "REJECT"


def test_gate_sector_liquidity_event_and_premium_limits():
    gate = RiskGate(RiskLimits(max_position_fraction=0.5))
    pos = [Position("XLK", 30, 100.0, sector="TECH")]
    d = gate.evaluate(ProposedTrade("QQQ", 30, 100.0, sector="TECH"), _state(cash=7_000, positions=pos))
    assert d.decision == "REDUCE" and "sector" in d.reasons[0]
    d = gate.evaluate(ProposedTrade("THIN", 10, 100.0, adv_dollar=50_000), _state())
    assert d.approved_quantity == pytest.approx(5) and "liquidity" in d.reasons[0]
    d = gate.evaluate(ProposedTrade("ERN", 10, 100.0, event_risk=True), _state())
    assert d.approved_quantity == pytest.approx(10) and d.decision == "APPROVE"
    d = gate.evaluate(ProposedTrade("ERN", 20, 100.0, event_risk=True), _state())
    assert d.decision == "REDUCE" and "event_risk" in d.reasons[0]
    # a $100 account cannot afford a $300 option contract (premium-at-risk + cash)
    t = ProposedTrade("SPY", 1, 3.0, kind="option", multiplier=100, max_loss=300.0)
    assert gate.evaluate(t, _state(equity=100, cash=100)).decision == "REJECT"


def test_gate_always_allows_risk_reduction():
    pos = [Position("SPY", 50, 100.0)]
    d = RiskGate().evaluate(ProposedTrade("SPY", -50, 100.0), _state(equity=6_000, cash=1_000, positions=pos, peak=10_000))
    assert d.decision == "APPROVE"


def test_signal_combination_does_not_double_count():
    rng = np.random.default_rng(2)
    base = rng.normal(0.001, 0.01, 1000)
    other = rng.normal(0.001, 0.01, 1000)
    R = pd.DataFrame({"A": base, "A_copy": base + rng.normal(0, 1e-5, 1000), "B": other})
    sigs = [SignalInput(k, 0.001, 0.0003, "STABLE") for k in R.columns]
    out = combine_signals(sigs, R)
    w = out["weights"]
    assert w["A"] + w["A_copy"] == pytest.approx(w["B"], rel=0.15)  # the duplicated signal shares one weight
    assert 1.5 < out["effective_independent_signals"] < 2.2  # participation ratio: eigenvalues (2,1,0) → 1.8
    decayed = combine_signals([SignalInput("A", 0.001, 0.0003, "DISAPPEARED"),
                               SignalInput("B", 0.001, 0.0003, "DISAPPEARED")], R)
    assert decayed["status"] == "NO_TRADE"
    inactive = combine_signals([SignalInput("A", 0.001, 0.0003, "STABLE", active=False)], R)
    assert inactive["status"] == "NO_TRADE"


def test_fractional_kelly_portfolio_respects_caps():
    mu = pd.Series({"A": 0.0008, "B": 0.0004, "C": -0.0002})
    cov = pd.DataFrame(np.diag([0.0001, 0.0001, 0.0001]), index=mu.index, columns=mu.index)
    w = fractional_kelly_portfolio(mu, cov, fraction=0.25, cap=0.2)
    assert w["C"] == 0 and (w <= 0.2 + 1e-9).all() and w.sum() <= 1 + 1e-9
    assert w["A"] >= w["B"]
    assert (fractional_kelly_portfolio(-mu.abs(), cov) == 0).all()
