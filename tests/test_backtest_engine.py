from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.benchmarks import buy_and_hold
from quantlab.backtest.costs import COST_PROFILES, CostModel, get_cost_model
from quantlab.backtest.engine import BacktestConfig, run_portfolio_backtest, simulate
from quantlab.backtest.metrics import max_drawdown, summarize_returns
from quantlab.data.adjust import build_panel
from quantlab.data.pit import LookAheadError

FRICTIONLESS = COST_PROFILES["frictionless"]


def test_single_asset_frictionless_matches_open_to_open_total_return(small_panel):
    sym = "SYN001"
    w = pd.DataFrame({sym: 1.0}, index=small_panel["close"].index)
    res = simulate(small_panel, w, BacktestConfig(capital=1e6, cost_model=replace(FRICTIONLESS, max_participation=1e9)))
    eq = res.daily["equity"]
    # entered at open of session 1; from then equity tracks the adjusted (total-return) series
    o, c = small_panel["open"][sym], small_panel["close"][sym]
    expected = 1e6 * c.iloc[5:] / o.iloc[1]
    # Not exact: the engine reinvests dividend cash at the ex-date *open* (next rebalance) while the
    # TRI reinvests at the ex-date *close*. The difference compounds to a few basis points.
    assert np.allclose(eq.iloc[5:].to_numpy(), expected.to_numpy(), rtol=2e-3)
    assert np.allclose(eq.iloc[5:40].to_numpy(), expected.iloc[:35].to_numpy(), rtol=1e-9)  # before any dividend


def test_splits_and_dividends_do_not_create_pnl(small_market, small_panel):
    """Holding through ex-dates: equity must follow total return, not the raw price drop."""
    acts = small_market.corporate_actions().frame
    sym = acts.loc[acts["action"] == "split", "symbol"].iloc[0]
    w = pd.DataFrame({sym: 1.0}, index=small_panel["close"].index)
    d = simulate(small_panel, w, BacktestConfig(capital=1e6, cost_model=replace(FRICTIONLESS, max_participation=1e9))).daily
    r = d["ret"].iloc[2:]
    tr = small_panel["ret"][sym].iloc[2:]
    ex = acts[(acts["symbol"] == sym)]
    split_days = ex.loc[ex["action"] == "split", "ex_date"]
    # On split days the equity return equals the total return exactly (no phantom P&L)...
    assert np.allclose(r.loc[split_days].to_numpy(), tr.loc[split_days].to_numpy(), atol=1e-10)
    # ...and overall the difference is only dividend-reinvestment timing (open vs close)
    assert np.abs(r - tr).max() < 5e-4
    assert d["dividends"].sum() > 0


def test_accounting_identity_with_costs(small_panel):
    idx = small_panel["close"].index
    syms = ["SYN000", "SYN002", "SYN003"]
    rng = np.random.default_rng(1)
    w = pd.DataFrame(rng.dirichlet(np.ones(3), size=len(idx)) * 0.9, index=idx, columns=syms)
    cfg = BacktestConfig(capital=250_000, cost_model=get_cost_model("per_share_broker"))
    d = simulate(small_panel, w, cfg).daily
    # equity change = market P&L + dividends - costs - borrow; check the cost bookkeeping is non-trivial
    assert (d["cost_spread_slip_impact"] > 0).sum() > 100 and (d["cost_commission"] >= 1.0).any()
    res = run_portfolio_backtest(small_panel, w, cfg)
    gross_tr, net_tr = res.gross.daily["equity"].iloc[-1], res.net.daily["equity"].iloc[-1]
    assert net_tr < gross_tr  # costs strictly reduce performance
    assert res.gross.daily[["cost_spread_slip_impact", "cost_commission"]].sum().sum() == 0


def test_gross_equals_net_when_frictionless(small_panel):
    idx = small_panel["close"].index
    w = buy_and_hold(idx, ["SYN000", "SYN001"])
    res = run_portfolio_backtest(small_panel, w, BacktestConfig(cost_model=FRICTIONLESS))
    assert np.allclose(res.gross.daily["equity"], res.net.daily["equity"])


def test_no_trade_is_acceptable(small_panel):
    idx = small_panel["close"].index
    w = pd.DataFrame(0.0, index=idx, columns=["SYN000"])
    d = simulate(small_panel, w, BacktestConfig(capital=1000)).daily
    assert (d["equity"] == 1000).all() and d["n_fills"].sum() == 0


def test_integer_shares_small_account(small_panel):
    idx = small_panel["close"].index
    w = pd.DataFrame({"SYN000": 1.0}, index=idx)
    px = small_panel["raw_open"]["SYN000"].iloc[1]
    cap = px * 0.5
    cfg = BacktestConfig(capital=cap, fractional_shares=False)
    res = simulate(small_panel, w, cfg)
    d = res.daily
    # no fill while one share costs more than the whole account; any fill that happens is affordable
    unaffordable = small_panel["raw_open"]["SYN000"] > d["equity"].shift(1).fillna(cap)
    assert d.loc[unaffordable, "n_fills"].sum() == 0
    assert d["n_fills"].iloc[:5].sum() == 0
    assert (res.trades["shares"] == res.trades["shares"].round()).all()
    assert (d["cash"] > -0.01 * cap).all()


def test_liquidity_cap_leaves_orders_unfilled(small_panel):
    idx = small_panel["close"].index
    w = pd.DataFrame({"SYN000": 1.0}, index=idx)
    cfg = BacktestConfig(capital=1e12, cost_model=replace(FRICTIONLESS, max_participation=0.01))
    d = simulate(small_panel, w, cfg).daily
    assert d["unfilled_value"].iloc[1] > 0
    adv = small_panel["dollar_volume"]["SYN000"].rolling(20, min_periods=1).mean()
    assert d["turnover"].iloc[1] * 1e12 <= 0.01 * adv.iloc[0] * 1.000001


def test_no_fill_on_missing_or_zero_volume_bar(small_market):
    bars = small_market.bars().frame.copy()
    target_day = bars["session"].unique()[10]
    bars.loc[(bars["symbol"] == "SYN000") & (bars["session"] == target_day), "volume"] = 0.0
    panel = build_panel(bars, small_market.corporate_actions().frame)
    idx = panel["close"].index
    w = pd.DataFrame(np.nan, index=idx, columns=["SYN000"])
    w.iloc[9] = 1.0  # decision at 9 → would execute on day 10, which has zero volume
    d = simulate(panel, w, BacktestConfig(capital=1e5)).daily
    assert d.loc[target_day, "n_fills"] == 0


def test_shorting_requires_permission(small_panel):
    idx = small_panel["close"].index
    w = pd.DataFrame({"SYN000": -0.5}, index=idx)
    with pytest.raises(ValueError, match="shorting"):
        simulate(small_panel, w, BacktestConfig())
    d = simulate(small_panel, w, BacktestConfig(cost_model=replace(get_cost_model("retail_etf"), allow_short=True))).daily
    assert d["cost_borrow"].sum() > 0 and (d["net_exposure"].iloc[5:] < 0).all()


def test_leverage_limit_enforced(small_panel):
    idx = small_panel["close"].index
    w = pd.DataFrame({"SYN000": 0.8, "SYN001": 0.8}, index=idx)
    with pytest.raises(ValueError, match="leverage"):
        simulate(small_panel, w, BacktestConfig(max_gross_leverage=1.0))


def test_timing_guard_rejects_impossible_availability(small_panel):
    p = dict(small_panel)
    av = p["available_at"].copy()
    av.iloc[20] = av.iloc[20] + pd.Timedelta(days=3)  # a bar that only became available days later
    p["available_at"] = av
    w = pd.DataFrame({"SYN000": 1.0}, index=p["close"].index)
    with pytest.raises(LookAheadError):
        simulate(p, w, BacktestConfig())


def test_backtest_is_deterministic(small_panel):
    idx = small_panel["close"].index
    w = buy_and_hold(idx, ["SYN000", "SYN004"])
    cfg = BacktestConfig(cost_model=get_cost_model("retail_largecap"))
    a = run_portfolio_backtest(small_panel, w, cfg).net.daily
    b = run_portfolio_backtest(small_panel, w, cfg).net.daily
    pd.testing.assert_frame_equal(a, b)


def test_cost_model_properties():
    cm = CostModel(half_spread_bps=5, slippage_bps=5, impact_coef=0.1, commission_per_share=0.005, commission_min=1.0)
    small = cm.price_impact_frac(1_000, 1e8, 0.02)
    big = cm.price_impact_frac(1_000_000, 1e8, 0.02)
    assert big > small > 0.001  # impact grows with size
    assert cm.commission(100, 1) == 1.0  # minimum applies
    assert cm.scaled(2).half_spread_bps == 10 and cm.scaled(2).max_participation == cm.max_participation
    assert cm.zero().price_impact_frac(1e6, 1e8, 0.02) == 0
    # commission minimum dominates tiny trades: $1 on a $50 trade is 2% each way
    assert cm.round_trip_frac(50, 50, 1e9, 0.01) > 0.04


def test_metrics_known_values():
    r = pd.Series([0.1, -0.5, 0.2], index=pd.bdate_range("2020-01-01", periods=3))
    assert np.isclose(max_drawdown(r), -0.5)
    s = summarize_returns(pd.Series(np.full(252, 0.001), index=pd.bdate_range("2020-01-01", periods=252)))
    assert np.isclose(s["total_return"], 1.001**252 - 1)
    assert np.isclose(s["cagr"], 1.001**252 - 1)
    assert s["max_drawdown"] == 0
