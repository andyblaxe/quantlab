import numpy as np
import pandas as pd
import pytest

from quantlab.options.chain import arbitrage_checks, atm_term_structure, enrich_chain, simulated_chain, skew_25d
from quantlab.options.execution import LegSpec, OptionCostModel, simulate_option_trade
from quantlab.options.pricing import (
    bs_greeks, bs_price, breeden_litzenberger, crr_price, early_exercise_premium, expected_move_from_iv,
    expected_move_from_straddle, implied_event_move, implied_vol, parity_arbitrage, parity_gap, price_bounds, prob_above,
)
from quantlab.options.strategies import covered_call, iron_condor, long_call, straddle, vertical


def test_black_scholes_known_values():
    # Hull, classic example: S=K=100, T=1, r=5%, sigma=20%
    assert bs_price(100, 100, 1, 0.05, 0, 0.2, "C") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(100, 100, 1, 0.05, 0, 0.2, "P") == pytest.approx(5.5735, abs=1e-4)
    # Hull 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5 → c=4.76, p=0.81
    assert bs_price(42, 40, 0.5, 0.10, 0, 0.2, "C") == pytest.approx(4.76, abs=0.01)
    assert bs_price(42, 40, 0.5, 0.10, 0, 0.2, "P") == pytest.approx(0.81, abs=0.01)


def test_put_call_parity_holds_for_model_prices():
    S, K, T, r, q, s = 95.0, 100.0, 0.7, 0.03, 0.015, 0.31
    c, p = bs_price(S, K, T, r, q, s, "C"), bs_price(S, K, T, r, q, s, "P")
    assert parity_gap(c, p, S, K, T, r, q) == pytest.approx(0.0, abs=1e-10)


def test_greeks_match_finite_differences():
    S, K, T, r, q, s = 100.0, 105.0, 0.4, 0.02, 0.01, 0.25
    for right in ("C", "P"):
        g = bs_greeks(S, K, T, r, q, s, right)
        h = 1e-3
        f = lambda **kw: float(bs_price(kw.get("S", S), K, kw.get("T", T), kw.get("r", r), q, kw.get("s", s), right))
        assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-5)
        assert g["gamma"] == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-3)
        assert g["vega"] == pytest.approx((f(s=s + h) - f(s=s - h)) / (2 * h), rel=1e-5)
        assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-4)
        assert g["rho"] == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-4)


def test_implied_vol_roundtrip_and_bounds():
    S, T, r, q = 100.0, 0.5, 0.03, 0.0
    K = np.array([80, 90, 100, 110, 120.0])
    sig = np.array([0.35, 0.28, 0.22, 0.2, 0.21])
    for right in ("C", "P"):
        px = bs_price(S, K, T, r, q, sig, right)
        assert np.allclose(implied_vol(px, S, K, T, r, q, right), sig, atol=1e-7)
    lo, hi = price_bounds(100, 80, 0.5, 0.03, 0, "C")
    assert np.isnan(implied_vol(lo - 0.5, 100, 80, 0.5, 0.03, 0, "C"))  # below intrinsic: no IV, not faked
    assert np.isnan(implied_vol(hi + 1, 100, 80, 0.5, 0.03, 0, "C"))
    assert np.isnan(implied_vol(5.0, 100, 100, 0.0, 0.03, 0, "C"))  # expired


def test_binomial_converges_and_early_exercise():
    eu = crr_price(100, 100, 1, 0.05, 0, 0.2, "P", american=False, steps=800)
    assert eu == pytest.approx(5.5735, abs=0.01)
    am = crr_price(100, 100, 1, 0.05, 0, 0.2, "P", american=True, steps=800)
    assert am > eu  # early exercise has value for puts
    # American call on a non-dividend stock is never exercised early
    assert early_exercise_premium(100, 100, 1, 0.05, 0.0, 0.2, "C") == pytest.approx(0.0, abs=1e-6)
    assert early_exercise_premium(100, 100, 1, 0.05, 0.08, 0.2, "C") > 0  # with a high dividend yield it is


def test_expected_and_event_moves():
    em = expected_move_from_iv(100, 0.2, 30 / 365)
    assert em["one_sd_move"] == pytest.approx(100 * 0.2 * np.sqrt(30 / 365))
    atm_straddle = float(bs_price(100, 100, 30 / 365, 0, 0, 0.2, "C") + bs_price(100, 100, 30 / 365, 0, 0, 0.2, "P"))
    fs = expected_move_from_straddle(atm_straddle, 100)
    assert fs["one_sd_move"] == pytest.approx(em["one_sd_move"], rel=0.01)
    # construct a term structure with a 5% event and 25% diffusive vol, then recover the event
    T1, T2, sd, e = 7 / 365, 35 / 365, 0.25, 0.05
    iv1, iv2 = np.sqrt(sd**2 + e**2 / T1), np.sqrt(sd**2 + e**2 / T2)
    out = implied_event_move(iv1, T1, iv2, T2)
    assert out["consistent"] and out["event_sd"] == pytest.approx(e, rel=1e-9)
    assert out["diffusive_vol"] == pytest.approx(sd, rel=1e-9)
    assert implied_event_move(0.2, T1, 0.4, T2)["consistent"] is False
    assert 0 < prob_above(100, 110, 0.5, 0.02, 0, 0.3) < 0.5


def test_breeden_litzenberger_recovers_lognormal_mass():
    K = np.linspace(40, 200, 801)
    C = bs_price(100, K, 0.5, 0.0, 0.0, 0.25, "C")
    k, dens = breeden_litzenberger(K, C, 0.0, 0.5)
    mass = np.trapezoid(dens, k)
    assert mass == pytest.approx(1.0, abs=0.01)


def test_parity_arbitrage_uses_bid_ask():
    fair_c, fair_p = float(bs_price(100, 100, 0.25, 0.04, 0, 0.3, "C")), float(bs_price(100, 100, 0.25, 0.04, 0, 0.3, "P"))
    res = parity_arbitrage(fair_c - 0.05, fair_c + 0.05, fair_p - 0.05, fair_p + 0.05, 99.99, 100.01, 100, 0.25, 0.04)
    assert not res["violation"]  # fair mids are not an arbitrage once spreads are paid
    res2 = parity_arbitrage(fair_c + 1.0, fair_c + 1.1, fair_p - 0.05, fair_p + 0.05, 99.99, 100.01, 100, 0.25, 0.04)
    assert res2["violation"] and res2["conversion_profit"] > 0


def test_strategy_payoffs():
    st = straddle(100, 0.1, 3.0, 2.5)
    a = st.analyze(100)
    assert a["net_premium"] == pytest.approx(550)
    assert a["max_loss"] == pytest.approx(-550) and a["max_profit"] == float("inf")
    assert sorted(round(b, 1) for b in a["breakevens"]) == [94.5, 105.5]
    v = vertical("C", 100, 110, 0.1, 4.0, 1.0)
    av = v.analyze(100)
    assert av["max_loss"] == pytest.approx(-300) and av["max_profit"] == pytest.approx(700)
    ic = iron_condor(90, 95, 105, 110, 0.1, (0.5, 1.5, 1.5, 0.5))
    ai = ic.analyze(100)
    assert ai["max_profit"] == pytest.approx(200) and ai["max_loss"] == pytest.approx(-300)
    naked = long_call(100, 0.1, 2.0, qty=-1)
    assert naked.analyze(100)["max_loss"] == float("-inf")  # unlimited risk reported, never capped
    cc = covered_call(100, 105, 0.1, 2.0)
    assert cc.greeks(100, 0.02, 0, 0.2)["delta"] < 100
    g = st.greeks(100, 0.02, 0, 0.2)
    assert abs(g["delta"]) < 20 and g["gamma"] > 0 and g["theta_per_day"] < 0


def test_chain_analytics_on_simulated_chain():
    ch = simulated_chain(100.0, "2024-03-01", r=0.04, base_iv=0.2, skew=-0.15, event_move=0.06, event_before="2024-03-10")
    e = enrich_chain(ch, r=0.04)
    ok = e["iv_reliable"]
    assert ok.mean() > 0.3
    liquid = ok & (e["mid"] >= 0.5)  # cheap options: 1-cent rounding moves IV a lot (tiny vega)
    assert np.allclose(e.loc[liquid, "iv_mid"], e.loc[liquid, "true_iv"], atol=0.02)
    tick = e[(e["bid"] == 0)]
    assert len(tick) and not tick["iv_reliable"].any()  # min-tick quotes are flagged, not trusted
    assert (e.loc[ok, "iv_bid"] <= e.loc[ok, "iv_ask"] + 1e-12).all()
    ts = atm_term_structure(e)
    # event on Mar 10: the 7-day expiry (Mar 8) excludes it; the 14-day expiry carries the bump
    assert ts["atm_iv"].iloc[1] > ts["atm_iv"].iloc[0] and ts["atm_iv"].iloc[1] > ts["atm_iv"].iloc[-1]
    sk = skew_25d(e)
    assert (sk["risk_reversal_25d"] > 0).all()  # negative skew: puts richer than calls
    assert arbitrage_checks(e)["butterfly_violations"] >= 0


def test_option_execution_is_conservative():
    cm = OptionCostModel()
    assert cm.fill_price(1.0, 1.2, +1) == pytest.approx(1.2)  # buy at ask
    assert cm.fill_price(1.0, 1.2, -1) == pytest.approx(1.0)  # sell at bid
    assert cm.fill_price(0.0, 0.05, -1) is None  # cannot sell into a zero bid
    assert OptionCostModel(spread_capture=1.0).fill_price(1.0, 1.2, +1) == pytest.approx(1.1)


def test_option_trade_simulation_timing_and_settlement():
    days = pd.bdate_range("2024-03-01", "2024-03-15")
    fixed = [float(k) for k in range(80, 121, 5)]  # listed strikes do not move with the spot
    quotes = pd.concat([simulated_chain(100.0 + i * 0.5, str(d.date()), expiries_days=(14, 30), strikes=fixed)
                        for i, d in enumerate(days)])
    quotes["expiration"] = pd.Timestamp("2024-03-15")  # force a common expiry for the test
    closes = pd.Series(100.0 + np.arange(len(days)) * 0.5, index=days)
    res = simulate_option_trade(quotes, closes, days[0], [LegSpec("C", "atm", 1), LegSpec("P", "atm", 1)], OptionCostModel())
    assert res["status"] == "FILLED"
    assert res["entry_session"] == days[1]  # never fills on the signal session's snapshot
    assert res["exit_kind"] == "expiry" and any("exercised" in n for n in res["notes"])
    assert res["pnl_net"] < res["pnl_gross"]  # commissions
    hold = simulate_option_trade(quotes, closes, days[0], [LegSpec("C", "atm", 1)], OptionCostModel(), hold_sessions=3)
    assert hold["exit_kind"] == "rule" and hold["exit_session"] == days[4]
    gap = quotes[~((pd.to_datetime(quotes["session"]) == days[4]) & (quotes["strike"] == hold["legs"][0]["strike"]))]
    delayed = simulate_option_trade(gap, closes, days[0], [LegSpec("C", "atm", 1)], OptionCostModel(), hold_sessions=3)
    assert delayed["exit_session"] == days[5] and any("delayed" in n for n in delayed["notes"])
    rej = simulate_option_trade(quotes, closes, days[0], [LegSpec("C", "atm", 1000)], OptionCostModel())
    assert rej["status"] == "LIQUIDITY_REJECT"
