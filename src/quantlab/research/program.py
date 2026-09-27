"""Research program v1 — the first ten pre-registered hypotheses (see PLAN.md §12).

These specs are committed to git *before any real data has been examined* (git history is the
timestamp). Registering them in a local registry (``quantlab research register-program v1``) hashes
them; any later edit produces a different hash and therefore a *new* hypothesis, visible in the
multiple-testing count.

Every hypothesis states how it could be false. Several were proposed by the project owner; they are
framed here so that they can be rejected.

Universes use ETFs to limit survivorship bias (ETFs that existed through the sample, with
inception dates enforced by the data itself: an ETF has no bars before it launched).
"""

from __future__ import annotations

from quantlab.research.hypotheses import HypothesisSpec, Mechanism

UNIVERSES_V1: dict[str, list[str]] = {
    "spy": ["SPY"],
    "us_index_etfs": ["SPY", "QQQ", "IWM", "DIA"],
    # the nine original Select Sector SPDRs (Dec 1998); XLRE (2015) and XLC (2018) excluded for history
    "sector_spdrs": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
    "multi_asset_etfs": ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
}
# Every universe must declare its kind (see research.data.universe_flags_from_kinds); an undeclared
# universe is treated as survivorship-biased and its results cannot be promoted.
UNIVERSE_KINDS_V1: dict[str, str] = {name: "etf" for name in UNIVERSES_V1}
MARKET_SYMBOL_V1 = "SPY"
MACRO_V1 = {"VIX": "cboe:VIX", "VIX3M": "cboe:VIX3M"}

_ZRET = "zret(h=3,vol_n=63)"


def program_v1() -> list[HypothesisSpec]:
    return [
        HypothesisSpec(
            name="H1 time-series momentum (multi-asset ETFs)",
            question="Does an asset's own 12-1 month return predict whether holding it beats holding everything?",
            statement="Holding each multi-asset ETF only while its 12-1 month return is positive beats equal-weight "
                      "buy-and-hold on a risk-adjusted, after-cost basis.",
            mechanism=Mechanism.BEHAVIORAL,
            rationale="Documented across asset classes (Moskowitz, Ooi & Pedersen 2012); explanations include "
                      "under-reaction and slow-moving capital. Could be false if the effect decayed after publication "
                      "or is driven only by 2008.",
            family="trend", universe="multi_asset_etfs", strategy="state_position",
            params={"conditions": [["mom(lookback=252,skip=21)", ">", 0.0]], "fixed_slots": True},
            param_neighbors=[{"conditions": [["mom(lookback=189,skip=21)", ">", 0.0]], "fixed_slots": True},
                             {"conditions": [["mom(lookback=315,skip=21)", ">", 0.0]], "fixed_slots": True}],
            direction="long", holding_period=21, execution="next_close",
            primary_metric="excess_vs_buy_and_hold", cost_profile="retail_etf", instrument_type="etf",
        ),
        HypothesisSpec(
            name="H2 SPY 200-day moving-average filter",
            question="Does holding SPY only above its 200-day average improve on buy-and-hold after costs?",
            statement="SPY held only while close > SMA(200), else cash, has higher net returns than buy-and-hold.",
            mechanism=Mechanism.BEHAVIORAL,
            rationale="Owner-proposed and widely claimed. Expected to reduce drawdowns; whether it adds return after "
                      "whipsaw costs is the question. Likely false on returns, possibly true on drawdown.",
            family="trend", universe="spy", strategy="state_position",
            params={"conditions": [["sma_ratio(n=200)", ">", 0.0]]},
            param_neighbors=[{"conditions": [["sma_ratio(n=150)", ">", 0.0]]},
                             {"conditions": [["sma_ratio(n=250)", ">", 0.0]]}],
            direction="long", holding_period=21, execution="next_close",
            primary_metric="excess_vs_buy_and_hold", cost_profile="retail_etf", instrument_type="etf",
        ),
        HypothesisSpec(
            name="H3 SPY golden/death cross (50/200)",
            question="Does the 50/200-day moving-average cross add value over buy-and-hold?",
            statement="SPY held only while SMA(50) > SMA(200) has higher net returns than buy-and-hold.",
            mechanism=Mechanism.BEHAVIORAL,
            rationale="Owner-proposed. Few crossings per decade ⇒ small effective sample; likely INCONCLUSIVE or "
                      "subsumed by H2. Compare with H2 in reports (correlation of daily returns).",
            family="trend", universe="spy", strategy="state_position",
            params={"conditions": [["ma_cross(fast=50,slow=200)", ">", 0.0]]},
            param_neighbors=[{"conditions": [["ma_cross(fast=40,slow=200)", ">", 0.0]]},
                             {"conditions": [["ma_cross(fast=50,slow=150)", ">", 0.0]]}],
            direction="long", holding_period=21, execution="next_close",
            primary_metric="excess_vs_buy_and_hold", cost_profile="retail_etf", instrument_type="etf",
        ),
        HypothesisSpec(
            name="H4 short-term reversal in index ETFs within an uptrend",
            question="Do sharp 3-day declines in index ETFs revert when the long-term trend is up?",
            statement="After a 3-day standardized return below -2 while above SMA(200), index ETFs earn positive "
                      "net 5-day returns, beating random entries.",
            mechanism=Mechanism.LIQUIDITY,
            rationale="Short-horizon reversal as compensation for liquidity provision; conditioning on trend avoids "
                      "catching falling knives in bear markets. Could be false if it only worked pre-2010, or only "
                      "because of a few crash rebounds (outlier test).",
            family="mean_reversion", universe="us_index_etfs", strategy="threshold_event",
            params={"conditions": [[_ZRET, "<", -2.0], ["sma_ratio(n=200)", ">", 0.0]]},
            param_neighbors=[{"conditions": [[_ZRET, "<", -1.5], ["sma_ratio(n=200)", ">", 0.0]]},
                             {"conditions": [[_ZRET, "<", -2.5], ["sma_ratio(n=200)", ">", 0.0]]},
                             {"conditions": [[_ZRET, "<", -2.0], ["sma_ratio(n=150)", ">", 0.0]]}],
            direction="long", holding_period=5, execution="next_open", cost_profile="retail_etf", instrument_type="etf",
        ),
        HypothesisSpec(
            name="H5 Bollinger bands: incremental information beyond z-scored returns",
            question="Do Bollinger-band extremes predict reversal when a plain standardized-return signal is NOT firing?",
            statement="Entries where bollinger(20,2) < -1 but the 3-day z-return is above -1 (so H4-type signal "
                      "absent) earn positive net 5-day returns beating random entries.",
            mechanism=Mechanism.UNKNOWN,
            rationale="Owner asked whether Bollinger Bands matter. BB position is close to a z-score of price versus "
                      "its mean, so it may be redundant. Testing BB only where the simpler signal is silent isolates "
                      "any incremental information. Expected: none.",
            family="mean_reversion", universe="us_index_etfs", strategy="threshold_event",
            params={"conditions": [["bollinger(n=20,k=2.0)", "<", -1.0], [_ZRET, ">", -1.0]]},
            param_neighbors=[{"conditions": [["bollinger(n=20,k=2.0)", "<", -0.8], [_ZRET, ">", -1.0]]},
                             {"conditions": [["bollinger(n=15,k=2.0)", "<", -1.0], [_ZRET, ">", -1.0]]}],
            direction="long", holding_period=5, execution="next_open", cost_profile="retail_etf", instrument_type="etf",
        ),
        HypothesisSpec(
            name="H6 SPY overnight drift is tradable after costs",
            question="Is holding SPY only overnight (close→open) profitable after paying a round trip every day?",
            statement="SPY overnight-only holding earns positive net returns after daily round-trip costs.",
            mechanism=Mechanism.STRUCTURAL,
            rationale="The overnight/intraday return asymmetry is documented. With ~250 round trips a year, costs "
                      "are the crux: this tests the executability filter. Expected: gross positive, net negative.",
            family="calendar", universe="spy", strategy="segment_hold",
            params={"segment": "overnight"},
            param_neighbors=[{"segment": "overnight", "conditions": [["sma_ratio(n=200)", ">", 0.0]]}],
            direction="long", holding_period=1, execution="next_close", cost_profile="retail_etf", instrument_type="etf",
        ),
        HypothesisSpec(
            name="H7 volatility risk premium (measurement)",
            question="Does implied variance (VIX²) exceed subsequently realized SPY variance on average?",
            statement="Mean of VIX²/10⁴ minus realized variance over the next 21 sessions is positive.",
            mechanism=Mechanism.RISK_PREMIUM,
            rationale="Foundational for any option-selling idea; insurers of crash risk should be paid. Measured "
                      "only — trading it requires historical option bid/ask (paid data).",
            family="volatility", universe="spy", strategy="measurement:vrp",
            params={"horizon": 21, "implied": "VIX"},
            direction="volatility", holding_period=21, execution="next_close", data_requirements=["bars_daily", "VIX"],
            instrument_type="index",
        ),
        HypothesisSpec(
            name="H8 VIX term-structure inversion predicts higher realized volatility (measurement)",
            question="When VIX > VIX3M, is subsequent 21-day realized SPY volatility higher?",
            statement="Forward 21-day realized vol is higher on days with VIX/VIX3M > 1 than on other days.",
            mechanism=Mechanism.VOLATILITY,
            rationale="Inversion signals acute stress; useful for risk sizing even if it carries no return edge. "
                      "Few independent episodes ⇒ inference must count episodes, not days.",
            family="volatility", universe="spy", strategy="measurement:conditional_forward",
            params={"numerator": "VIX", "denominator": "VIX3M", "threshold": 1.0, "horizon": 21},
            direction="volatility", holding_period=21, execution="next_close",
            data_requirements=["bars_daily", "VIX", "VIX3M"], instrument_type="index",
        ),
        HypothesisSpec(
            name="H9 sector ETF cross-sectional momentum",
            question="Do the strongest sectors over 12-1 months outperform over the next month?",
            statement="Holding the top third of sector SPDRs by 12-1 momentum (monthly) beats random sector picks "
                      "after costs.",
            mechanism=Mechanism.BEHAVIORAL,
            rationale="Industry momentum literature (Moskowitz & Grinblatt 1999). Only nine assets ⇒ low breadth "
                      "and power; a null result here is weak evidence either way.",
            family="cross_sectional", universe="sector_spdrs", strategy="quantile_event",
            params={"feature": "mom(lookback=252,skip=21)", "quantile": 0.34, "side": "top", "rebalance_every": 21},
            param_neighbors=[{"feature": "mom(lookback=189,skip=21)", "quantile": 0.34, "side": "top", "rebalance_every": 21},
                             {"feature": "mom(lookback=126,skip=21)", "quantile": 0.34, "side": "top", "rebalance_every": 21}],
            direction="long", holding_period=21, execution="next_close", cost_profile="retail_etf", instrument_type="etf",
        ),
        HypothesisSpec(
            name="H10 turn-of-the-month effect in SPY",
            question="Are SPY returns higher from the last trading day of a month through the third day of the next?",
            statement="Holding SPY only in the turn-of-month window earns more per day than random holding periods.",
            mechanism=Mechanism.INSTITUTIONAL,
            rationale="Attributed to month-end fund flows and payroll investment. Documented, possibly decayed. "
                      "Belongs to the calendar family, which is corrected jointly.",
            family="calendar", universe="spy", strategy="state_position",
            params={"conditions": [["turn_of_month(before=1,after=3,lead=2)", ">", 0.5]]},
            param_neighbors=[{"conditions": [["turn_of_month(before=2,after=3,lead=2)", ">", 0.5]]},
                             {"conditions": [["turn_of_month(before=1,after=2,lead=2)", ">", 0.5]]}],
            direction="long", holding_period=4, execution="next_close", cost_profile="retail_etf", instrument_type="etf",
        ),
    ]


PROGRAMS = {"v1": program_v1}
