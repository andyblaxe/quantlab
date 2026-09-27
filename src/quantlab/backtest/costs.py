"""Transaction-cost model for equities/ETFs (options costs live in :mod:`quantlab.options.execution`).

Cost of a trade of notional ``V`` (dollars, absolute) in a security with average daily dollar volume
``ADV`` and daily return volatility ``sigma``:

    spread      = half_spread_bps / 1e4 * V            (crossing half the quoted spread)
    slippage    = slippage_bps / 1e4 * V               (adverse selection / timing, fixed)
    impact      = impact_coef * sigma * sqrt(V / ADV) * V   (square-root market-impact law)
    commission  = max(commission_min, per_share * shares) + commission_pct * V

and the **fill price is moved against us** by (spread + slippage + impact) / V. Participation is
capped at ``max_participation`` of ADV; the engines partially fill larger orders and record the
shortfall rather than assuming unlimited liquidity.

Defaults are deliberately conservative for a retail account. They are assumptions, not
measurements; when quote data is available, spreads should be estimated from it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CostModel:
    name: str = "retail_etf"
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    impact_coef: float = 0.1
    commission_per_share: float = 0.0
    commission_min: float = 0.0
    commission_pct: float = 0.0
    max_participation: float = 0.01  # max fraction of ADV per trade
    short_borrow_bps_annual: float = 50.0
    allow_short: bool = False

    def scaled(self, k: float) -> "CostModel":
        """All frictions multiplied by k (for cost-stress tests). Liquidity limits unchanged."""
        return replace(self, name=f"{self.name}_x{k:g}", half_spread_bps=self.half_spread_bps * k,
                       slippage_bps=self.slippage_bps * k, impact_coef=self.impact_coef * k,
                       commission_per_share=self.commission_per_share * k, commission_min=self.commission_min * k,
                       commission_pct=self.commission_pct * k, short_borrow_bps_annual=self.short_borrow_bps_annual * k)

    def zero(self) -> "CostModel":
        """Frictionless copy with the same feasibility limits (used for GROSS results)."""
        return replace(self, name=f"{self.name}_gross", half_spread_bps=0.0, slippage_bps=0.0, impact_coef=0.0,
                       commission_per_share=0.0, commission_min=0.0, commission_pct=0.0,
                       short_borrow_bps_annual=0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    # ---------------------------------------------------------------------------------------------
    def price_impact_frac(self, notional, adv_dollar, daily_vol):
        """Fractional adverse price move (spread + slippage + impact) for trades of given notional."""
        notional = np.abs(np.asarray(notional, dtype=float))
        adv = np.asarray(adv_dollar, dtype=float)
        vol = np.nan_to_num(np.asarray(daily_vol, dtype=float), nan=0.02)
        with np.errstate(divide="ignore", invalid="ignore"):
            part = np.where(adv > 0, notional / adv, np.inf)
        impact = self.impact_coef * vol * np.sqrt(np.minimum(part, 1.0))
        frac = (self.half_spread_bps + self.slippage_bps) / 1e4 + impact
        return np.where(notional > 0, frac, 0.0)

    def commission(self, notional, shares):
        notional = np.abs(np.asarray(notional, dtype=float))
        shares = np.abs(np.asarray(shares, dtype=float))
        per = np.maximum(self.commission_min, self.commission_per_share * shares) + self.commission_pct * notional
        return np.where(notional > 0, per, 0.0)

    def round_trip_frac(self, notional, price, adv_dollar, daily_vol):
        """Total round-trip cost as a fraction of notional (entry + exit), for event-trade engines."""
        notional = np.asarray(notional, dtype=float)
        shares = np.abs(notional) / np.asarray(price, dtype=float)
        one_way = self.price_impact_frac(notional, adv_dollar, daily_vol)
        comm = self.commission(notional, shares)
        with np.errstate(divide="ignore", invalid="ignore"):
            comm_frac = np.where(np.abs(notional) > 0, comm / np.abs(notional), 0.0)
        return 2 * (one_way + comm_frac)


COST_PROFILES: dict[str, CostModel] = {
    # Liquid ETFs (SPY/QQQ/sector SPDRs) at a zero-commission retail broker.
    "retail_etf": CostModel("retail_etf", half_spread_bps=1.0, slippage_bps=2.0, impact_coef=0.1),
    # S&P 500 single names.
    "retail_largecap": CostModel("retail_largecap", half_spread_bps=3.0, slippage_bps=3.0, impact_coef=0.1),
    # Small caps: wide spreads, thin books.
    "retail_smallcap": CostModel("retail_smallcap", half_spread_bps=20.0, slippage_bps=10.0, impact_coef=0.2,
                                 max_participation=0.005),
    # Per-share commission broker (e.g. $0.005/share, $1 minimum) — punishing for tiny accounts.
    "per_share_broker": CostModel("per_share_broker", half_spread_bps=2.0, slippage_bps=2.0, impact_coef=0.1,
                                  commission_per_share=0.005, commission_min=1.0),
    "frictionless": CostModel("frictionless", 0.0, 0.0, 0.0),
}


def get_cost_model(name: str) -> CostModel:
    if name not in COST_PROFILES:
        raise KeyError(f"unknown cost profile {name!r}; known: {sorted(COST_PROFILES)}")
    return COST_PROFILES[name]
