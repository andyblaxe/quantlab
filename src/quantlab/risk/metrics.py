"""Portfolio risk measurement: VaR, expected shortfall, volatility, beta, concentration, exposures.

This module measures; it does not decide. Decisions belong to :mod:`quantlab.risk.gate`, which
never imports signal or strategy code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

ANN = 252


def historical_var_es(returns, level: float = 0.95) -> dict:
    """One-period historical VaR and expected shortfall as positive loss fractions."""
    r = pd.Series(returns, dtype=float).dropna().to_numpy()
    if len(r) < 20:
        return {"var": np.nan, "es": np.nan, "n": len(r), "method": "historical"}
    q = np.quantile(r, 1 - level)
    return {"var": float(-q), "es": float(-r[r <= q].mean()), "n": int(len(r)), "method": "historical", "level": level}


def parametric_var_es(mu: float, sigma: float, level: float = 0.95) -> dict:
    z = norm.ppf(1 - level)
    return {"var": float(-(mu + z * sigma)), "es": float(-(mu - sigma * norm.pdf(z) / (1 - level))),
            "method": "gaussian", "level": level}


def cornish_fisher_var(mu: float, sigma: float, skew: float, ex_kurt: float, level: float = 0.95) -> float:
    """VaR with a skew/kurtosis correction (fat tails raise it; used as a cross-check)."""
    z = norm.ppf(1 - level)
    zcf = z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * ex_kurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36
    return float(-(mu + zcf * sigma))


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame, annualize: bool = True) -> float:
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    v = float(np.sqrt(max(w @ cov.to_numpy() @ w, 0.0)))
    return v * np.sqrt(ANN) if annualize else v


def shrunk_covariance(returns: pd.DataFrame, shrink: float | None = None) -> pd.DataFrame:
    """Ledoit–Wolf style shrinkage toward a constant-correlation target (simple, stable)."""
    X = returns.dropna(how="all").fillna(0.0)
    S = X.cov().to_numpy()
    sd = np.sqrt(np.diag(S))
    corr = S / np.outer(sd, sd)
    n = corr.shape[0]
    rbar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, sd**2)
    if shrink is None:
        shrink = float(min(1.0, max(0.0, 10.0 / max(len(X), 1))))  # more shrinkage with less data
    out = shrink * F + (1 - shrink) * S
    return pd.DataFrame(out, index=returns.columns, columns=returns.columns)


def beta(asset: pd.Series, market: pd.Series) -> float:
    df = pd.concat([asset, market], axis=1).dropna()
    if len(df) < 20:
        return np.nan
    c = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    return float(c[0, 1] / c[1, 1])


@dataclass
class Position:
    symbol: str
    quantity: float  # shares, or contracts for options
    price: float
    kind: str = "equity"  # equity | etf | option
    sector: str = "UNKNOWN"
    beta: float = 1.0
    multiplier: float = 1.0
    greeks: dict = field(default_factory=dict)  # per-position (already × qty × multiplier)
    event_risk: bool = False  # e.g. earnings before next session
    adv_dollar: float = np.nan

    @property
    def market_value(self) -> float:
        return self.quantity * self.price * self.multiplier


@dataclass
class PortfolioState:
    equity: float
    cash: float
    positions: list[Position] = field(default_factory=list)
    peak_equity: float | None = None
    daily_returns: pd.Series | None = None  # recent portfolio returns for VaR

    def exposures(self) -> dict:
        gross = sum(abs(p.market_value) for p in self.positions)
        net = sum(p.market_value for p in self.positions)
        by_sector: dict[str, float] = {}
        for p in self.positions:
            by_sector[p.sector] = by_sector.get(p.sector, 0.0) + p.market_value
        largest = max((abs(p.market_value) for p in self.positions), default=0.0)
        greeks = {k: sum(p.greeks.get(k, 0.0) for p in self.positions) for k in ("delta", "gamma", "theta", "vega")}
        eq = self.equity if self.equity > 0 else np.nan
        return {
            "gross_leverage": gross / eq, "net_leverage": net / eq,
            "beta_exposure": sum(p.market_value * p.beta for p in self.positions) / eq,
            "largest_position": largest / eq,
            "sector_net": {k: v / eq for k, v in by_sector.items()},
            "option_greeks": greeks,
            "event_risk_exposure": sum(abs(p.market_value) for p in self.positions if p.event_risk) / eq,
            "overnight_exposure": gross / eq,  # all positions held overnight in a daily system
            "drawdown": (self.equity / self.peak_equity - 1) if self.peak_equity else 0.0,
            "illiquid_fraction": sum(abs(p.market_value) for p in self.positions
                                     if np.isfinite(p.adv_dollar) and abs(p.market_value) > 0.05 * p.adv_dollar) / eq,
        }
