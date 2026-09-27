"""Position sizing: fractional Kelly with estimation-uncertainty haircuts and hard caps.

Full Kelly maximises long-run geometric growth *if the edge is known exactly*. It is never known:
an overestimated edge at full Kelly leads to large drawdowns and, with fat tails, ruin. Therefore:

1. The edge used for sizing is the **lower confidence bound** of the mean trade return (or a
   Bayesian-shrunk mean), not the point estimate.
2. The Kelly fraction is computed numerically from the empirical trade distribution
   (maximise mean log growth), which respects skew and fat tails, instead of μ/σ².
3. Only a **fraction** of Kelly is used (default ¼), then capped by a hard per-position limit.
4. A non-positive conservative edge ⇒ size zero. "No trade" is a valid output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def kelly_gaussian(mu: float, sigma: float) -> float:
    """Continuous-time Kelly fraction μ/σ² (reference only)."""
    return float(mu / sigma**2) if sigma > 0 else 0.0


def kelly_empirical(trade_returns, max_f: float = 10.0) -> float:
    """Fraction f maximising mean log(1 + f r) over observed trade returns (0 if no positive edge)."""
    r = pd.Series(trade_returns, dtype=float).dropna().to_numpy()
    if len(r) < 10 or r.mean() <= 0:
        return 0.0
    worst = r.min()
    upper = min(max_f, 0.999 / -worst) if worst < 0 else max_f  # keep 1 + f r > 0 for every observed trade
    res = minimize_scalar(lambda f: -np.mean(np.log1p(f * r)), bounds=(0.0, upper), method="bounded")
    return float(max(res.x, 0.0))


@dataclass(frozen=True)
class SizingPolicy:
    kelly_fraction: float = 0.25  # never full Kelly
    max_position_fraction: float = 0.20  # hard cap of equity per position
    ci_z: float = 1.645  # one-sided 95% lower bound on the mean edge
    min_trades: int = 30


def recommended_fraction(trade_returns, policy: SizingPolicy = SizingPolicy()) -> dict:
    """Conservative fraction of equity to commit per trade, with the reasoning attached."""
    r = pd.Series(trade_returns, dtype=float).dropna()
    n = len(r)
    if n < policy.min_trades:
        return {"fraction": 0.0, "reason": f"only {n} trades (< {policy.min_trades}); edge too uncertain to size"}
    se = r.std(ddof=1) / np.sqrt(n)
    lower = r.mean() - policy.ci_z * se
    if lower <= 0:
        return {"fraction": 0.0, "reason": f"lower confidence bound of mean trade return is {lower:.4%} ≤ 0",
                "mean": float(r.mean()), "lower_bound": float(lower)}
    shifted = r - (r.mean() - lower)  # same distribution shape, edge reduced to its lower bound
    fk = kelly_empirical(shifted)
    frac = min(policy.kelly_fraction * fk, policy.max_position_fraction)
    return {"fraction": float(frac), "full_kelly_on_lower_bound": fk, "kelly_fraction_used": policy.kelly_fraction,
            "capped": bool(policy.kelly_fraction * fk > policy.max_position_fraction), "mean": float(r.mean()),
            "lower_bound": float(lower), "reason": "fractional Kelly on the lower-bound edge, capped"}


def growth_rate(trade_returns, f: float) -> float:
    r = pd.Series(trade_returns, dtype=float).dropna().to_numpy()
    x = 1 + f * r
    return float(np.mean(np.log(x))) if (x > 0).all() else float("-inf")


def ruin_probability_gaussian(mu: float, sigma: float, f: float, ruin_level: float = 0.5, horizon: float = np.inf) -> float:
    """Probability that log-equity ever falls to log(ruin_level) under GBM with per-trade drift/vol.

    For infinite horizon with positive log drift g = f μ − f²σ²/2:  P = ruin_level^(2g/(fσ)²).
    Rough analytic cross-check for the Monte Carlo estimate; ignores fat tails (optimistic).
    """
    g = f * mu - 0.5 * (f * sigma) ** 2
    s2 = (f * sigma) ** 2
    if s2 == 0:
        return 0.0 if g >= 0 else 1.0
    if g <= 0:
        return 1.0
    return float(ruin_level ** (2 * g / s2))
