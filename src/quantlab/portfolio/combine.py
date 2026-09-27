"""Combining validated signals and constructing portfolios.

Signal combination (conceptually E[return | signals, regime]):

* each signal's historical edge is **shrunk toward zero** (skeptical Bayesian prior; see
  :func:`quantlab.stats.signal.normal_shrinkage_posterior`), then multiplied by a **decay factor**
  from its stability assessment and an optional **regime multiplier** estimated from regime
  performance;
* signals are weighted by Σ⁻¹μ using a *shrunk* covariance of their return streams. Correlated
  signals therefore share weight: two identical signals receive the weight of one, never double;
* the output is a combined expected return, volatility and a Normal predictive distribution, with
  the effective number of independent signals reported.

Portfolio construction: fractional-Kelly mean–variance with explicit caps (long-only by default),
i.e. maximise  μᵀw − ½ (1/f) wᵀΣw  subject to 0 ≤ wᵢ ≤ cap and Σ|w| ≤ gross cap. The fraction f
(default ¼) and caps guard against estimation error; historical CAGR is never the objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from quantlab.risk.metrics import shrunk_covariance
from quantlab.stats.signal import normal_shrinkage_posterior

DECAY_FACTORS = {"STRENGTHENED": 1.0, "STABLE": 1.0, "WEAKENED": 0.5, "DISAPPEARED": 0.0,
                 "NO_EDGE_IN_EITHER_HALF": 0.0, "INSUFFICIENT_DATA": 0.5, None: 0.5}


@dataclass
class SignalInput:
    signal_id: str
    mean: float  # historical mean return per period (net)
    se: float  # standard error of that mean
    stability: str | None = None
    regime_multiplier: float = 1.0
    active: bool = True  # is the signal firing now? (inactive → contributes nothing: NO TRADE)


def adjusted_edges(signals: list[SignalInput]) -> pd.DataFrame:
    rows = []
    for s in signals:
        post = normal_shrinkage_posterior(s.mean, s.se)
        decay = DECAY_FACTORS.get(s.stability, 0.5)
        mu = (post["post_mean"] if np.isfinite(post["post_mean"]) else 0.0) * decay * s.regime_multiplier
        rows.append({"signal_id": s.signal_id, "raw_mean": s.mean, "shrunk_mean": post["post_mean"], "decay": decay,
                     "regime_multiplier": s.regime_multiplier, "active": s.active, "adjusted_mean": mu if s.active else 0.0,
                     "p_positive": post["p_positive"]})
    return pd.DataFrame(rows).set_index("signal_id")


def effective_number(corr: pd.DataFrame) -> float:
    """Participation ratio of correlation eigenvalues: N for independent signals, →1 if identical."""
    ev = np.clip(np.linalg.eigvalsh(corr.to_numpy()), 0, None)
    return float(ev.sum() ** 2 / (ev**2).sum()) if ev.sum() > 0 else 0.0


def combine_signals(signals: list[SignalInput], returns: pd.DataFrame, long_only: bool = True) -> dict:
    """Combine signals given their historical return streams (columns = signal ids)."""
    edges = adjusted_edges(signals)
    ids = [s.signal_id for s in signals if s.signal_id in returns.columns]
    if not ids:
        return {"status": "NO_SIGNALS"}
    cov = shrunk_covariance(returns[ids])
    mu = edges.loc[ids, "adjusted_mean"].to_numpy()
    if not np.any(mu > 0):
        return {"status": "NO_TRADE", "reason": "no signal has a positive adjusted edge right now", "edges": edges}
    w = np.linalg.solve(cov.to_numpy() + 1e-12 * np.eye(len(ids)), mu)
    if long_only:
        w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return {"status": "NO_TRADE", "reason": "combination has no positive weights", "edges": edges}
    w = w / w.sum()
    comb_mu = float(w @ mu)
    comb_sd = float(np.sqrt(w @ cov.to_numpy() @ w))
    corr = returns[ids].corr()
    return {"status": "OK", "weights": dict(zip(ids, w.tolist())), "expected_return": comb_mu, "volatility": comb_sd,
            "p_positive": float(norm.sf(0, comb_mu, comb_sd)) if comb_sd > 0 else float(comb_mu > 0),
            "predictive_quantiles": {q: float(norm.ppf(q, comb_mu, comb_sd)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "effective_independent_signals": effective_number(corr), "n_signals": len(ids), "edges": edges,
            "caveat": "Normal predictive distribution from historical moments; tails are likely fatter."}


def fractional_kelly_portfolio(mu: pd.Series, cov: pd.DataFrame, fraction: float = 0.25, cap: float = 0.2,
                               gross_cap: float = 1.0, long_only: bool = True) -> pd.Series:
    """Maximise μᵀw − ½(1/f) wᵀΣw under box and gross-exposure constraints."""
    ids = list(mu.index)
    m, S = mu.to_numpy(float), cov.loc[ids, ids].to_numpy(float)
    if not np.any(m > 0) and long_only:
        return pd.Series(0.0, index=ids)
    obj = lambda w: -(m @ w - 0.5 / fraction * w @ S @ w)
    bounds = [(0.0 if long_only else -cap, cap)] * len(ids)
    cons = [{"type": "ineq", "fun": lambda w: gross_cap - np.abs(w).sum()}]
    res = minimize(obj, x0=np.zeros(len(ids)), bounds=bounds, constraints=cons, method="SLSQP")
    w = np.where(np.abs(res.x) < 1e-8, 0.0, res.x)
    return pd.Series(w, index=ids)
