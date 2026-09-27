"""Monte Carlo capital simulation from historical trade outcomes.

Method: stationary block bootstrap of the *sequence* of per-trade net returns (blocks preserve
streaks and volatility clustering), applied to an account that risks a fixed fraction of equity per
trade. Capital-dependent frictions are modelled explicitly, because they are what separates a $100
account from a $100,000 account running the same strategy:

* a fixed per-trade dollar cost (commission minimum) is a larger *fraction* of a small trade;
* a minimum position size (one share / one option contract) makes some trades impossible —
  they are skipped, not scaled down to fictitious fractions;
* ruin: equity at or below ``ruin_fraction`` × start, or below the minimum position size.

Caveat printed with every result: resampling history cannot produce crises that are not in the
sample. The ``stressed`` variant haircuts the mean trade return (edge decay) to show fragility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantlab.stats.core import stationary_bootstrap_indices

DEFAULT_CAPITALS = (100.0, 1_000.0, 10_000.0, 100_000.0)


@dataclass(frozen=True)
class MonteCarloConfig:
    n_paths: int = 2000
    horizon_trades: int | None = None  # default: same number of trades as history
    trades_per_year: float = 50.0
    position_fraction: float = 0.2  # fraction of equity committed per trade
    fixed_cost_per_trade: float = 0.0  # dollars per round trip, on top of the per-trade % costs
    min_position_value: float = 0.0  # e.g. price of one share / one contract premium ×100
    ruin_fraction: float = 0.2  # equity <= 20% of start counts as ruin
    mean_block: float | None = None
    seed: int = 0
    stressed_haircut: float = 0.5  # stressed variant: mean trade return × (1 - haircut)

    def to_dict(self) -> dict:
        return asdict(self)


def _simulate_paths(r: np.ndarray, capital: float, cfg: MonteCarloConfig, rng: np.random.Generator) -> dict:
    n_hist = len(r)
    H = cfg.horizon_trades or n_hist
    P = cfg.n_paths
    block = cfg.mean_block or max(1.0, n_hist ** (1 / 3))
    reps = int(np.ceil(H / n_hist))
    idx = np.stack([np.concatenate([stationary_bootstrap_indices(n_hist, block, rng) for _ in range(reps)])[:H]
                    for _ in range(P)])
    R = r[idx]  # (paths, trades)
    eq = np.full(P, capital, dtype=float)
    peak = eq.copy()
    dd = np.zeros(P)
    active = np.ones(P, dtype=bool)
    ruined = np.zeros(P, dtype=bool)
    t_double = np.full(P, np.nan)
    skipped = np.zeros(P)
    for i in range(H):
        pos = eq * cfg.position_fraction
        can = active & (pos >= cfg.min_position_value) & (pos > cfg.fixed_cost_per_trade)
        skipped += active & ~can
        broke = active & (eq < cfg.min_position_value)  # cannot take any further trade
        ruined |= broke
        active &= ~broke
        can &= active
        eq = np.where(can, eq + pos * R[:, i] - cfg.fixed_cost_per_trade, eq)
        peak = np.maximum(peak, eq)
        dd = np.minimum(dd, eq / peak - 1)
        newly = np.isnan(t_double) & (eq >= 2 * capital)
        t_double[newly] = (i + 1) / cfg.trades_per_year
        hit = active & (eq <= cfg.ruin_fraction * capital)
        ruined |= hit
        active &= ~hit
    ending = eq
    maxdd = dd
    years = H / cfg.trades_per_year
    growth = np.clip(ending / capital, 1e-12, None)
    cagr = growth ** (1 / years) - 1
    ret = ending / capital - 1
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    return {
        "capital": capital,
        "horizon_trades": H,
        "horizon_years": years,
        "ending_capital_quantiles": dict(zip(["p05", "p25", "p50", "p75", "p95"], np.quantile(ending, q).tolist())),
        "ending_capital_mean": float(ending.mean()),
        "cagr_quantiles": dict(zip(["p05", "p25", "p50", "p75", "p95"], np.quantile(cagr, q).tolist())),
        "max_drawdown_quantiles": dict(zip(["p05", "p25", "p50", "p75", "p95"], np.quantile(maxdd, q).tolist())),
        "p_loss": float(np.mean(ret < 0)),
        "p_loss_10": float(np.mean(ret <= -0.10)),
        "p_loss_25": float(np.mean(ret <= -0.25)),
        "p_loss_50": float(np.mean(ret <= -0.50)),
        "p_ruin": float(ruined.mean()),
        "p_double": float(np.mean(ending >= 2 * capital)),
        "time_to_double_years_quantiles": (
            dict(zip(["p25", "p50", "p75"], np.nanquantile(t_double, [0.25, 0.5, 0.75]).tolist()))
            if np.isfinite(t_double).any() else None),
        "avg_trades_skipped": float(skipped.mean()),
        "ending_capital_sample": np.quantile(ending, np.linspace(0, 1, 101)).tolist(),
        "max_drawdown_sample": np.quantile(maxdd, np.linspace(0, 1, 101)).tolist(),
    }


def monte_carlo(trade_returns: pd.Series | np.ndarray, cfg: MonteCarloConfig = MonteCarloConfig(),
                capitals: tuple[float, ...] = DEFAULT_CAPITALS) -> dict:
    """Simulate each starting capital, for the historical and the stressed (edge-decay) trade set."""
    r = pd.Series(trade_returns, dtype=float).dropna().to_numpy()
    if len(r) < 10:
        return {"status": "INSUFFICIENT_DATA", "n_trades": int(len(r))}
    stressed = r - cfg.stressed_haircut * r.mean() if r.mean() > 0 else r.copy()
    out = {"status": "OK", "n_trades": int(len(r)), "config": cfg.to_dict(),
           "caveat": "Resampled from history: cannot generate crises absent from the sample. "
                     "Quantiles, not the best path, describe expectations.",
           "historical": {}, "stressed": {}}
    for cap in capitals:
        rng = np.random.default_rng(cfg.seed)
        out["historical"][str(int(cap))] = _simulate_paths(r, cap, cfg, rng)
        rng = np.random.default_rng(cfg.seed)
        out["stressed"][str(int(cap))] = _simulate_paths(stressed, cap, cfg, rng)
    return out
