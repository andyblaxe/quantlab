"""Multi-leg option positions: payoffs, P&L curves, breakevens, max profit/loss, aggregate Greeks.

Quantities are in contracts (positive = long, negative = short); the contract multiplier (default
100) converts per-share values to dollars. "Unlimited" risk is reported as ``inf``, never capped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantlab.options.pricing import bs_greeks, bs_price


@dataclass(frozen=True)
class Leg:
    right: str  # "C" | "P"
    strike: float
    expiry_years: float
    qty: int  # contracts, + long / − short
    premium: float  # per share, paid (+) for longs / received for shorts — the fill price
    multiplier: int = 100


@dataclass
class OptionPosition:
    legs: list[Leg]
    stock_qty: int = 0  # shares (e.g. covered call)
    stock_cost: float = 0.0
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def net_premium(self) -> float:
        """Dollars paid (+) or received (−) to open (excluding commissions)."""
        return float(sum(l.qty * l.premium * l.multiplier for l in self.legs) + self.stock_qty * self.stock_cost)

    def payoff_at_expiry(self, S_T) -> np.ndarray:
        """Dollar value of the position at expiry (before subtracting the opening premium)."""
        S_T = np.asarray(S_T, dtype=float)
        v = self.stock_qty * S_T
        for l in self.legs:
            intr = np.maximum(S_T - l.strike, 0) if l.right == "C" else np.maximum(l.strike - S_T, 0)
            v = v + l.qty * l.multiplier * intr
        return v

    def pnl_at_expiry(self, S_T) -> np.ndarray:
        return self.payoff_at_expiry(S_T) - self.net_premium

    def value(self, S, t_elapsed, r, q, sigma) -> float:
        """Model (BS) mark of the position after ``t_elapsed`` years at spot S and flat vol sigma."""
        v = self.stock_qty * S
        for l in self.legs:
            T = max(l.expiry_years - t_elapsed, 0.0)
            v += l.qty * l.multiplier * float(bs_price(S, l.strike, T, r, q, sigma, l.right))
        return float(v)

    def greeks(self, S, r, q, sigma, t_elapsed: float = 0.0) -> dict[str, float]:
        out = {"delta": float(self.stock_qty), "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for l in self.legs:
            T = max(l.expiry_years - t_elapsed, 1e-9)
            g = bs_greeks(S, l.strike, T, r, q, sigma, l.right)
            for k in out:
                out[k] += l.qty * l.multiplier * float(g[k])
        out["vega_per_vol_point"] = out["vega"] / 100
        out["theta_per_day"] = out["theta"] / 365
        return out

    def analyze(self, S: float, width: float = 0.6, n: int = 2001) -> dict:
        grid = np.linspace(max(S * (1 - width), 0.0), S * (1 + width), n)
        pnl = self.pnl_at_expiry(grid)
        # slope beyond the grid decides unbounded outcomes
        top_slope = self.stock_qty + sum(l.qty * l.multiplier for l in self.legs if l.right == "C")
        sign = np.sign(pnl)
        crossings = np.where(np.diff(sign) != 0)[0]
        bes = [float(grid[i] - pnl[i] * (grid[i + 1] - grid[i]) / (pnl[i + 1] - pnl[i])) for i in crossings]
        max_profit = float("inf") if top_slope > 0 else float(max(pnl.max(), self.pnl_at_expiry(0.0)))
        max_loss = float("-inf") if top_slope < 0 else float(min(pnl.min(), self.pnl_at_expiry(0.0)))
        return {"net_premium": self.net_premium, "breakevens": bes, "max_profit": max_profit, "max_loss": max_loss,
                "grid": grid, "pnl": pnl}


def _leg(right, K, T, qty, premium, m=100):
    return Leg(right, float(K), float(T), int(qty), float(premium), m)


def long_call(K, T, prem, qty=1): return OptionPosition([_leg("C", K, T, qty, prem)], name="long call")
def long_put(K, T, prem, qty=1): return OptionPosition([_leg("P", K, T, qty, prem)], name="long put")


def straddle(K, T, call_prem, put_prem, qty=1):
    return OptionPosition([_leg("C", K, T, qty, call_prem), _leg("P", K, T, qty, put_prem)], name="long straddle")


def strangle(K_put, K_call, T, put_prem, call_prem, qty=1):
    return OptionPosition([_leg("P", K_put, T, qty, put_prem), _leg("C", K_call, T, qty, call_prem)], name="long strangle")


def vertical(right, K_long, K_short, T, long_prem, short_prem, qty=1):
    return OptionPosition([_leg(right, K_long, T, qty, long_prem), _leg(right, K_short, T, -qty, short_prem)],
                          name=f"{'call' if right == 'C' else 'put'} vertical {K_long}/{K_short}")


def iron_condor(Kp_long, Kp_short, Kc_short, Kc_long, T, prems: tuple[float, float, float, float], qty=1):
    pl, ps, cs, cl = prems
    return OptionPosition([_leg("P", Kp_long, T, qty, pl), _leg("P", Kp_short, T, -qty, ps),
                           _leg("C", Kc_short, T, -qty, cs), _leg("C", Kc_long, T, qty, cl)], name="iron condor")


def butterfly(right, K1, K2, K3, T, prems: tuple[float, float, float], qty=1):
    a, b, c = prems
    return OptionPosition([_leg(right, K1, T, qty, a), _leg(right, K2, T, -2 * qty, b), _leg(right, K3, T, qty, c)],
                          name="butterfly")


def covered_call(S0, K, T, call_prem, shares=100):
    return OptionPosition([_leg("C", K, T, -(shares // 100), call_prem)], stock_qty=shares, stock_cost=S0, name="covered call")
