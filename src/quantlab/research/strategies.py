"""Strategy templates: how a registered hypothesis turns features into trades.

Templates are deliberately few and generic. A hypothesis names a template and its parameters; the
evaluator produces a :class:`StrategyRun` (trades + daily gross/net returns) from a panel. No
template ever sees labels or forward returns.

Two kinds:

* **event** templates emit entry signals (+1/-1) evaluated by the trade engine with a fixed
  holding period — e.g. "enter after a -2σ day, hold 5 sessions".
* **position** templates emit target weights evaluated by the portfolio engine — e.g. "hold SPY
  while it is above its 200-day average, else cash". Holding spells are converted to trades so
  per-trade statistics exist for every strategy.

Conditions: ``[("feature spec", op, value), ...]`` with op in ``> >= < <=``; all must hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import BacktestConfig, run_portfolio_backtest
from quantlab.backtest.metrics import trades_to_daily_returns
from quantlab.backtest.trades import TradeConfig, generate_trades
from quantlab.features import get_feature

OPS: dict[str, Callable[[pd.DataFrame, float], pd.DataFrame]] = {
    ">": lambda x, v: x > v, ">=": lambda x, v: x >= v, "<": lambda x, v: x < v, "<=": lambda x, v: x <= v,
}


@dataclass
class StrategyRun:
    trades: pd.DataFrame
    daily_gross: pd.Series
    daily_net: pd.Series
    kind: str
    meta: dict = field(default_factory=dict)


def _conditions_mask(panel, conditions, cache: dict) -> pd.DataFrame | None:
    mask = None
    for spec, op, value in conditions:
        if spec not in cache:
            cache[spec] = get_feature(spec).compute(panel)
        m = OPS[op](cache[spec], float(value)) & cache[spec].notna()
        mask = m if mask is None else (mask & m.reindex_like(mask).fillna(False))
    return mask


def _restrict(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    return df.reindex(columns=symbols)


# ---------------------------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------------------------
def threshold_signal(panel, params, symbols, cache) -> pd.DataFrame:
    """Enter when every condition holds (optionally only on the first day it becomes true)."""
    mask = _conditions_mask(panel, params["conditions"], cache)
    mask = _restrict(mask, symbols).fillna(False).astype(bool)
    if params.get("on_transition", False):
        mask = mask & ~mask.shift(1, fill_value=False)
    return mask.astype(float) * float(params.get("direction", 1.0))


def quantile_signal(panel, params, symbols, cache) -> pd.DataFrame:
    """Cross-sectional: enter the top (or bottom) quantile of a feature, optionally gated by conditions."""
    spec = params["feature"]
    if spec not in cache:
        cache[spec] = get_feature(spec).compute(panel)
    x = _restrict(cache[spec], symbols)
    pct = x.rank(axis=1, pct=True)
    q = float(params.get("quantile", 0.2))
    side = params.get("side", "top")
    mask = (pct > 1 - q) if side == "top" else (pct <= q)
    mask &= x.notna()
    if params.get("conditions"):
        mask &= _restrict(_conditions_mask(panel, params["conditions"], cache), symbols).fillna(False).astype(bool)
    rebalance = int(params.get("rebalance_every", 1))
    if rebalance > 1:
        keep = np.zeros(len(mask), dtype=bool)
        keep[::rebalance] = True
        mask = mask.where(pd.Series(keep, index=mask.index), False)
    return mask.astype(float) * float(params.get("direction", 1.0))


def state_weights(panel, params, symbols, cache) -> pd.DataFrame:
    """Position template: equal-weight the symbols whose conditions hold; cash otherwise."""
    mask = _restrict(_conditions_mask(panel, params["conditions"], cache), symbols)
    valid = mask.notna().any(axis=1)
    m = mask.fillna(False).astype(bool).astype(float)
    n = m.sum(axis=1)
    if params.get("fixed_slots"):
        w = m / float(len(symbols))  # each symbol owns 1/N of capital, idle slots in cash
    else:
        w = m.div(n.replace(0, np.nan), axis=0).fillna(0.0)
    return (w * float(params.get("gross", 1.0))).where(valid, np.nan)


def constant_weights(panel, params, symbols, cache) -> pd.DataFrame:
    """Buy-and-hold benchmark as a position template."""
    return pd.DataFrame(1.0 / len(symbols), index=panel["close"].index, columns=symbols)


SIGNAL_TEMPLATES = {"threshold_event": threshold_signal, "quantile_event": quantile_signal}
POSITION_TEMPLATES = {"state_position": state_weights, "buy_and_hold": constant_weights}
SEGMENT_TEMPLATES = {"segment_hold"}


def template_kind(name: str) -> str:
    if name in SIGNAL_TEMPLATES:
        return "event"
    if name in POSITION_TEMPLATES:
        return "position"
    if name in SEGMENT_TEMPLATES:
        return "segment"
    if name.startswith("measurement:"):
        return "measurement"
    raise KeyError(f"unknown strategy template {name!r}")


def segment_run(panel, params, symbols, cost: CostModel, capital: float, max_concurrent: int) -> StrategyRun:
    """Hold only one part of each session: ``overnight`` (close t → open t+1) or ``intraday``
    (open t → close t). Every held segment is a round trip, so costs are charged per segment.

    Decision for segment starting at session t uses information up to the close of t-1 (for
    intraday) or is unconditional/calendar-based; conditions are evaluated at t-1 accordingly.
    """
    seg = params["segment"]
    cache: dict = {}
    mask = None
    if params.get("conditions"):
        mask = _restrict(_conditions_mask(panel, params["conditions"], cache), symbols).fillna(False).astype(bool)
    o, c = panel["open"][symbols], panel["close"][symbols]
    raw_o, raw_c = panel["raw_open"][symbols], panel["raw_close"][symbols]
    vol = panel["raw_volume"][symbols]
    adv = panel["dollar_volume"][symbols].rolling(20, min_periods=1).mean()
    dvol = panel["ret"][symbols].rolling(21, min_periods=5).std()
    idx = c.index
    notional = capital / max_concurrent
    rows = []
    for sym in symbols:
        for i in range(1, len(idx) - 1):
            if seg == "overnight":
                # decide at close of i-1 (info through i-1), enter at close of i, exit at open of i+1
                d, e, x = i - 1, i, i + 1
                entry, exit_ = c[sym].iloc[e], o[sym].iloc[x]
                raw_entry = raw_c[sym].iloc[e]
            elif seg == "intraday":
                d, e, x = i - 1, i, i
                entry, exit_ = o[sym].iloc[e], c[sym].iloc[x]
                raw_entry = raw_o[sym].iloc[e]
            else:
                raise ValueError("segment must be overnight or intraday")
            if mask is not None and not mask[sym].iloc[d]:
                continue
            if not (vol[sym].iloc[e] > 0 and vol[sym].iloc[x] > 0) or not np.isfinite(entry) or not np.isfinite(exit_):
                continue
            g = exit_ / entry - 1.0
            cf = float(cost.round_trip_frac(notional, raw_entry, adv[sym].iloc[d], dvol[sym].iloc[d]))
            rows.append((sym, idx[d], idx[e], idx[x], 1.0, raw_entry, g, cf, g - cf, notional / adv[sym].iloc[d]))
    tr = pd.DataFrame(rows, columns=["symbol", "signal_session", "entry_session", "exit_session", "direction",
                                     "entry_price_raw", "gross_ret", "cost_frac", "net_ret", "participation"])
    # daily book: one segment per day per symbol, equal slots
    dg = tr.groupby("exit_session")["gross_ret"].sum().reindex(idx, fill_value=0.0) / max(len(symbols), 1)
    dn = tr.groupby("exit_session")["net_ret"].sum().reindex(idx, fill_value=0.0) / max(len(symbols), 1)
    return StrategyRun(tr, dg, dn, "segment", {"segment": seg, "round_trips_per_day": 1})


# ---------------------------------------------------------------------------------------------
# Running a template
# ---------------------------------------------------------------------------------------------
def spells_to_trades(weights: pd.DataFrame, panel, execution: str, cost_frac_per_side: float) -> pd.DataFrame:
    """Holding spells (contiguous non-zero target weight) → trade rows with gross/net returns."""
    px = panel["open"] if execution == "next_open" else panel["close"]
    raw = panel["raw_open"] if execution == "next_open" else panel["raw_close"]
    rows = []
    idx = weights.index
    for sym in weights.columns:
        on = (weights[sym].fillna(0.0).abs() > 0).to_numpy()
        t = 0
        T = len(on)
        while t < T:
            if on[t]:
                start = t
                while t < T and on[t]:
                    t += 1
                e, x = start + 1, t + 1  # decision at start → entry next session; exit after last "on" decision
                if x < T and e < T:
                    g = px[sym].iloc[x] / px[sym].iloc[e] - 1.0
                    if np.isfinite(g):
                        c = 2 * cost_frac_per_side
                        rows.append((sym, idx[start], idx[e], idx[x], 1.0, float(raw[sym].iloc[e]), g, c, g - c))
            else:
                t += 1
    return pd.DataFrame(rows, columns=["symbol", "signal_session", "entry_session", "exit_session", "direction",
                                       "entry_price_raw", "gross_ret", "cost_frac", "net_ret"])


def run_strategy(panel: dict[str, pd.DataFrame], name: str, params: dict[str, Any], symbols: list[str],
                 cost: CostModel, holding: int = 5, execution: str = "next_open", capital: float = 100_000.0,
                 max_concurrent: int = 5, notional: float | None = None) -> StrategyRun:
    cache: dict = {}
    symbols = [s for s in symbols if s in panel["close"].columns]
    kind = template_kind(name)
    if kind == "segment":
        return segment_run(panel, params, symbols, cost, capital, max_concurrent)
    if kind == "measurement":
        raise ValueError("measurement hypotheses are evaluated by quantlab.research.measurements, not run_strategy")
    if kind == "event":
        sig = SIGNAL_TEMPLATES[name](panel, params, symbols, cache)
        notional = notional or capital / max_concurrent
        tr, un = generate_trades(panel, sig, TradeConfig(holding=holding, execution=execution, notional=notional), cost)
        idx = panel["close"].index
        dg = trades_to_daily_returns(tr, idx, max_concurrent, "gross_ret") if len(tr) else pd.Series(0.0, index=idx)
        dn = trades_to_daily_returns(tr, idx, max_concurrent, "net_ret") if len(tr) else pd.Series(0.0, index=idx)
        return StrategyRun(tr, dg, dn, kind, {"n_signals": int((sig != 0).sum().sum()), "n_unfilled": len(un),
                                               "avg_participation": float(tr["participation"].mean()) if len(tr) else np.nan})
    w = POSITION_TEMPLATES[name](panel, params, symbols, cache)
    res = run_portfolio_backtest(panel, w, BacktestConfig(capital=capital, execution=execution, cost_model=cost))
    one_side = (cost.half_spread_bps + cost.slippage_bps) / 1e4
    tr = spells_to_trades(w, panel, execution, one_side)
    return StrategyRun(tr, res.gross.daily["ret"], res.net.daily["ret"], kind,
                       {"avg_turnover": float(res.net.daily["turnover"].mean()),
                        "avg_gross_exposure": float(res.net.daily["gross_exposure"].mean()),
                        "total_costs": float(res.net.daily[["cost_spread_slip_impact", "cost_commission",
                                                            "cost_borrow"]].sum().sum())})
