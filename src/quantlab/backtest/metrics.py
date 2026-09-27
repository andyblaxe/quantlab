"""Performance metrics for daily return series and trade lists.

All ratios are computed on simple returns; annualisation uses 252 sessions. Sharpe here is the
excess-over-zero Sharpe unless a risk-free series is supplied. Win rate is reported but is never a
selection criterion on its own (see RESEARCH_METHODOLOGY.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def _clean(r: pd.Series) -> pd.Series:
    return pd.Series(r, dtype=float).dropna()


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1 + _clean(returns)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else np.nan


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest run of sessions spent below a previous equity peak."""
    dd = drawdown_series(returns)
    under = (dd < 0).astype(int).to_numpy()
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return int(best)


def sharpe(returns: pd.Series, rf_daily: pd.Series | float = 0.0) -> float:
    r = _clean(returns - rf_daily)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ANN))


def sortino(returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = _clean(returns) - rf_daily
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    if len(r) < 2 or downside == 0:
        return np.nan
    return float(r.mean() / downside * np.sqrt(ANN))


def cagr(returns: pd.Series) -> float:
    r = _clean(returns)
    if not len(r):
        return np.nan
    growth = float((1 + r).prod())
    years = len(r) / ANN
    if growth <= 0:
        return -1.0
    return growth ** (1 / years) - 1


def summarize_returns(returns: pd.Series) -> dict:
    r = _clean(returns)
    if len(r) == 0:
        return {"n_days": 0}
    var95 = float(-np.quantile(r, 0.05))
    es95 = float(-r[r <= np.quantile(r, 0.05)].mean())
    return {
        "n_days": int(len(r)),
        "total_return": float((1 + r).prod() - 1),
        "cagr": cagr(r),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(ANN)),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(r),
        "max_dd_duration_days": max_drawdown_duration(r),
        "calmar": (cagr(r) / -max_drawdown(r)) if max_drawdown(r) < 0 else np.nan,
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurt()),
        "var_95_1d": var95,
        "es_95_1d": es95,
        "pct_positive_days": float((r > 0).mean()),
    }


def summarize_trades(trade_returns: pd.Series) -> dict:
    """Per-trade statistics. ``trade_returns`` are fractional returns per trade."""
    r = _clean(trade_returns)
    n = len(r)
    if n == 0:
        return {"n_trades": 0}
    wins, losses = r[r > 0], r[r < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    sd = float(r.std(ddof=1)) if n > 1 else np.nan
    return {
        "n_trades": n,
        "mean": float(r.mean()),
        "median": float(r.median()),
        "std": sd,
        "t_stat_naive": float(r.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
        "win_rate": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else np.nan,
        "best": float(r.max()),
        "worst": float(r.min()),
        "skew": float(r.skew()) if n > 2 else np.nan,
    }


def yearly_returns(returns: pd.Series) -> pd.Series:
    r = _clean(returns)
    return (1 + r).groupby(r.index.year).prod() - 1


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    r = _clean(returns)
    return r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(ANN)


def trades_to_daily_returns(trades: pd.DataFrame, sessions: pd.DatetimeIndex, max_concurrent: int = 5,
                            ret_col: str = "net_ret") -> pd.Series:
    """Portfolio daily returns from a trade list: capital split into ``max_concurrent`` equal slots.

    Each trade's return is spread geometrically over its holding days and booked in its slot; idle
    slots earn zero (cash). Trades arriving when every slot is busy are skipped (capacity limit),
    so the result is an *implementable* portfolio, not a sum of overlapping trades.
    """
    sessions = pd.DatetimeIndex(sessions)
    daily = pd.Series(0.0, index=sessions)
    busy_until = [pd.Timestamp.min] * max_concurrent
    for tr in trades.sort_values("entry_session").itertuples():
        slot = next((k for k, b in enumerate(busy_until) if b <= tr.entry_session), None)
        if slot is None:
            continue
        span = sessions[(sessions > tr.entry_session) & (sessions <= tr.exit_session)]
        if len(span) == 0:
            continue
        r = getattr(tr, ret_col)
        per_day = (1 + r) ** (1 / len(span)) - 1 if r > -1 else -1.0
        daily.loc[span] += per_day / max_concurrent
        busy_until[slot] = tr.exit_session
    return daily
