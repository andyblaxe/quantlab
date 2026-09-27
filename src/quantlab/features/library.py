"""The reusable feature library (all trailing-window, point-in-time).

Price-based features use the forward-adjusted panel (``close``/``open``/``high``/``low``), whose
ratios are PIT-safe. Level-sensitive features (price filters, dollar volume) use raw fields.

Naming: ``family(param=value,...)`` e.g. ``sma_ratio(n=200)``; see :func:`quantlab.features.base.get_feature`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.calendar import get_calendar
from quantlab.features.base import Panel, feature_family

ANN = 252


def _logret(p: Panel) -> pd.DataFrame:
    return np.log1p(p["ret"])


def _sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).mean()


def _wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    # Recursive smoothing; adjust=False makes each value depend only on the past.
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ---------------------------------------------------------------------------------------------
# Returns and momentum
# ---------------------------------------------------------------------------------------------
@feature_family("ret", defaults={"h": 1}, lookback=lambda q: q["h"], category="returns",
                description="h-session total return (close to close).")
def _ret(h):
    return lambda p: p["close"] / p["close"].shift(h) - 1.0


@feature_family("mom", defaults={"lookback": 252, "skip": 21}, lookback=lambda q: q["lookback"],
                category="momentum", description="Return from t-lookback to t-skip (classic 12-1 momentum by default).")
def _mom(lookback, skip):
    return lambda p: p["close"].shift(skip) / p["close"].shift(lookback) - 1.0


@feature_family("vol", defaults={"n": 21}, lookback=lambda q: q["n"], category="volatility",
                description="Annualised realised volatility of daily log returns over n sessions.")
def _vol(n):
    return lambda p: _logret(p).rolling(n, min_periods=n).std() * np.sqrt(ANN)


@feature_family("zret", defaults={"h": 1, "vol_n": 63}, lookback=lambda q: q["h"] + q["vol_n"],
                category="returns",
                description="h-session log return divided by the volatility measured *before* the move "
                            "(window ending at t-h), scaled by sqrt(h).")
def _zret(h, vol_n):
    def f(p):
        lr = _logret(p)
        move = lr.rolling(h, min_periods=h).sum()
        sd = lr.rolling(vol_n, min_periods=vol_n).std().shift(h)
        return move / (sd * np.sqrt(h))
    return f


@feature_family("vol_adj_mom", defaults={"lookback": 252, "skip": 21, "vol_n": 63},
                lookback=lambda q: q["lookback"] + 1, category="momentum",
                description="Momentum divided by trailing annualised volatility.")
def _vol_adj_mom(lookback, skip, vol_n):
    def f(p):
        m = p["close"].shift(skip) / p["close"].shift(lookback) - 1.0
        v = _logret(p).rolling(vol_n, min_periods=vol_n).std() * np.sqrt(ANN)
        return m / v
    return f


@feature_family("overnight_ret", defaults={}, lookback=lambda q: 1, category="intraday",
                description="Open / previous close - 1 (adjusted; dividends/splits accounted for).",
                inputs=("open", "close"))
def _overnight():
    return lambda p: p["open"] / p["close"].shift(1) - 1.0


@feature_family("intraday_ret", defaults={}, lookback=lambda q: 0, category="intraday",
                description="Close / open - 1 for the same session.", inputs=("open", "close"))
def _intraday():
    return lambda p: p["close"] / p["open"] - 1.0


# ---------------------------------------------------------------------------------------------
# Trend / moving averages / oscillators
# ---------------------------------------------------------------------------------------------
@feature_family("sma_ratio", defaults={"n": 50}, lookback=lambda q: q["n"], category="trend",
                description="Close / SMA(n) - 1.")
def _sma_ratio(n):
    return lambda p: p["close"] / _sma(p["close"], n) - 1.0


@feature_family("sma_slope", defaults={"n": 50, "k": 5}, lookback=lambda q: q["n"] + q["k"], category="trend",
                description="SMA(n)_t / SMA(n)_{t-k} - 1: direction of the moving average.")
def _sma_slope(n, k):
    return lambda p: _sma(p["close"], n) / _sma(p["close"], n).shift(k) - 1.0


@feature_family("ma_cross", defaults={"fast": 50, "slow": 200}, lookback=lambda q: q["slow"], category="trend",
                description="SMA(fast) / SMA(slow) - 1 (>0 ⇔ 'golden cross' state).")
def _ma_cross(fast, slow):
    return lambda p: _sma(p["close"], fast) / _sma(p["close"], slow) - 1.0


@feature_family("bollinger", defaults={"n": 20, "k": 2.0}, lookback=lambda q: q["n"], category="mean_reversion",
                description="(Close - SMA(n)) / (k * rolling std(n)); ±1 = on the bands.")
def _bollinger(n, k):
    def f(p):
        c = p["close"]
        return (c - _sma(c, n)) / (k * c.rolling(n, min_periods=n).std())
    return f


@feature_family("rsi", defaults={"n": 14}, lookback=lambda q: q["n"] + 1, category="mean_reversion",
                description="Wilder RSI (0-100).")
def _rsi(n):
    def f(p):
        d = p["close"].diff()
        up = _wilder(d.clip(lower=0), n)
        dn = _wilder((-d).clip(lower=0), n)
        rs = up / dn
        return 100 - 100 / (1 + rs)
    return f


@feature_family("atr_pct", defaults={"n": 14}, lookback=lambda q: q["n"] + 1, category="volatility",
                description="Wilder ATR as a fraction of close.", inputs=("high", "low", "close"))
def _atr(n):
    def f(p):
        pc = p["close"].shift(1)
        tr = np.maximum(np.maximum(p["high"] - p["low"], (p["high"] - pc).abs()), (p["low"] - pc).abs())
        return _wilder(tr, n) / p["close"]
    return f


@feature_family("drawdown", defaults={"n": 252}, lookback=lambda q: q["n"], category="trend",
                description="Close / rolling max close over n sessions - 1.")
def _drawdown(n):
    return lambda p: p["close"] / p["close"].rolling(n, min_periods=1).max() - 1.0


@feature_family("breakout", defaults={"n": 55}, lookback=lambda q: q["n"] + 1, category="trend",
                description="Close / highest high of the *previous* n sessions - 1 (>0 = breakout).",
                inputs=("high", "close"))
def _breakout(n):
    return lambda p: p["close"] / p["high"].rolling(n, min_periods=n).max().shift(1) - 1.0


# ---------------------------------------------------------------------------------------------
# Volume / liquidity
# ---------------------------------------------------------------------------------------------
@feature_family("rel_volume", defaults={"n": 20}, lookback=lambda q: q["n"] + 1, category="volume",
                description="Volume / mean volume of the previous n sessions (split-adjusted).", inputs=("volume",))
def _rel_volume(n):
    return lambda p: p["volume"] / p["volume"].rolling(n, min_periods=n).mean().shift(1)


@feature_family("adv_dollar", defaults={"n": 20}, lookback=lambda q: q["n"], category="liquidity",
                description="Average raw dollar volume over n sessions (liquidity / capacity filter).",
                inputs=("raw_close", "raw_volume"))
def _adv_dollar(n):
    return lambda p: p["dollar_volume"].rolling(n, min_periods=n).mean()


@feature_family("price", defaults={}, lookback=lambda q: 0, category="liquidity",
                description="Raw (as-traded) close, for price filters.", inputs=("raw_close",))
def _price():
    return lambda p: p["raw_close"].copy()


# ---------------------------------------------------------------------------------------------
# Relative / market features
# ---------------------------------------------------------------------------------------------
@feature_family("beta", defaults={"n": 63, "market": "SPY"}, lookback=lambda q: q["n"], category="market",
                description="Rolling OLS beta of daily returns to the market symbol.")
def _beta(n, market):
    def f(p):
        r = p["ret"]
        m = r[market]
        cov = r.rolling(n, min_periods=n).cov(m)
        var = m.rolling(n, min_periods=n).var()
        return cov.div(var, axis=0)
    return f


@feature_family("corr", defaults={"n": 63, "market": "SPY"}, lookback=lambda q: q["n"], category="market",
                description="Rolling correlation of daily returns with the market symbol.")
def _corr(n, market):
    return lambda p: p["ret"].rolling(n, min_periods=n).corr(p["ret"][market])


@feature_family("rel_ret", defaults={"h": 21, "market": "SPY"}, lookback=lambda q: q["h"], category="relative",
                description="h-session return minus the market's h-session return.")
def _rel_ret(h, market):
    def f(p):
        r = p["close"] / p["close"].shift(h) - 1.0
        return r.sub(r[market], axis=0)
    return f


@feature_family("breadth", defaults={"n": 50}, lookback=lambda q: q["n"], category="market",
                description="Fraction of the panel's symbols above their SMA(n) (same value in every column).")
def _breadth(n):
    def f(p):
        c = p["close"]
        above = (c > _sma(c, n)).where(_sma(c, n).notna())
        frac = above.mean(axis=1, skipna=True)
        return pd.DataFrame(np.repeat(frac.to_numpy()[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    return f


@feature_family("xs_rank_mom", defaults={"lookback": 252, "skip": 21}, lookback=lambda q: q["lookback"],
                category="cross_sectional",
                description="Cross-sectional percentile rank (0-1) of momentum on each date.")
def _xs_rank_mom(lookback, skip):
    def f(p):
        m = p["close"].shift(skip) / p["close"].shift(lookback) - 1.0
        return m.rank(axis=1, pct=True)
    return f


# ---------------------------------------------------------------------------------------------
# Calendar features (known in advance, from the exchange calendar)
# ---------------------------------------------------------------------------------------------
@feature_family("turn_of_month", defaults={"before": 1, "after": 3, "lead": 1}, lookback=lambda q: 0,
                category="calendar",
                description="1 if the session `lead` sessions ahead is within the turn-of-month window "
                            "(last `before` sessions of a month through the first `after`), else 0. "
                            "Use lead=2 with next_close execution to hold exactly the flagged session.")
def _tom(before, after, lead):
    def f(p):
        idx = p["close"].index
        # Uses the exchange calendar, which is published in advance — not future panel rows.
        # Caveat: unscheduled closures (e.g. 2012 Hurricane Sandy) are in today's calendar but were
        # not known in advance; the effect on a turn-of-month flag is negligible and documented.
        cal = get_calendar()
        sess = cal.sessions(idx[0], idx[-1] + pd.Timedelta(days=70))
        s = pd.Series(sess, index=sess)
        month = s.dt.to_period("M")
        pos_from_start = s.groupby(month).cumcount() + 1
        pos_from_end = s.groupby(month).cumcount(ascending=False) + 1
        in_window = ((pos_from_start <= after) | (pos_from_end <= before)).astype(float)
        nxt = in_window.shift(-lead).reindex(idx)
        return pd.DataFrame(np.repeat(nxt.to_numpy()[:, None], p["close"].shape[1], axis=1),
                            index=idx, columns=p["close"].columns)
    return f
