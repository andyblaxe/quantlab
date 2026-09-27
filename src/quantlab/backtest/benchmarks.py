"""Benchmarks every strategy must be compared against (see RESEARCH_METHODOLOGY.md §Benchmarks).

All return target-weight panels (decision at close t → executed next session by the engine) or
signal panels for the trade engine. Model-based benchmarks (logistic/linear) live in
:mod:`quantlab.research.models`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(index: pd.DatetimeIndex, symbols: list[str], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Constant target weights (equal by default) — rebalanced only when drift exceeds the engine's threshold."""
    if weights is None:
        weights = {s: 1.0 / len(symbols) for s in symbols}
    return pd.DataFrame({s: float(weights.get(s, 0.0)) for s in symbols}, index=index)


def random_signal(like: pd.DataFrame, frequency: float, seed: int, direction: float = 1.0) -> pd.DataFrame:
    """Random entries with a given per-cell probability — the 'random prediction' benchmark."""
    rng = np.random.default_rng(seed)
    mask = rng.random(like.shape) < frequency
    out = pd.DataFrame(np.where(mask, direction, 0.0), index=like.index, columns=like.columns)
    return out.where(like.notna(), 0.0)


def time_series_momentum_weights(close: pd.DataFrame, lookback: int = 252, skip: int = 21,
                                 long_only: bool = True) -> pd.DataFrame:
    """Simple momentum: hold (equal weight) assets whose trailing return is positive."""
    m = close.shift(skip) / close.shift(lookback) - 1.0
    sig = (m > 0).astype(float) if long_only else np.sign(m)
    sig = sig.where(m.notna())
    n = sig.abs().sum(axis=1).replace(0, np.nan)
    return sig.div(n, axis=0).fillna(0.0).where(m.notna().any(axis=1), np.nan)


def mean_reversion_signal(ret: pd.DataFrame, lookback: int = 5, z: float = -2.0, vol_n: int = 63) -> pd.DataFrame:
    """Simple mean reversion: long after a lookback-day move below ``z`` standard deviations."""
    lr = np.log1p(ret)
    move = lr.rolling(lookback).sum()
    sd = lr.rolling(vol_n).std().shift(lookback) * np.sqrt(lookback)
    return ((move / sd) < z).astype(float)
