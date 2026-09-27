"""Relative-value toolkit: correlation, cointegration, spreads, residuals, PCA, clustering.

Tools, not claims. Correlation does not imply a tradable relationship, and a cointegration test
passed in-sample is a hypothesis to be registered and validated like any other — spreads between
related securities regularly "break" (mergers, index changes, regime shifts). Every estimator here
uses trailing windows when producing a tradable series (hedge ratios, z-scores); full-sample
estimates are for description only and are labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import adfuller, coint


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    return a.rolling(window, min_periods=window).corr(b)


def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """Full-sample Engle–Granger cointegration test (DESCRIPTIVE: uses the whole sample)."""
    df = pd.concat([np.log(y), np.log(x)], axis=1).dropna()
    if len(df) < 100:
        return {"status": "INSUFFICIENT_DATA", "n": len(df)}
    t, p, crit = coint(df.iloc[:, 0], df.iloc[:, 1])
    beta = np.polyfit(df.iloc[:, 1], df.iloc[:, 0], 1)[0]
    return {"status": "OK", "t_stat": float(t), "p_value": float(p), "hedge_ratio_full_sample": float(beta),
            "n": int(len(df)), "note": "full-sample (look-ahead) — describe only; trade with rolling_spread"}


def rolling_spread(y: pd.Series, x: pd.Series, window: int = 252) -> pd.DataFrame:
    """Log-price spread with a trailing OLS hedge ratio and trailing z-score (PIT-safe).

    Ratio and mean/sd at t use data up to t only.
    """
    ly, lx = np.log(y), np.log(x)
    cov = ly.rolling(window, min_periods=window).cov(lx)
    var = lx.rolling(window, min_periods=window).var()
    beta = cov / var
    alpha = ly.rolling(window, min_periods=window).mean() - beta * lx.rolling(window, min_periods=window).mean()
    spread = ly - (alpha + beta * lx)
    mu = spread.rolling(window, min_periods=window // 2).mean()
    sd = spread.rolling(window, min_periods=window // 2).std()
    return pd.DataFrame({"hedge_ratio": beta, "spread": spread, "zscore": (spread - mu) / sd})


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life (sessions) from an AR(1) fit of spread changes on lagged level."""
    s = spread.dropna()
    if len(s) < 50:
        return float("nan")
    ds, lag = s.diff().dropna(), s.shift(1).dropna().loc[s.diff().dropna().index]
    b = np.polyfit(lag, ds, 1)[0]
    return float(-np.log(2) / b) if b < 0 else float("inf")


def adf_pvalue(series: pd.Series) -> float:
    s = series.dropna()
    return float(adfuller(s, autolag="AIC", result_object=False)[1]) if len(s) > 50 else float("nan")


def pca_residuals(returns: pd.DataFrame, n_factors: int = 1, window: int = 252) -> pd.DataFrame:
    """Factor-neutral residual returns: at each date remove the first k principal components estimated
    on the trailing window (PIT-safe; recomputed every 21 sessions for speed)."""
    R = returns.dropna(how="all").fillna(0.0)
    out = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    loadings = None
    for i in range(window, len(R)):
        if loadings is None or (i - window) % 21 == 0:
            hist = R.iloc[i - window:i]
            hist = hist - hist.mean()
            _, _, vt = np.linalg.svd(hist.to_numpy(), full_matrices=False)
            loadings = vt[:n_factors].T  # (assets × k)
        r = R.iloc[i].to_numpy()
        f = loadings.T @ r
        out.iloc[i] = r - loadings @ f
    return out


def explained_variance(returns: pd.DataFrame) -> np.ndarray:
    X = returns.dropna().to_numpy()
    X = X - X.mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    return s**2 / np.sum(s**2)


def correlation_clusters(returns: pd.DataFrame, threshold: float = 0.5) -> dict[str, int]:
    """Hierarchical clustering on correlation distance sqrt(2(1-ρ)). Descriptive grouping."""
    c = returns.corr().fillna(0.0)
    d = np.sqrt(np.clip(2 * (1 - c.to_numpy()), 0, None))
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(z, t=np.sqrt(2 * (1 - threshold)), criterion="distance")
    return dict(zip(c.columns, labels.tolist()))
