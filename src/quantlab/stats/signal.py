"""Signal-quality statistics: information coefficients, quantile spreads, Bayesian shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantlab.stats.core import mean_test_hac


def cross_sectional_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 5) -> pd.Series:
    """Spearman rank IC per date across symbols (NaN when fewer than ``min_names`` usable)."""
    s, f = signal.align(fwd, join="inner")
    out = {}
    for d in s.index:
        a, b = s.loc[d], f.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= min_names:
            out[d] = stats.spearmanr(a[m], b[m]).statistic
    return pd.Series(out, dtype=float)


def ic_summary(ic: pd.Series, horizon: int = 1) -> dict:
    """Mean IC with a HAC t-stat (lags ≥ horizon-1 because overlapping labels autocorrelate)."""
    ic = ic.dropna()
    t = mean_test_hac(ic, lags=max(horizon - 1, int(np.floor(4 * (len(ic) / 100) ** (2 / 9)))), alternative="two-sided")
    return {"mean_ic": float(ic.mean()) if len(ic) else np.nan, "ic_std": float(ic.std()) if len(ic) > 1 else np.nan,
            "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else np.nan,
            "t_stat": t.statistic, "p_value": t.p_value, "n_dates": int(len(ic)),
            "pct_positive": float((ic > 0).mean()) if len(ic) else np.nan}


def time_series_ic(signal: pd.Series, fwd: pd.Series) -> float:
    df = pd.concat([signal, fwd], axis=1).dropna()
    return float(stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1]).statistic) if len(df) > 10 else np.nan


def quantile_returns(signal: pd.Series, fwd: pd.Series, q: int = 5) -> pd.DataFrame:
    """Mean/median/count of forward return per signal quantile (pooled). Shows monotonicity."""
    df = pd.concat({"s": signal, "y": fwd}, axis=1).dropna()
    if len(df) < q * 10:
        return pd.DataFrame()
    df["bucket"] = pd.qcut(df["s"].rank(method="first"), q, labels=list(range(1, q + 1)))
    return df.groupby("bucket", observed=True)["y"].agg(["mean", "median", "count"])


def normal_shrinkage_posterior(sample_mean: float, sample_se: float, prior_mean: float = 0.0,
                               prior_sd: float | None = None) -> dict:
    """Posterior of a mean return under a skeptical Normal prior centred on zero.

    Plain English: "start from the belief that most edges are zero; the data must pull us away."
    The default prior s.d. equals the sampling s.e. (a data-scaled, fairly skeptical prior).
    """
    if not np.isfinite(sample_se) or sample_se <= 0:
        return {"post_mean": np.nan, "post_sd": np.nan, "p_positive": np.nan}
    prior_sd = prior_sd or sample_se
    w = prior_sd**2 / (prior_sd**2 + sample_se**2)
    post_mean = prior_mean + w * (sample_mean - prior_mean)
    post_sd = np.sqrt(1 / (1 / prior_sd**2 + 1 / sample_se**2))
    return {"post_mean": float(post_mean), "post_sd": float(post_sd),
            "p_positive": float(stats.norm.sf(0, post_mean, post_sd)), "shrinkage_weight_on_data": float(w)}
