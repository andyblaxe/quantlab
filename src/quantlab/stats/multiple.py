"""Multiple-testing corrections and selection-bias-aware Sharpe statistics.

* Bonferroni / Holm — family-wise error control (strict).
* Benjamini–Hochberg — false-discovery-rate control under independence/positive dependence.
* Benjamini–Yekutieli — FDR control under **arbitrary** dependence (default here, because
  hypotheses built from overlapping features and data are correlated).
* Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) — Bailey & López de Prado
  (2012, 2014). DSR asks: given that N strategies were tried, how likely is it that this Sharpe
  exceeds what the best of N *skill-less* strategies would show by luck?
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _arr(p) -> np.ndarray:
    return np.asarray(p, dtype=float)


def bonferroni(p) -> np.ndarray:
    p = _arr(p)
    return np.minimum(p * len(p), 1.0)


def holm(p) -> np.ndarray:
    p = _arr(p)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def benjamini_hochberg(p) -> np.ndarray:
    p = _arr(p)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    return adj


def benjamini_yekutieli(p) -> np.ndarray:
    p = _arr(p)
    m = len(p)
    if m == 0:
        return p
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    return np.minimum(benjamini_hochberg(p) * c_m, 1.0)


def adjust(p, method: str = "by") -> np.ndarray:
    """Adjusted p-values. NaN p-values (untestable hypotheses) are treated as 1.0 — they still
    count toward the family size, so skipping a test never makes the others look better."""
    p = np.nan_to_num(_arr(p), nan=1.0)
    return {"bonferroni": bonferroni, "holm": holm, "bh": benjamini_hochberg, "by": benjamini_yekutieli}[method](p)


def probabilistic_sharpe_ratio(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark) given an observed *per-period* Sharpe over n observations.

    ``kurt`` is raw (non-excess) kurtosis. Fat tails and negative skew widen the uncertainty.
    """
    if n < 3 or np.isnan(sr):
        return np.nan
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` skill-less strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(sr: float, n: int, n_trials: int, sr_std_across_trials: float | None = None,
                          skew: float = 0.0, kurt: float = 3.0) -> dict:
    """DSR = PSR evaluated against the expected max Sharpe of ``n_trials`` null strategies.

    If the cross-trial Sharpe dispersion is unknown, use the null sampling s.d. 1/sqrt(n-1)
    (what the best of N pure-noise strategies would show).
    """
    if sr_std_across_trials is None or not np.isfinite(sr_std_across_trials) or sr_std_across_trials <= 0:
        sr_std_across_trials = 1.0 / np.sqrt(max(n - 1, 1))
    bench = expected_max_sharpe(n_trials, sr_std_across_trials)
    return {"dsr": probabilistic_sharpe_ratio(sr, n, skew, kurt, bench), "sr_benchmark": bench,
            "n_trials": int(n_trials), "sr_std": float(sr_std_across_trials)}


def min_track_record_length(sr: float, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0,
                            prob: float = 0.95) -> float:
    """Observations needed for PSR ≥ prob (per-period Sharpe). ∞ if sr ≤ benchmark."""
    if sr <= sr_benchmark:
        return float("inf")
    z = stats.norm.ppf(prob)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / (sr - sr_benchmark)) ** 2)
