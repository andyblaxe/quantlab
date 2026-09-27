"""Core inference: HAC t-tests, block bootstrap, permutation tests, effective sample size.

Plain-English notes accompany each function because the Research Analyst quotes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    method: str
    n: int
    detail: dict

    def to_dict(self) -> dict:
        return {"statistic": self.statistic, "p_value": self.p_value, "method": self.method, "n": self.n,
                **self.detail}


def _pvalue(t: float, dof: float, alternative: str) -> float:
    if np.isnan(t):
        return np.nan
    if alternative == "greater":
        return float(stats.t.sf(t, dof))
    if alternative == "less":
        return float(stats.t.cdf(t, dof))
    return float(2 * stats.t.sf(abs(t), dof))


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """HAC (Newey–West, Bartlett kernel) standard error of the mean."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    u = x - x.mean()
    gamma0 = u @ u / n
    s = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (u[k:] @ u[:-k]) / n
    return float(np.sqrt(max(s, 0.0) / n))


def mean_test_hac(x, lags: int | None = None, alternative: str = "greater") -> TestResult:
    """t-test of mean > 0 robust to autocorrelation (e.g. overlapping holding periods).

    Plain English: "is the average return distinguishable from zero, allowing for the fact that
    consecutive observations are not independent?"
    """
    x = pd.Series(x, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 3:
        return TestResult(np.nan, np.nan, "newey_west_t", n, {"lags": lags})
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))  # Newey–West (1994) rule of thumb
    se = newey_west_se(x, lags)
    t = x.mean() / se if se > 0 else np.nan
    return TestResult(float(t), _pvalue(t, n - 1, alternative), "newey_west_t", n,
                      {"lags": lags, "mean": float(x.mean()), "se": se})


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap: random-length blocks (geometric), wrapping around."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if new_block[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_ci(x, statistic=np.mean, n_boot: int = 2000, level: float = 0.95, mean_block: float | None = None,
                 seed: int = 0) -> dict:
    """Stationary-block-bootstrap percentile CI for a statistic.

    Plain English: "if history had been reshuffled in chunks (keeping short-term dependence), how
    much would this number move?" A CI that includes 0 means the sign itself is uncertain.
    """
    x = pd.Series(x, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 5:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "n": n, "level": level}
    rng = np.random.default_rng(seed)
    b = mean_block or max(1.0, n ** (1 / 3))
    draws = np.array([statistic(x[stationary_bootstrap_indices(n, b, rng)]) for _ in range(n_boot)])
    a = (1 - level) / 2
    return {"estimate": float(statistic(x)), "lo": float(np.quantile(draws, a)), "hi": float(np.quantile(draws, 1 - a)),
            "n": n, "level": level, "mean_block": b, "n_boot": n_boot,
            "p_le_zero": float(np.mean(draws <= 0))}


def sign_flip_test(x, n_perm: int = 5000, seed: int = 0, alternative: str = "greater") -> TestResult:
    """Randomisation test of mean = 0 by flipping signs (assumes a symmetric null distribution)."""
    x = pd.Series(x, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 5:
        return TestResult(np.nan, np.nan, "sign_flip", n, {})
    rng = np.random.default_rng(seed)
    obs = x.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * np.abs(x)).mean(axis=1)
    if alternative == "greater":
        p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    elif alternative == "less":
        p = (1 + np.sum(null <= obs)) / (n_perm + 1)
    else:
        p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return TestResult(float(obs), float(p), "sign_flip", n, {"n_perm": n_perm})


def random_entry_test(trades: pd.DataFrame, fwd_returns: pd.DataFrame, n_perm: int = 2000, seed: int = 0,
                      ret_col: str = "gross_ret", eligible: pd.DataFrame | None = None) -> TestResult:
    """Is the strategy's mean trade return better than the same number of *random* entries?

    Null: timing carries no information. Random trades are drawn from the same symbols, the same
    period and the same holding horizon (``fwd_returns`` = forward returns at that horizon, aligned
    to decision dates), with the same direction mix. This is the "random prediction" benchmark: it
    controls for drift — a long-only rule in a rising market beats zero without any skill.
    """
    if len(trades) < 5:
        return TestResult(np.nan, np.nan, "random_entry", len(trades), {})
    rng = np.random.default_rng(seed)
    obs = float(trades[ret_col].mean())
    fr = fwd_returns[sorted(set(trades["symbol"]) & set(fwd_returns.columns))]
    lo, hi = trades["signal_session"].min(), trades["signal_session"].max()
    fr = fr.loc[(fr.index >= lo) & (fr.index <= hi)]
    if eligible is not None:
        fr = fr.where(eligible.reindex_like(fr).fillna(False).astype(bool))
    pool = fr.stack().dropna().to_numpy()
    if len(pool) < 20:
        return TestResult(np.nan, np.nan, "random_entry", len(trades), {"pool": len(pool)})
    dirs = trades["direction"].to_numpy()
    k = len(trades)
    null = np.array([(pool[rng.integers(len(pool), size=k)] * dirs).mean() for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return TestResult(obs, float(p), "random_entry", k,
                      {"null_mean": float(null.mean()), "null_sd": float(null.std()), "pool": int(len(pool)),
                       "n_perm": n_perm, "excess_over_random": obs - float(null.mean())})


def circular_shift_test(signal: pd.Series, target: pd.Series, n_perm: int = 2000, seed: int = 0,
                        min_shift: int = 21) -> TestResult:
    """Association test preserving each series' autocorrelation: circularly shift the signal.

    Statistic: Spearman correlation of signal with target. Plain English: "would a signal with the
    same persistence, but misaligned in time, correlate as strongly by chance?"
    """
    df = pd.concat([signal, target], axis=1).dropna()
    n = len(df)
    if n < 3 * min_shift:
        return TestResult(np.nan, np.nan, "circular_shift", n, {})
    s, y = df.iloc[:, 0].rank().to_numpy(), df.iloc[:, 1].rank().to_numpy()
    obs = float(np.corrcoef(s, y)[0, 1])
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, n - min_shift, size=n_perm)
    null = np.array([np.corrcoef(np.roll(s, k), y)[0, 1] for k in shifts])
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return TestResult(obs, float(p), "circular_shift", n, {"n_perm": n_perm})


def effective_n(entry_sessions: pd.Series, sessions: pd.DatetimeIndex, holding: int) -> int:
    """Conservative count of independent observations for (possibly clustered/overlapping) trades.

    The calendar is cut into consecutive blocks of ``holding`` sessions; each block containing at
    least one entry counts once. Twenty trades on the same day, or daily entries into overlapping
    5-day holds, therefore count as a handful of observations, not dozens.
    """
    if len(entry_sessions) == 0:
        return 0
    pos = pd.DatetimeIndex(sessions).get_indexer(pd.DatetimeIndex(entry_sessions))
    pos = pos[pos >= 0]
    return int(len(np.unique(pos // max(1, holding))))
