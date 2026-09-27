"""Market-regime labels, computed point-in-time from trailing data only.

Regimes are used two ways:
* **descriptive** — "how did the strategy do in bear markets?" (any regime definition is allowed,
  including ex-post ones such as NBER recessions, but they must never feed a trading rule);
* **conditioning** — a trading rule may only use regimes computable in real time, which is what
  this module produces.

Labels at session t use information up to the close of t (same availability as features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_labels(panel: dict[str, pd.DataFrame], market: str, vix: pd.Series | None = None,
                  rates: pd.Series | None = None) -> pd.DataFrame:
    c = panel["close"][market]
    lr = np.log1p(panel["ret"][market])
    sma200 = c.rolling(200, min_periods=200).mean()
    trend = pd.Series(np.where(c > sma200, "BULL", "BEAR"), index=c.index).where(sma200.notna())
    rv = lr.rolling(21, min_periods=21).std() * np.sqrt(252)
    vol_ref = vix / 100 if vix is not None else rv
    # high/low volatility relative to the trailing 3-year median (expanding at the start)
    med = vol_ref.rolling(756, min_periods=126).median()
    vol = pd.Series(np.where(vol_ref > med, "HIGH_VOL", "LOW_VOL"), index=c.index).where(med.notna() & vol_ref.notna())
    dd = c / c.cummax() - 1
    crash = pd.Series(np.where(dd <= -0.20, "DRAWDOWN_GT_20", "NORMAL"), index=c.index)
    out = pd.DataFrame({"trend": trend, "volatility": vol, "market_drawdown": crash,
                        "year": c.index.year.astype(str)}, index=c.index)
    if rates is not None:
        chg = rates.reindex(c.index).ffill().diff(63)
        out["rates"] = pd.Series(np.where(chg > 0, "RISING_RATES", "FALLING_RATES"), index=c.index).where(chg.notna())
    return out


def performance_by_regime(trades: pd.DataFrame, labels: pd.DataFrame, ret_col: str = "net_ret",
                          min_n: int = 10) -> dict:
    """Mean/median/count of trade returns by regime at the signal date. Small cells are flagged."""
    if trades.empty:
        return {}
    lab = labels.reindex(pd.DatetimeIndex(trades["signal_session"]))
    out = {}
    for col in labels.columns:
        g = pd.Series(trades[ret_col].to_numpy(), index=lab[col].to_numpy()).groupby(level=0)
        out[col] = {str(k): {"n": int(v.count()), "mean": float(v.mean()), "median": float(v.median()),
                             "win_rate": float((v > 0).mean()), "small_sample": bool(v.count() < min_n)}
                    for k, v in g}
    return out
