"""Measurement hypotheses: quantify a relationship without claiming a tradable strategy.

Some questions are prerequisites for strategies rather than strategies themselves, e.g. "is the
volatility risk premium positive, and how fat is its left tail?". They are pre-registered like any
hypothesis, evaluated on development data with HAC inference, and recorded — but they never enter
the signal catalog as tradable signals. Trading them usually needs data we may not have (option
prices), which the conclusion states.

Every measurement returns ``{"estimate", "p_value", "n", "effective_n", "details", "conclusion"}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.stats.core import bootstrap_ci, mean_test_hac

ANN = 252


def _realized_var_forward(ret: pd.Series, h: int) -> pd.Series:
    """Annualised realised variance over the next h sessions (t+1..t+h) — a label, future-looking."""
    lr = np.log1p(ret)
    fwd_sq = (lr**2).rolling(h).sum().shift(-h)
    return fwd_sq * ANN / h


def vrp(data, dev_end: pd.Timestamp, params: dict) -> dict:
    """Variance risk premium: implied variance (VIX²) minus subsequently realised variance."""
    h = int(params.get("horizon", 21))
    key = params.get("implied", data.vix_key)
    if key not in data.series:
        return {"status": "DATA_UNAVAILABLE", "missing": key}
    iv = (data.series[key] / 100.0) ** 2
    rv = _realized_var_forward(data.panel["ret"][data.market_symbol], h)
    prem = (iv - rv).loc[:dev_end].dropna()
    if len(prem) < 250:
        return {"status": "INSUFFICIENT_DATA", "n": int(len(prem))}
    t = mean_test_hac(prem, lags=h + 5, alternative="greater")
    q = prem.quantile([0.01, 0.05, 0.5, 0.95]).to_dict()
    return {"status": "OK", "estimate": float(prem.mean()), "p_value": t.p_value, "n": int(len(prem)),
            "effective_n": int(len(prem) // h),
            "details": {"mean_implied_var": float(iv.loc[prem.index].mean()), "mean_realized_var": float(rv.loc[prem.index].mean()),
                        "pct_positive": float((prem > 0).mean()), "quantiles": {str(k): float(v) for k, v in q.items()},
                        "worst": float(prem.min()), "hac": t.to_dict(),
                        "bootstrap_mean": bootstrap_ci(prem, n_boot=500, mean_block=h)},
            "conclusion_hint": "A positive average premium does not imply a profitable option-selling strategy: "
                               "that requires option bid/ask data, tail-risk sizing and margin modelling."}


def conditional_forward(data, dev_end: pd.Timestamp, params: dict) -> dict:
    """Forward h-day return and realised vol conditional on a series ratio crossing a threshold,
    e.g. VIX/VIX3M > 1 (term-structure inversion)."""
    h = int(params.get("horizon", 21))
    num, den = params["numerator"], params.get("denominator")
    if num not in data.series or (den and den not in data.series):
        return {"status": "DATA_UNAVAILABLE", "missing": [k for k in (num, den) if k and k not in data.series]}
    x = data.series[num] / (data.series[den] if den else 1.0)
    cond = x > float(params.get("threshold", 1.0))
    ret = data.panel["ret"][data.market_symbol]
    close = data.panel["close"][data.market_symbol]
    fwd_ret = close.shift(-(h + 1)) / close.shift(-1) - 1  # enter next close, hold h sessions
    fwd_vol = np.sqrt(_realized_var_forward(ret, h).shift(-1))
    df = pd.DataFrame({"cond": cond, "ret": fwd_ret, "vol": fwd_vol}).loc[:dev_end].dropna()
    on, off = df[df["cond"]], df[~df["cond"]]
    if len(on) < 30:
        return {"status": "INSUFFICIENT_DATA", "n_condition_days": int(len(on))}
    diff_vol = df["vol"].where(df["cond"]) - off["vol"].mean()
    tv = mean_test_hac(diff_vol.dropna(), lags=h + 5, alternative="greater")
    diff_ret = df["ret"].where(df["cond"]) - off["ret"].mean()
    tr = mean_test_hac(diff_ret.dropna(), lags=h + 5, alternative="two-sided")
    # episodes: consecutive condition days count once
    episodes = int((df["cond"] & ~df["cond"].shift(1, fill_value=False)).sum())
    return {"status": "OK", "estimate": float(on["vol"].mean() - off["vol"].mean()), "p_value": tv.p_value,
            "n": int(len(on)), "effective_n": episodes,
            "details": {"fwd_vol_when_true": float(on["vol"].mean()), "fwd_vol_when_false": float(off["vol"].mean()),
                        "fwd_ret_when_true": float(on["ret"].mean()), "fwd_ret_when_false": float(off["ret"].mean()),
                        "ret_difference_p_two_sided": tr.p_value, "n_episodes": episodes,
                        "frac_days_true": float(df["cond"].mean())},
            "conclusion_hint": "Condition days cluster into few episodes; effective sample size is the episode count."}


MEASUREMENTS = {"measurement:vrp": vrp, "measurement:conditional_forward": conditional_forward}
