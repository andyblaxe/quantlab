"""Event studies: average abnormal returns around dated events (earnings, macro releases, ...).

Abnormal return = asset return − beta × market return, with beta estimated on a window that ends
before the event window starts (no contamination by the event itself). Day 0 is the first session
whose return can reflect the event (for an after-close announcement, the next session).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.stats.core import bootstrap_ci


def reaction_session(announce_time: pd.Timestamp, sessions: pd.DatetimeIndex, closes_utc: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First session whose close-to-close return can reflect information released at ``announce_time``."""
    pos = np.searchsorted(closes_utc.asi8, pd.Timestamp(announce_time).tz_convert("UTC").value, side="left")
    return sessions[pos] if pos < len(sessions) else None


def event_study(ret: pd.DataFrame, market: pd.Series, events: pd.DataFrame, pre: int = 5, post: int = 20,
                est_window: int = 120, gap: int = 10, seed: int = 0) -> dict:
    """``events`` needs columns ``symbol`` and ``event_session`` (day 0).

    Returns the mean abnormal-return path (days -pre..post), the cumulative path, per-event CARs
    and bootstrap CIs for CAR(0..post) and CAR(1..post) (the latter = post-event drift).
    """
    sessions = ret.index
    rel = np.arange(-pre, post + 1)
    paths, rows = [], []
    for ev in events.itertuples():
        if ev.symbol not in ret.columns or ev.event_session not in sessions:
            continue
        i = sessions.get_loc(ev.event_session)
        if i - pre - gap - est_window < 0 or i + post >= len(sessions):
            continue
        est = slice(i - pre - gap - est_window, i - pre - gap)
        y, x = ret[ev.symbol].iloc[est], market.iloc[est]
        m = y.notna() & x.notna()
        if m.sum() < est_window // 2:
            continue
        beta = np.cov(y[m], x[m])[0, 1] / np.var(x[m], ddof=1)
        win = slice(i - pre, i + post + 1)
        ar = (ret[ev.symbol].iloc[win] - beta * market.iloc[win]).to_numpy()
        if np.isnan(ar).any():
            continue
        paths.append(ar)
        rows.append({"symbol": ev.symbol, "event_session": ev.event_session, "beta": beta,
                     "car_0_post": float(ar[pre:].sum()), "car_1_post": float(ar[pre + 1:].sum()),
                     "ar_0": float(ar[pre])})
    if not paths:
        return {"n_events": 0}
    P = np.vstack(paths)
    per_event = pd.DataFrame(rows)
    return {
        "n_events": len(per_event),
        "mean_ar": pd.Series(P.mean(axis=0), index=rel),
        "mean_car": pd.Series(P.mean(axis=0).cumsum(), index=rel),
        "per_event": per_event,
        "car_0_post_ci": bootstrap_ci(per_event["car_0_post"], seed=seed, mean_block=1),
        "drift_1_post_ci": bootstrap_ci(per_event["car_1_post"], seed=seed, mean_block=1),
    }
