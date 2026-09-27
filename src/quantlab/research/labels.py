"""Forward-looking labels (targets). The ONLY module where future returns are computed.

Kept separate from :mod:`quantlab.features` on purpose — a test asserts the feature package never
imports this module, so a target cannot leak into a predictor by accident.

Timing convention for a decision made with information available after the close of session t:

* ``next_open``:  enter at open of t+1, exit at open of t+1+h  → return open[t+1+h]/open[t+1] - 1
* ``next_close``: enter at close of t+1, exit at close of t+1+h → return close[t+1+h]/close[t+1] - 1

Both use the forward-adjusted (total-return) panel so dividends are included. The label at t is NaN
when the exit falls outside the data — never extrapolated.
"""

from __future__ import annotations

import pandas as pd

Panel = dict[str, pd.DataFrame]

EXECUTIONS = ("next_open", "next_close")


def forward_return(panel: Panel, horizon: int, execution: str = "next_open") -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1 session")
    if execution == "next_open":
        px = panel["open"]
    elif execution == "next_close":
        px = panel["close"]
    else:
        raise ValueError(f"execution must be one of {EXECUTIONS}")
    entry = px.shift(-1)
    exit_ = px.shift(-(1 + horizon))
    return exit_ / entry - 1.0


def entry_exit_sessions(index: pd.DatetimeIndex, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Entry and exit session for a decision at each session (NaT past the end of data)."""
    idx = pd.Series(index, index=index)
    return idx.shift(-1), idx.shift(-(1 + horizon))
