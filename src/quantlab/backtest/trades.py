"""Event/trade backtest engine: one row per trade, gross and net of costs.

Input is a boolean/sign signal panel (session × symbol): +1 = go long, -1 = go short, 0/NaN = no
trade. **No trade is always an acceptable output.** A signal at session t (known after t's close)
enters at the next open (or close) and exits ``holding`` sessions later at the same time of day.

Feasibility rules (never assume fills that could not have happened):

* no entry if the entry bar is missing or has zero volume — recorded as ``unfilled``;
* if the exit bar is missing, exit at the next available bar (recorded as ``delayed_exit``);
* optionally, no overlapping trades in the same symbol (``allow_overlap=False``).

Returns use the forward-adjusted panel (so dividends and splits are included); costs use the cost
model's round-trip fraction for the configured trade notional, with liquidity measured by trailing
dollar volume known at decision time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.calendar import get_calendar


@dataclass(frozen=True)
class TradeConfig:
    holding: int = 5
    execution: str = "next_open"
    notional: float = 10_000.0  # dollars per trade, for cost/capacity purposes
    allow_overlap: bool = False
    adv_window: int = 20
    vol_window: int = 21


TRADE_COLUMNS = ["symbol", "signal_session", "decision_available_at", "entry_session", "entry_time", "exit_session",
                 "direction", "entry_price_raw", "gross_ret", "cost_frac", "net_ret", "participation", "delayed_exit"]


def generate_trades(panel: dict[str, pd.DataFrame], signal: pd.DataFrame, cfg: TradeConfig,
                    cost: CostModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (trades, unfilled). ``trades`` columns: see TRADE_COLUMNS."""
    if cfg.execution not in ("next_open", "next_close"):
        raise ValueError("execution must be next_open or next_close")
    sig = signal.reindex(panel["close"].index).fillna(0.0)
    symbols = [s for s in sig.columns if s in panel["close"].columns]
    sessions = panel["close"].index
    T = len(sessions)
    px = (panel["open"] if cfg.execution == "next_open" else panel["close"])[symbols].to_numpy(dtype=float)
    raw_px = (panel["raw_open"] if cfg.execution == "next_open" else panel["raw_close"])[symbols].to_numpy(dtype=float)
    vol = panel["raw_volume"][symbols].to_numpy(dtype=float)
    adv = panel["dollar_volume"][symbols].rolling(cfg.adv_window, min_periods=1).mean().to_numpy(dtype=float)
    dvol = panel["ret"][symbols].rolling(cfg.vol_window, min_periods=5).std().to_numpy(dtype=float)
    avail = panel["available_at"][symbols] if "available_at" in panel else None
    cal = get_calendar()
    fill_times = cal.session_opens(sessions) if cfg.execution == "next_open" else cal.session_closes(sessions)
    S = sig[symbols].to_numpy(dtype=float)

    trades, unfilled = [], []
    for j, sym in enumerate(symbols):
        busy_until = -1
        for t in np.nonzero(S[:, j])[0]:
            direction = float(np.sign(S[t, j]))
            e = t + 1
            if e >= T:
                continue  # entry beyond the data
            if not cfg.allow_overlap and e <= busy_until:
                continue
            if np.isnan(px[e, j]) or not (vol[e, j] > 0):
                unfilled.append((sym, sessions[t], "no_tradeable_entry_bar"))
                continue
            x = e + cfg.holding
            if x >= T:
                continue  # outcome unknown: excluded, never extrapolated
            delayed = False
            while x < T and (np.isnan(px[x, j]) or not (vol[x, j] > 0)):
                x += 1
                delayed = True
            if x >= T:
                continue
            decision_avail = avail.iat[t, j] if avail is not None else cal.daily_bar_available_at(sessions[t:t + 1])[0]
            if pd.isna(decision_avail):
                unfilled.append((sym, sessions[t], "signal_on_missing_bar"))
                continue
            if not fill_times[e] > decision_avail:
                raise AssertionError(f"{sym}: entry {fill_times[e]} not after decision availability {decision_avail}")
            gross = direction * (px[x, j] / px[e, j] - 1.0)
            n = cfg.notional
            c = float(cost.round_trip_frac(n, raw_px[e, j], adv[t, j], dvol[t, j]))
            if direction < 0 and not cost.allow_short:
                raise ValueError("short signals require a cost model with allow_short=True")
            if direction < 0:  # stock-borrow fee for the holding period
                c += cost.short_borrow_bps_annual / 1e4 * (x - e) / 252
            part = n / adv[t, j] if adv[t, j] > 0 else np.inf
            trades.append((sym, sessions[t], decision_avail, sessions[e], fill_times[e], sessions[x], direction,
                           raw_px[e, j], gross, c, gross - c, part, delayed))
            busy_until = x
    tr = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    un = pd.DataFrame(unfilled, columns=["symbol", "signal_session", "reason"])
    return tr.sort_values(["entry_session", "symbol"]).reset_index(drop=True), un
