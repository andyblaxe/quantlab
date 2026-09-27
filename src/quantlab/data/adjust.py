"""Point-in-time price adjustment and panel construction.

Why not use vendor "adjusted close"? Vendors back-adjust the *entire* history every time a split
or dividend occurs, so the adjusted price you download for 2010 depends on dividends paid in 2024.
Ratio features are invariant to that, but level-based features (price filters, dollar volume,
"price < $5") are not — and the series itself is revised data.

Instead we build a **forward** total-return index from raw prices and corporate actions:

    TRI_t = TRI_{t-1} * (close_t * split_ratio_t + dividend_t) / close_{t-1}

which uses only information at or before t. Conventions:

* ``split_ratio`` = new shares per old share (2-for-1 → 2.0), effective on ``ex_date``.
* ``dividend`` = cash per share on a *pre-split* basis when a split and dividend share an ex-date.
* Adjusted OHLC = raw OHLC × (TRI_t / close_t); adjusted volume = raw volume / cumulative split
  ratio (so share counts are comparable across splits). Dollar volume needs no adjustment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PANEL_FIELDS = ("open", "high", "low", "close", "volume")


def wide(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    """Long bars → wide (session × symbol) frame for one field."""
    out = bars.pivot(index="session", columns="symbol", values=field).sort_index()
    out.columns.name = None
    return out


def action_panels(actions: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split-ratio and dividend panels aligned to (sessions × symbols); 1.0 / 0.0 where no action."""
    ratio = pd.DataFrame(1.0, index=index, columns=columns)
    div = pd.DataFrame(0.0, index=index, columns=columns)
    if actions is None or len(actions) == 0:
        return ratio, div
    a = actions[actions["symbol"].isin(columns) & actions["ex_date"].isin(index)]
    for (sym, ex, kind), grp in a.groupby(["symbol", "ex_date", "action"]):
        if kind == "split":
            ratio.loc[ex, sym] *= float(np.prod(grp["value"]))
        elif kind == "dividend":
            div.loc[ex, sym] += float(grp["value"].sum())
        else:
            raise ValueError(f"unknown corporate action type {kind!r}")
    return ratio, div


def total_return(close: pd.DataFrame, ratio: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    """Daily total simple returns.

    NaN on a symbol's first observation and on missing sessions. After a gap (e.g. a trading
    halt) the return on the reappearance day spans the whole gap, measured from the last valid
    close — it is never silently set to zero.
    """
    prev = close.ffill().shift(1)
    return ((close * ratio + div) / prev - 1.0).where(close.notna())


def build_panel(bars: pd.DataFrame, actions: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Construct the research panel from unadjusted bars and corporate actions.

    Returns wide frames keyed by: raw_open/high/low/close/volume, open/high/low/close (adjusted,
    forward, PIT-safe), volume (split-adjusted), ret (total daily return close-to-close),
    dollar_volume, split_ratio, dividend, available_at.
    """
    raw = {f: wide(bars, f) for f in PANEL_FIELDS}
    close = raw["close"]
    ratio, div = action_panels(actions if actions is not None else pd.DataFrame(), close.index, close.columns)
    ret = total_return(close, ratio, div)
    # forward TRI anchored at the first valid close of each symbol
    growth = (1.0 + ret).where(close.notna())
    tri = growth.fillna(1.0).cumprod()
    first_close = close.bfill().iloc[0]
    tri = tri * first_close
    tri = tri.where(close.notna())
    factor = tri / close
    cum_split = ratio.cumprod()
    panel = {f"raw_{f}": raw[f] for f in PANEL_FIELDS}
    panel.update({
        "open": raw["open"] * factor,
        "high": raw["high"] * factor,
        "low": raw["low"] * factor,
        "close": tri,
        "volume": raw["volume"] * cum_split.iloc[0] / cum_split,
        "ret": ret,
        "dollar_volume": raw["close"] * raw["volume"],
        "split_ratio": ratio,
        "dividend": div,
    })
    avail = bars.pivot(index="session", columns="symbol", values="available_at").sort_index()
    avail.columns.name = None
    panel["available_at"] = avail
    return panel
