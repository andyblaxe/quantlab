"""Chart builders (Plotly). Charts only *display* stored research data; they compute nothing new
beyond presentation transforms (cumulative products, rolling windows, histograms).

Style: reference categorical palette (slot 1 blue = NET / primary, slot 2 orange = GROSS),
transparent backgrounds with mid-gray ink so charts read on light and dark pages, a single y-axis
per chart, recessive grid, hover tooltips on by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
ANN = 252


def _style(fig: go.Figure, title: str, ytitle: str = "", yfmt: str | None = None, height: int = 320) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 14}}, height=height, margin={"l": 50, "r": 16, "t": 44, "b": 36},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": GRAY, "size": 12},
        hovermode="x unified", legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right", "yanchor": "bottom"},
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(127,127,127,.4)")
    fig.update_yaxes(title=ytitle, gridcolor="rgba(127,127,127,.18)", zerolinecolor="rgba(127,127,127,.4)",
                     tickformat=yfmt)
    return fig


def equity_curve(daily: pd.DataFrame, title: str = "Equity curve (development period)") -> go.Figure:
    fig = go.Figure()
    for col, color, name in (("gross", ORANGE, "Gross"), ("net", BLUE, "Net of costs")):
        if col in daily:
            eq = (1 + daily[col].fillna(0)).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, line={"color": color, "width": 2},
                                     hovertemplate="%{y:.3f}"))
    return _style(fig, title, "Growth of $1")


def drawdown(daily: pd.DataFrame, col: str = "net") -> go.Figure:
    eq = (1 + daily[col].fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line={"color": BLUE, "width": 1.5}, name="Drawdown",
                               hovertemplate="%{y:.1%}"))
    return _style(fig, "Drawdown (net)", "", ".0%")


def rolling_sharpe(daily: pd.DataFrame, window: int = 252, col: str = "net") -> go.Figure:
    r = daily[col]
    rs = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(ANN)
    fig = go.Figure(go.Scatter(x=rs.index, y=rs, line={"color": BLUE, "width": 2}, name="Rolling Sharpe",
                               hovertemplate="%{y:.2f}"))
    return _style(fig, f"Rolling {window}-session Sharpe (net)", "Sharpe")


def rolling_trade_mean(trades: pd.DataFrame, window: int = 50) -> go.Figure:
    t = trades.sort_values("entry_session")
    m = t["net_ret"].rolling(window, min_periods=max(10, window // 2)).mean()
    fig = go.Figure(go.Scatter(x=t["entry_session"], y=m, line={"color": BLUE, "width": 2}, name="Rolling mean",
                               hovertemplate="%{y:.2%}"))
    return _style(fig, f"Rolling expected return per trade ({window} trades, net)", "", ".2%")


def yearly(daily: pd.DataFrame, col: str = "net") -> go.Figure:
    y = (1 + daily[col].fillna(0)).groupby(daily.index.year).prod() - 1
    fig = go.Figure(go.Bar(x=y.index.astype(str), y=y, marker_color=BLUE, name="Net return",
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    return _style(fig, "Net return by calendar year", "", ".0%")


def trade_distribution(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=trades["gross_ret"], name="Gross", marker_color=ORANGE, opacity=0.55, nbinsx=60))
    fig.add_trace(go.Histogram(x=trades["net_ret"], name="Net", marker_color=BLUE, opacity=0.75, nbinsx=60))
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(tickformat=".1%")
    return _style(fig, "Distribution of per-trade returns", "Trades")


def win_loss(trades: pd.DataFrame) -> go.Figure:
    wins, losses = trades.loc[trades["net_ret"] > 0, "net_ret"], trades.loc[trades["net_ret"] <= 0, "net_ret"]
    fig = go.Figure()
    fig.add_trace(go.Box(y=wins, name=f"Winners (n={len(wins)})", marker_color=AQUA, boxmean=True))
    fig.add_trace(go.Box(y=losses, name=f"Losers (n={len(losses)})", marker_color=ORANGE, boxmean=True))
    fig.update_layout(hovermode="closest", showlegend=False)
    return _style(fig, "Winners vs losers (net)", "", ".1%")


def by_regime(regimes: dict, dimension: str) -> go.Figure | None:
    cells = regimes.get(dimension)
    if not cells:
        return None
    keys = list(cells)
    y = [cells[k]["mean"] for k in keys]
    txt = [f"n={cells[k]['n']}" + (" (small)" if cells[k]["small_sample"] else "") for k in keys]
    fig = go.Figure(go.Bar(x=keys, y=y, marker_color=BLUE, text=txt, textposition="outside",
                           hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _style(fig, f"Mean net trade return by {dimension.replace('_', ' ')}", "", ".2%")


def by_strength(strength: dict) -> go.Figure | None:
    if strength.get("status") != "OK":
        return None
    b = strength["buckets"]
    fig = go.Figure(go.Bar(x=[f"Q{x['quintile']}" for x in b], y=[x["mean_net"] for x in b], marker_color=BLUE,
                           text=[f"n={x['n']}" for x in b], textposition="outside",
                           hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _style(fig, f"Mean net trade return by quintile of {strength['feature']}", "", ".2%")


def mc_distribution(mc: dict, capital: str, key: str = "ending_capital_sample", title: str | None = None) -> go.Figure | None:
    h = mc.get("historical", {}).get(capital)
    if not h:
        return None
    vals = np.asarray(h[key])
    fig = go.Figure(go.Histogram(x=vals, marker_color=BLUE, nbinsx=30, name="Simulated paths (percentiles)",
                                 hovertemplate="%{x}<extra></extra>"))
    if key == "ending_capital_sample":
        fig.add_vline(x=float(capital), line_dash="dot", line_color=GRAY, annotation_text="start")
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(tickformat=".0%" if "drawdown" in key else ",.0f")
    return _style(fig, title or f"Monte Carlo ending capital, start ${int(float(capital)):,}", "Share of percentiles")
