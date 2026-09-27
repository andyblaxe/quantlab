"""Portfolio (target-weight) backtest engine.

Timing contract
---------------
``target_weights`` row *t* is a decision made with information available after the close of
session *t* (i.e. at the bar's ``available_at``). It is executed at the **next** opportunity:

* ``next_open``  — at the raw open of session t+1;
* ``next_close`` — at the raw close of session t+1.

Trading at the close of *t* on a signal computed from that same close is not offered: the close is
not known until after the auction. The engine asserts that every fill time is strictly after the
decision's availability time.

Accounting
----------
Positions are held in **shares at raw (as-traded) prices**. On an ex-date, dividends are credited
per pre-split share and then share counts are multiplied by the split ratio — exactly what a
brokerage statement would show. Fills cannot occur on a missing bar or a zero-volume bar, are
capped at ``max_participation`` of trailing average dollar volume (the unfilled remainder is
recorded), and pay the cost model's spread + slippage + impact in the fill price plus commission.

GROSS results come from a separate frictionless run with identical feasibility limits, so
``gross − net`` isolates the cost of trading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.calendar import get_calendar
from quantlab.data.pit import LookAheadError


@dataclass(frozen=True)
class BacktestConfig:
    capital: float = 100_000.0
    execution: str = "next_open"
    cost_model: CostModel = field(default_factory=CostModel)
    fractional_shares: bool = True
    max_gross_leverage: float = 1.0
    adv_window: int = 20
    vol_window: int = 21
    rebalance_threshold: float = 0.0  # skip trades smaller than this fraction of equity

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost_model"] = self.cost_model.to_dict()
        return d


@dataclass
class SimulationResult:
    daily: pd.DataFrame  # equity, ret, cash, gross/net exposure, turnover, costs ...
    trades: pd.DataFrame  # fill ledger
    config: BacktestConfig


@dataclass
class BacktestResult:
    gross: SimulationResult
    net: SimulationResult

    @property
    def net_returns(self) -> pd.Series:
        return self.net.daily["ret"]

    @property
    def gross_returns(self) -> pd.Series:
        return self.gross.daily["ret"]


def _check_timing(sessions: pd.DatetimeIndex, available_at: pd.DataFrame | None, execution: str) -> None:
    """Every decision's availability must precede the corresponding fill time."""
    if len(sessions) < 2:
        return
    cal = get_calendar()
    fill_times = cal.session_opens(sessions[1:]) if execution == "next_open" else cal.session_closes(sessions[1:])
    if available_at is not None:
        decision_avail = available_at.reindex(sessions[:-1]).max(axis=1)
    else:
        decision_avail = pd.Series(cal.daily_bar_available_at(sessions[:-1]), index=sessions[:-1])
    ok = pd.Series(fill_times, index=sessions[:-1]) > decision_avail
    if not ok[decision_avail.notna()].all():
        bad = ok[~ok & decision_avail.notna()].index[:3]
        raise LookAheadError(f"fills would precede information availability at decisions {list(bad)}")


def simulate(panel: dict[str, pd.DataFrame], target_weights: pd.DataFrame, cfg: BacktestConfig) -> SimulationResult:
    if cfg.execution not in ("next_open", "next_close"):
        raise ValueError("execution must be next_open or next_close")
    cm = cfg.cost_model
    w = target_weights.sort_index()
    symbols = list(w.columns)
    sessions = panel["raw_close"].index
    w = w.reindex(sessions)
    gross_lev = w.abs().sum(axis=1)
    if (gross_lev > cfg.max_gross_leverage + 1e-9).any():
        raise ValueError(f"target gross leverage exceeds {cfg.max_gross_leverage} on "
                         f"{list(gross_lev[gross_lev > cfg.max_gross_leverage + 1e-9].index[:3])}")
    if not cm.allow_short and (w < -1e-12).any().any():
        raise ValueError("negative target weights but the cost model does not allow shorting")
    _check_timing(sessions, panel.get("available_at", pd.DataFrame()).reindex(columns=symbols)
                  if "available_at" in panel else None, cfg.execution)

    ro = panel["raw_open"][symbols].to_numpy(dtype=float)
    rc = panel["raw_close"][symbols].to_numpy(dtype=float)
    vol_raw = panel["raw_volume"][symbols].to_numpy(dtype=float)
    ratio = panel["split_ratio"][symbols].to_numpy(dtype=float)
    div = panel["dividend"][symbols].to_numpy(dtype=float)
    adv = panel["dollar_volume"][symbols].rolling(cfg.adv_window, min_periods=1).mean().to_numpy(dtype=float)
    dvol = panel["ret"][symbols].rolling(cfg.vol_window, min_periods=5).std().to_numpy(dtype=float)
    W = w.to_numpy(dtype=float)
    exec_px = ro if cfg.execution == "next_open" else rc
    mark = pd.DataFrame(rc).ffill().to_numpy()

    T, N = rc.shape
    shares = np.zeros(N)
    cash = cfg.capital
    rows = []
    fills = []
    prev_equity = cfg.capital
    for i in range(T):
        dividends = 0.0
        borrow = 0.0
        # 1) corporate actions effective this session (before the open)
        if i > 0:
            d = np.nan_to_num(div[i]) * shares
            dividends = float(d.sum())
            cash += dividends
            shares = shares * np.nan_to_num(ratio[i], nan=1.0)
            short_val = float(np.nansum(np.where(shares < 0, -shares * mark[i - 1], 0.0)))
            borrow = short_val * cm.short_borrow_bps_annual / 1e4 / 252
            cash -= borrow
        # 2) execute the decision made at the previous session's close
        traded_value = spread_cost = commission = unfilled = 0.0
        n_fills = 0
        if i > 0 and not np.all(np.isnan(W[i - 1])):
            p = exec_px[i]
            val_px = np.where(np.isnan(p), mark[i - 1], p)
            equity_now = cash + float(np.nansum(shares * val_px))
            target = np.nan_to_num(W[i - 1]) * equity_now
            tradeable = ~np.isnan(p) & (np.nan_to_num(vol_raw[i]) > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                desired = np.where(tradeable, target / p, shares)
            if not cfg.fractional_shares:
                desired = np.trunc(desired)
            delta = np.where(tradeable, desired - shares, 0.0)
            notional = np.abs(delta) * np.nan_to_num(p)
            if cfg.rebalance_threshold > 0:
                small = notional < cfg.rebalance_threshold * equity_now
                delta[small] = 0.0
                notional[small] = 0.0
            cap = cm.max_participation * np.nan_to_num(adv[i - 1])
            over = notional > cap
            if over.any():
                scale = np.where(over, cap / np.where(notional > 0, notional, 1), 1.0)
                new_delta = delta * scale
                if not cfg.fractional_shares:
                    new_delta = np.trunc(new_delta)
                unfilled = float(np.sum((np.abs(delta) - np.abs(new_delta)) * np.nan_to_num(p)))
                delta = new_delta
                notional = np.abs(delta) * np.nan_to_num(p)
            frac = cm.price_impact_frac(notional, adv[i - 1], dvol[i - 1])
            comm = cm.commission(notional, delta)
            fill_px = np.nan_to_num(p) * (1 + np.sign(delta) * frac)
            cash -= float(np.sum(delta * fill_px)) + float(np.sum(comm))
            shares = shares + delta
            traded_value = float(notional.sum())
            spread_cost = float(np.sum(notional * frac))
            commission = float(np.sum(comm))
            for j in np.nonzero(delta)[0]:
                fills.append((sessions[i], symbols[j], float(delta[j]), float(p[j]), float(fill_px[j]),
                              float(notional[j] * frac[j]), float(comm[j])))
                n_fills += 1
        # 3) mark to market at the close
        pos_val = shares * mark[i]
        equity = cash + float(np.nansum(pos_val))
        gross_exp = float(np.nansum(np.abs(pos_val)))
        net_exp = float(np.nansum(pos_val))
        rows.append((sessions[i], equity, equity / prev_equity - 1.0 if i else 0.0, cash, gross_exp / equity if equity else np.nan,
                     net_exp / equity if equity else np.nan, traded_value / prev_equity if prev_equity else 0.0,
                     spread_cost, commission, borrow, dividends, unfilled, n_fills))
        prev_equity = equity
        if equity <= 0:
            # ruin: stop trading, keep recording the (non-positive) equity
            W[i:] = 0.0
    daily = pd.DataFrame(rows, columns=["session", "equity", "ret", "cash", "gross_exposure", "net_exposure",
                                        "turnover", "cost_spread_slip_impact", "cost_commission", "cost_borrow",
                                        "dividends", "unfilled_value", "n_fills"]).set_index("session")
    trades = pd.DataFrame(fills, columns=["session", "symbol", "shares", "exec_price", "fill_price", "impact_cost",
                                          "commission"])
    return SimulationResult(daily, trades, cfg)


def run_portfolio_backtest(panel: dict[str, pd.DataFrame], target_weights: pd.DataFrame,
                           cfg: BacktestConfig) -> BacktestResult:
    net = simulate(panel, target_weights, cfg)
    gross = simulate(panel, target_weights, replace(cfg, cost_model=cfg.cost_model.zero()))
    return BacktestResult(gross=gross, net=net)
