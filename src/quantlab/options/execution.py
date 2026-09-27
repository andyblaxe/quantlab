"""Conservative option execution and a historical multi-leg trade simulator on EOD bid/ask quotes.

Execution rules (defaults deliberately pessimistic):
* buys fill at the ASK, sells at the BID (``spread_capture=0``). Mid fills require explicitly setting
  ``spread_capture=1.0`` and are labelled optimistic in results;
* per-contract commission (+ per-leg minimum), contract multiplier 100;
* no sell into a zero bid, no fill on a missing quote; order size ≤ ``max_oi_frac`` of open interest;
* timing: a decision made with information through session t's close is filled from session t+1's
  quote snapshot — never from the same snapshot that produced the signal;
* expiry: long ITM options are exercised and short ITM options assigned at intrinsic value on the
  underlying's closing price (cash-equivalent P&L), OTM expire worthless;
* early exercise: American short calls that are ITM on the session before an ex-dividend date with
  time value < dividend are flagged as early-assignment risk (not silently ignored).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptionCostModel:
    commission_per_contract: float = 0.65
    min_commission_per_leg: float = 0.0
    spread_capture: float = 0.0  # 0 = cross the full spread (fill at ask/bid); 1 = mid (optimistic)
    max_oi_frac: float = 0.05
    multiplier: int = 100

    def fill_price(self, bid: float, ask: float, side: int) -> float | None:
        """side +1 buy / −1 sell. None when no executable price exists."""
        if not np.isfinite(bid) or not np.isfinite(ask) or ask <= 0 or ask < bid:
            return None
        if side < 0 and bid <= 0:
            return None  # cannot sell into a zero bid
        mid = (bid + ask) / 2
        half = (ask - bid) / 2
        return float(mid + side * half * (1 - self.spread_capture))

    def commission(self, contracts: int) -> float:
        return max(self.min_commission_per_leg, self.commission_per_contract * abs(contracts))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LegSpec:
    right: str
    target: str  # "atm" | "delta:0.25" | "moneyness:1.05"
    qty: int  # contracts (+ long, − short)
    min_dte: int = 1  # choose the first expiry at least this many calendar days out


def _select_contract(chain_t: pd.DataFrame, leg: LegSpec) -> pd.Series | None:
    c = chain_t[(chain_t["right"] == leg.right)]
    if c.empty:
        return None
    dte = (pd.to_datetime(c["expiration"]) - pd.to_datetime(c["session"])).dt.days
    c = c[dte >= leg.min_dte]
    if c.empty:
        return None
    exp = c["expiration"].min()
    c = c[c["expiration"] == exp]
    S = float(c["underlying_price"].iloc[0])
    if leg.target == "atm":
        return c.iloc[(c["strike"] - S).abs().argsort().iloc[0]]
    kind, _, val = leg.target.partition(":")
    if kind == "moneyness":
        return c.iloc[(c["strike"] - S * float(val)).abs().argsort().iloc[0]]
    if kind == "delta" and "model_delta" in c:
        target = float(val) * (1 if leg.right == "C" else -1)
        return c.iloc[(c["model_delta"] - target).abs().argsort().iloc[0]]
    raise ValueError(f"unsupported leg target {leg.target}")


def simulate_option_trade(quotes: pd.DataFrame, underlying_close: pd.Series, signal_session: pd.Timestamp,
                          legs: list[LegSpec], cost: OptionCostModel, hold_sessions: int | None = None,
                          ex_div_dates: dict | None = None) -> dict:
    """Simulate one multi-leg trade from historical EOD quotes.

    ``quotes`` : OPTION_QUOTES_EOD rows for one underlying (optionally enriched with model_delta).
    ``underlying_close`` : raw close indexed by session (for expiry settlement).
    Entry uses the first quote session AFTER ``signal_session``; exit after ``hold_sessions`` quote
    sessions (at bid/ask) or at expiry (intrinsic), whichever first.
    """
    sessions = sorted(pd.to_datetime(quotes["session"].unique()))
    after = [s for s in sessions if s > pd.Timestamp(signal_session)]
    if not after:
        return {"status": "NO_ENTRY_QUOTES"}
    entry_s = after[0]
    chain_e = quotes[pd.to_datetime(quotes["session"]) == entry_s]
    chosen, cash, comm, notes = [], 0.0, 0.0, []
    for leg in legs:
        row = _select_contract(chain_e, leg)
        if row is None:
            return {"status": "NO_CONTRACT", "leg": asdict(leg)}
        if abs(leg.qty) > cost.max_oi_frac * max(float(row["open_interest"]), 0):
            return {"status": "LIQUIDITY_REJECT", "leg": asdict(leg), "open_interest": float(row["open_interest"])}
        px = cost.fill_price(float(row["bid"]), float(row["ask"]), int(np.sign(leg.qty)))
        if px is None:
            return {"status": "NO_EXECUTABLE_PRICE", "leg": asdict(leg)}
        cash -= leg.qty * px * cost.multiplier
        comm += cost.commission(leg.qty)
        chosen.append((leg, row, px))
    expiry = min(pd.Timestamp(r["expiration"]) for _, r, _ in chosen)
    later = [s for s in sessions if s > entry_s]
    exit_s = later[hold_sessions - 1] if hold_sessions and len(later) >= hold_sessions else None

    def _exit_quotes(when):
        chain_x = quotes[pd.to_datetime(quotes["session"]) == when]
        found = []
        for leg, row, _ in chosen:
            m = chain_x[(chain_x["right"] == leg.right) & (chain_x["strike"] == row["strike"]) &
                        (chain_x["expiration"] == row["expiration"])]
            if m.empty:
                return None
            found.append(m.iloc[0])
        return found

    exit_rows = None
    if exit_s is not None and exit_s < expiry:
        # a missing exit quote delays the exit (recorded); the trade is never silently dropped
        for when in [s for s in sessions if exit_s <= s < expiry]:
            exit_rows = _exit_quotes(when)
            if exit_rows is not None:
                if when != exit_s:
                    notes.append(f"exit delayed from {exit_s.date()} to {when.date()} (missing quotes)")
                exit_s = when
                break
        if exit_rows is None:
            notes.append("no exit quotes before expiry; held to expiry settlement")
    if exit_rows is None:
        # settle at expiry on the underlying close
        settle_px = underlying_close.loc[:expiry]
        if settle_px.empty:
            return {"status": "NO_SETTLEMENT_PRICE"}
        S_T = float(settle_px.iloc[-1])
        for leg, row, _ in chosen:
            intr = max(S_T - row["strike"], 0) if leg.right == "C" else max(row["strike"] - S_T, 0)
            cash += leg.qty * intr * cost.multiplier
            if intr > 0:
                notes.append(f"{'exercised' if leg.qty > 0 else 'assigned'} {leg.right}{row['strike']} at expiry (S_T={S_T:.2f})")
        exit_kind, exit_when = "expiry", expiry
    else:
        for (leg, row, _), m in zip(chosen, exit_rows):
            px = cost.fill_price(float(m["bid"]), float(m["ask"]), -int(np.sign(leg.qty)))
            if px is None:
                px = 0.0 if leg.qty > 0 else float(m["ask"])  # worthless long / must buy back at ask
                notes.append(f"no bid for {leg.right}{row['strike']} at exit; valued conservatively")
            cash += leg.qty * px * cost.multiplier
            comm += cost.commission(leg.qty)
        exit_kind, exit_when = "rule", exit_s
    if ex_div_dates:
        for leg, row, _ in chosen:
            if leg.right == "C" and leg.qty < 0:
                for d in ex_div_dates.get(row.get("underlying", ""), []):
                    if entry_s < pd.Timestamp(d) <= exit_when:
                        notes.append(f"early-assignment risk on short call {row['strike']} before ex-div {d}")
    debit = -sum(leg.qty * px * cost.multiplier for leg, _, px in chosen)
    return {"status": "FILLED", "entry_session": entry_s, "exit_session": exit_when, "exit_kind": exit_kind,
            "legs": [{"right": l.right, "strike": float(r["strike"]), "expiration": str(pd.Timestamp(r["expiration"]).date()),
                      "qty": l.qty, "fill": px} for l, r, px in chosen],
            "entry_cash_flow": debit, "pnl_gross": float(cash), "commissions": float(comm), "pnl_net": float(cash - comm),
            "return_on_debit": float((cash - comm) / abs(debit)) if debit else None,
            "optimistic_mid_fills": cost.spread_capture > 0, "notes": notes}
