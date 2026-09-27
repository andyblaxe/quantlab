"""Forward (paper) testing: record every recommended trade and compare with historical expectations.

Every paper trade is written to the append-only ``paper_trades`` registry table as an OPEN event and
a later CLOSE event, carrying: timestamp, signal, market state, recommended trade, the bid/ask that
was actually available, the simulated fill, position size, reason, expected value and distribution,
exit, realised result, slippage versus the quote mid, and the strategy (spec) version.

Only signals whose catalog status is PAPER_TRADING may open paper trades, and every order passes the
RiskGate. Monitoring compares realised forward results with the historical predictive distribution
and flags degradation (it recommends DEGRADED; a human applies it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.paper.broker import Order, PaperBroker
from quantlab.research.acceptance import AcceptanceCriteria
from quantlab.research.catalog import latest_record
from quantlab.research.registry import Registry
from quantlab.risk.gate import GateDecision, ProposedTrade, RiskGate
from quantlab.risk.metrics import PortfolioState, Position


class NotPaperTradable(PermissionError):
    pass


@dataclass
class ForwardTester:
    reg: Registry
    broker: PaperBroker
    gate: RiskGate

    def _state(self) -> PortfolioState:
        pos = [Position(s, q, self.broker.quote(s).mid) for s, q in self.broker.positions().items()]
        eq = self.broker.cash() + sum(p.market_value for p in pos)
        return PortfolioState(eq, self.broker.cash(), pos, peak_equity=max(eq, self.broker.starting_cash))

    def open(self, signal_id: str, symbol: str, quantity: float, reason: str, market_state: dict,
             expected: dict, kind: str = "etf", sector: str = "UNKNOWN") -> dict:
        rec = latest_record(self.reg, signal_id)
        if rec is None or rec["Status"] not in ("PAPER_TRADING", "LIVE_ELIGIBLE"):
            raise NotPaperTradable(f"{signal_id} is {rec['Status'] if rec else 'unknown'}; paper trading requires PAPER_TRADING")
        q = self.broker.quote(symbol)
        decision: GateDecision = self.gate.evaluate(ProposedTrade(symbol, quantity, q.ask, kind=kind, sector=sector,
                                                                  signal_id=signal_id), self._state())
        base = {"event": "OPEN", "signal_id": signal_id, "symbol": symbol, "timestamp": str(q.ts), "reason": reason,
                "market_state": market_state, "recommended_quantity": quantity, "quote": {"bid": q.bid, "ask": q.ask, "source": q.source},
                "expected": expected, "strategy_version": rec.get("Hypothesis_ID"), "risk_decision": decision.to_dict()}
        if decision.decision == "REJECT":
            rid = self.reg.append("paper_trades", {"signal_id": signal_id, "symbol": symbol}, {**base, "status": "RISK_REJECTED"})
            return {"id": rid, "status": "RISK_REJECTED", "reasons": decision.reasons}
        fill = self.broker.submit(Order(symbol, decision.approved_quantity, signal_id=signal_id))
        slippage = (fill.price / q.mid - 1) if fill.price else None
        rid = self.reg.append("paper_trades", {"signal_id": signal_id, "symbol": symbol},
                              {**base, "status": fill.status, "fill": fill.to_dict(), "position_size": fill.quantity,
                               "slippage_vs_mid": slippage})
        return {"id": rid, "status": fill.status, "fill_price": fill.price, "quantity": fill.quantity, "slippage_vs_mid": slippage}

    def close(self, open_id: str, reason: str) -> dict:
        op = self.reg.get("paper_trades", open_id)["payload"]
        if op.get("status") != "FILLED":
            raise ValueError(f"{open_id} is not an open filled paper trade")
        sym, qty = op["symbol"], op["fill"]["quantity"]
        q = self.broker.quote(sym)
        fill = self.broker.submit(Order(sym, -qty, signal_id=op["signal_id"]))
        entry = op["fill"]["price"]
        gross = (fill.price / entry - 1) * np.sign(qty) if fill.price else None
        comm = op["fill"]["commission"] + fill.commission
        net = gross - comm / abs(qty * entry) if gross is not None else None
        rid = self.reg.append("paper_trades", {"signal_id": op["signal_id"], "symbol": sym},
                              {"event": "CLOSE", "open_id": open_id, "timestamp": str(q.ts), "reason": reason,
                               "quote": {"bid": q.bid, "ask": q.ask}, "fill": fill.to_dict(), "realized_gross": gross,
                               "realized_net": net, "slippage_vs_mid": (fill.price / q.mid - 1) if fill.price else None})
        return {"id": rid, "realized_net": net, "realized_gross": gross}


def forward_results(reg: Registry, signal_id: str) -> pd.DataFrame:
    rows = [r["payload"] for r in reg.find("paper_trades", signal_id=signal_id) if r["payload"].get("event") == "CLOSE"]
    return pd.DataFrame(rows)


def compare_with_history(forward_net: pd.Series, historical_net: pd.Series, n_boot: int = 5000, seed: int = 0) -> dict:
    """Where does the forward mean sit in the distribution of historical means over the same n?"""
    f = pd.Series(forward_net, dtype=float).dropna()
    h = pd.Series(historical_net, dtype=float).dropna().to_numpy()
    if len(f) == 0 or len(h) < 10:
        return {"status": "INSUFFICIENT_DATA", "n_forward": int(len(f))}
    rng = np.random.default_rng(seed)
    means = h[rng.integers(0, len(h), size=(n_boot, len(f)))].mean(axis=1)
    pct = float(np.mean(means <= f.mean()))
    return {"status": "OK", "n_forward": int(len(f)), "forward_mean": float(f.mean()), "historical_mean": float(h.mean()),
            "percentile_of_forward_mean": pct, "predictive_90": [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))],
            "below_5th_percentile": pct < 0.05}


def cusum_degradation(forward_net: pd.Series, expected_mean: float, expected_sd: float, k: float = 0.5,
                      h: float = 4.0) -> dict:
    """One-sided CUSUM on standardised shortfall vs expectation; alarm ⇒ recommend DEGRADED."""
    z = (pd.Series(forward_net, dtype=float).dropna() - expected_mean) / expected_sd
    s, path = 0.0, []
    for v in z:
        s = max(0.0, s - v - k)
        path.append(s)
    alarm = bool(path and max(path) > h)
    return {"alarm": alarm, "max_statistic": float(max(path)) if path else 0.0, "threshold": h,
            "recommendation": "DEGRADED" if alarm else "NO_ACTION"}


def live_eligibility(reg: Registry, signal_id: str, historical_net: pd.Series, modelled_slippage: float,
                     criteria: AcceptanceCriteria = AcceptanceCriteria()) -> dict:
    """Check forward-test requirements for LIVE_ELIGIBLE. Returns criteria results; does not transition."""
    fr = forward_results(reg, signal_id)
    opens = [r["payload"] for r in reg.find("paper_trades", signal_id=signal_id) if r["payload"].get("event") == "OPEN"]
    days = 0
    if opens and len(fr):
        days = int(np.busday_count(pd.Timestamp(opens[0]["timestamp"]).date(), pd.Timestamp(fr["timestamp"].iloc[-1]).date()))
    cmp = compare_with_history(fr["realized_net"] if len(fr) else pd.Series(dtype=float), historical_net)
    slip = pd.Series([o.get("slippage_vs_mid") for o in opens], dtype=float).abs().mean() if opens else np.nan
    res: dict[str, Any] = {
        "paper_days": {"value": days, "required": criteria.live_min_paper_days, "passed": days >= criteria.live_min_paper_days},
        "paper_trades": {"value": int(len(fr)), "required": criteria.live_min_paper_trades,
                         "passed": len(fr) >= criteria.live_min_paper_trades},
        "forward_within_expectation": {"value": cmp.get("percentile_of_forward_mean"),
                                       "required": f">= {criteria.live_min_forward_percentile}",
                                       "passed": cmp.get("status") == "OK" and not cmp["below_5th_percentile"]},
        "slippage": {"value": slip, "required": f"<= {criteria.live_max_slippage_ratio} x {modelled_slippage}",
                     "passed": bool(np.isfinite(slip) and slip <= criteria.live_max_slippage_ratio * modelled_slippage)},
    }
    res["eligible"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    return res
