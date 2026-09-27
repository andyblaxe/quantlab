"""The signal catalog: one persistent, versioned record per researched signal.

Records are *appended* as new versions (``SR-…`` rows); the latest version is the current view.
Status changes go through :func:`transition`, which enforces the allowed lifecycle:

    EXPERIMENTAL → VALIDATING → ACCEPTED → PAPER_TRADING → LIVE_ELIGIBLE
         │              │           │            │               │
         └──→ REJECTED ←┘           └──→ DEGRADED ←──────────────┘ ──→ RETIRED

* REJECTED and RETIRED are terminal. A rejected idea can only come back as a *new* registered
  hypothesis (which counts toward the multiple-testing burden).
* PAPER_TRADING requires explicit human approval; LIVE_ELIGIBLE requires forward-test evidence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from quantlab.research.registry import Registry


class SignalStatus(StrEnum):
    EXPERIMENTAL = "EXPERIMENTAL"
    VALIDATING = "VALIDATING"
    ACCEPTED = "ACCEPTED"
    PAPER_TRADING = "PAPER_TRADING"
    LIVE_ELIGIBLE = "LIVE_ELIGIBLE"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


S = SignalStatus
ALLOWED: dict[SignalStatus, set[SignalStatus]] = {
    S.EXPERIMENTAL: {S.VALIDATING, S.REJECTED},
    S.VALIDATING: {S.ACCEPTED, S.REJECTED},
    S.ACCEPTED: {S.PAPER_TRADING, S.DEGRADED, S.RETIRED},
    S.PAPER_TRADING: {S.LIVE_ELIGIBLE, S.DEGRADED, S.RETIRED},
    S.LIVE_ELIGIBLE: {S.DEGRADED, S.RETIRED},
    S.DEGRADED: {S.ACCEPTED, S.PAPER_TRADING, S.RETIRED},
    S.REJECTED: set(),
    S.RETIRED: set(),
}
REQUIRES_APPROVAL = {S.PAPER_TRADING, S.LIVE_ELIGIBLE}

# The fields every signal record carries (None until the relevant analysis has been performed).
RECORD_FIELDS = [
    "Signal_ID", "Name", "Description", "Hypothesis", "Hypothesis_ID", "Instrument_Universe", "Holding_Period",
    "Direction", "Features", "Expected_Return", "Median_Return", "Win_Rate", "Average_Win", "Average_Loss",
    "Profit_Factor", "Information_Coefficient", "Sharpe", "Sortino", "Max_Drawdown", "Volatility", "Sample_Size",
    "Effective_Sample_Size", "P_Value", "Adjusted_P_Value", "Turnover", "Estimated_Transaction_Cost",
    "Estimated_Slippage", "Capacity_Estimate", "Regime_Performance", "Correlation_With_Other_Signals",
    "In_Sample_Performance", "Validation_Performance", "Out_Of_Sample_Performance", "Walk_Forward_Performance",
    "Monte_Carlo_Results", "Paper_Trading_Performance", "Signal_Decay", "Last_Validated", "Status",
    "Data_Label", "Data_Flags", "Experiment_IDs", "Notes",
]


class InvalidTransition(ValueError):
    pass


def signal_id_for(hypothesis_id: str) -> str:
    return "SIG-" + hypothesis_id.split("-", 1)[1]


def latest_record(reg: Registry, signal_id: str) -> dict | None:
    rows = reg.find("signal_records", order="DESC", limit=1, signal_id=signal_id)
    return rows[0]["payload"] if rows else None


def all_signals(reg: Registry) -> list[dict]:
    """Latest version of every signal record."""
    latest: dict[str, dict] = {}
    for row in reg.find("signal_records"):
        latest[row["signal_id"]] = row["payload"]
    return list(latest.values())


def upsert_record(reg: Registry, signal_id: str, updates: dict[str, Any], reason: str) -> str:
    """Append a new version merging ``updates`` into the latest record. Status cannot be set here."""
    if "Status" in updates:
        raise ValueError("use transition() to change status")
    unknown = set(updates) - set(RECORD_FIELDS)
    if unknown:
        raise ValueError(f"unknown signal record fields: {unknown}")
    cur = latest_record(reg, signal_id) or {f: None for f in RECORD_FIELDS}
    version = int(cur.get("_version") or 0) + 1
    rec = {**cur, **updates, "Signal_ID": signal_id, "_version": version, "_reason": reason}
    if rec.get("Status") is None:
        rec["Status"] = SignalStatus.EXPERIMENTAL.value
        reg.status_event("signal", signal_id, SignalStatus.EXPERIMENTAL, "signal created")
    return reg.append("signal_records", {"signal_id": signal_id, "version": version, "status": rec["Status"]}, rec)


def transition(reg: Registry, signal_id: str, to: SignalStatus, reason: str, evidence: dict | None = None,
               approved_by: str | None = None) -> str:
    cur_rec = latest_record(reg, signal_id)
    if cur_rec is None:
        raise KeyError(f"no signal record {signal_id}")
    cur = SignalStatus(cur_rec["Status"])
    to = SignalStatus(to)
    if to not in ALLOWED[cur]:
        raise InvalidTransition(f"{signal_id}: {cur} → {to} is not an allowed transition")
    if to in REQUIRES_APPROVAL and not approved_by:
        raise InvalidTransition(f"{signal_id}: {to} requires explicit human approval (approved_by)")
    reg.status_event("signal", signal_id, to, reason, from_status=cur.value, evidence=evidence or {},
                     approved_by=approved_by)
    version = int(cur_rec.get("_version") or 0) + 1
    rec = {**cur_rec, "Status": to.value, "_version": version, "_reason": reason}
    rid = reg.append("signal_records", {"signal_id": signal_id, "version": version, "status": to.value}, rec)
    reg.journal("signal_status_change", reason=reason, signal_id=signal_id, from_status=cur.value,
                to_status=to.value, approved_by=approved_by)
    return rid
