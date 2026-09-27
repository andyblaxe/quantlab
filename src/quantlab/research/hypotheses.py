"""Hypothesis specification and pre-registration.

A hypothesis must be registered — hashed and written to the append-only registry — **before** any
evaluation touches it. Evaluation code must present the registered ID *and* the spec; if the spec
no longer matches the registered hash, evaluation is refused. This makes "try variants until one
works, then register the winner" visible: every variant is a separate registered hypothesis and
counts toward the multiple-testing burden.

Hypothesis lifecycle (research process; distinct from the signal catalog status):

    REGISTERED → EVALUATED (train/validation/walk-forward done) → FROZEN (final spec locked)
               → TESTED (untouched test opened once) → CONCLUDED
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from quantlab.research.acceptance import AcceptanceCriteria, loosened_vs_defaults
from quantlab.research.registry import Registry, canonical_json


class Mechanism(StrEnum):
    RISK_PREMIUM = "RISK_PREMIUM"
    BEHAVIORAL = "BEHAVIORAL"
    LIQUIDITY = "LIQUIDITY"
    STRUCTURAL = "STRUCTURAL"
    VOLATILITY = "VOLATILITY"
    INSTITUTIONAL = "INSTITUTIONAL"
    MICROSTRUCTURE = "MICROSTRUCTURE"
    UNKNOWN = "UNKNOWN"  # allowed, but reported as "we do not know why this would exist"


class HypothesisStatus(StrEnum):
    REGISTERED = "REGISTERED"
    EVALUATED = "EVALUATED"
    FROZEN = "FROZEN"
    TESTED = "TESTED"
    CONCLUDED = "CONCLUDED"


class HypothesisSpec(BaseModel):
    model_config = {"frozen": True}

    name: str
    question: str
    statement: str  # the falsifiable claim, in words
    mechanism: Mechanism
    rationale: str
    family: str  # multiple-testing family, e.g. "trend", "mean_reversion", "calendar"
    universe: str
    instrument_type: str = "equity"  # equity | etf | option | index
    strategy: str  # registered strategy/evaluator name
    params: dict[str, Any] = Field(default_factory=dict)
    param_neighbors: list[dict[str, Any]] = Field(default_factory=list)  # pre-declared sensitivity set
    direction: str = "long"  # long | short | long_short | volatility
    holding_period: int = 1  # sessions
    execution: str = "next_open"  # next_open | next_close
    primary_metric: str = "mean_net_trade_return"
    alternative: str = "greater"  # greater | less | two-sided
    cost_profile: str = "retail_etf"
    data_requirements: list[str] = Field(default_factory=lambda: ["bars_daily"])
    generated_by: str = "human"
    exploratory: bool = False
    acceptance: dict[str, Any] = Field(default_factory=lambda: AcceptanceCriteria().to_dict())

    def spec_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()


class DuplicateHypothesis(ValueError):
    def __init__(self, existing_id: str):
        super().__init__(f"identical hypothesis already registered as {existing_id}")
        self.existing_id = existing_id


class UnregisteredHypothesis(PermissionError):
    pass


def register_hypothesis(reg: Registry, spec: HypothesisSpec, reason: str = "") -> str:
    """Pre-register a hypothesis. Must happen before any evaluation."""
    crit = AcceptanceCriteria(**spec.acceptance)
    loose = loosened_vs_defaults(crit)
    if loose and not crit.justification.strip():
        raise ValueError(f"acceptance thresholds {loose} are looser than the project defaults; pre-register a "
                         "methodological justification (AcceptanceCriteria.justification)")
    h = spec.spec_hash()
    existing = reg.find("hypotheses", spec_hash=h)
    if existing:
        raise DuplicateHypothesis(existing[0]["id"])
    hid = reg.append("hypotheses",
                     {"family": spec.family, "name": spec.name, "spec_hash": h, "generated_by": spec.generated_by},
                     {"spec": spec.model_dump(mode="json"), "reason": reason})
    reg.status_event("hypothesis", hid, HypothesisStatus.REGISTERED, reason or "pre-registered before evaluation")
    reg.journal("register_hypothesis", question=spec.question, hypothesis=spec.statement, reason=reason,
                hypothesis_id=hid, spec_hash=h, family=spec.family, generated_by=spec.generated_by)
    return hid


def load_hypothesis(reg: Registry, hid: str) -> HypothesisSpec:
    rec = reg.get("hypotheses", hid)
    spec = HypothesisSpec(**rec["payload"]["spec"])
    if spec.spec_hash() != rec["spec_hash"]:
        raise ValueError(f"{hid}: stored spec does not match its registered hash")
    return spec


def require_registered(reg: Registry, hid: str, spec: HypothesisSpec) -> None:
    """Refuse evaluation of a spec that is not exactly what was registered under ``hid``."""
    try:
        rec = reg.get("hypotheses", hid)
    except KeyError as e:
        raise UnregisteredHypothesis(f"{hid} is not registered; register before evaluating") from e
    if rec["spec_hash"] != spec.spec_hash():
        raise UnregisteredHypothesis(
            f"spec presented for {hid} differs from the registered spec; register the variant as a new hypothesis")


def hypothesis_status(reg: Registry, hid: str) -> str | None:
    return reg.current_status("hypothesis", hid)


def advance_hypothesis(reg: Registry, hid: str, to_status: HypothesisStatus, reason: str, **details: Any) -> str:
    order = list(HypothesisStatus)
    cur = hypothesis_status(reg, hid)
    if cur is None:
        raise UnregisteredHypothesis(hid)
    if order.index(HypothesisStatus(to_status)) <= order.index(HypothesisStatus(cur)):
        raise ValueError(f"{hid}: cannot move from {cur} to {to_status} (lifecycle only moves forward)")
    return reg.status_event("hypothesis", hid, to_status, reason, from_status=cur, **details)


def family_size(reg: Registry, family: str | None = None) -> int:
    """Number of hypotheses ever registered (in a family, or overall) — the multiple-testing burden."""
    return reg.count("hypotheses", family=family) if family else reg.count("hypotheses")
