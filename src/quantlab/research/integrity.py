"""Research-integrity promotion gate (permanent project rule; RESEARCH_METHODOLOGY.md §Integrity gate).

A signal may not reach ACCEPTED, PAPER_TRADING or LIVE_ELIGIBLE while its evidence has an unresolved
**material** integrity problem. The rule is enforced in :func:`quantlab.research.catalog.transition`
(and again before paper trades and in the live-eligibility check), never only in reports.

Findings come from two places:

* **automatic** — derived from the signal record the pipeline writes (data flags, look-ahead check,
  cost model, execution timing, sample size, multiple-testing correction, vault contamination, forced
  datasets). The gate takes the union over **every version** of the record: the registry is
  append-only, so a later edit cannot erase a problem that was once detected.
* **manual** — :func:`raise_finding` records a problem a person or check found (e.g. a corporate-action
  error noticed while reviewing trades). :func:`resolve_finding` closes it only with a reason, a
  resolver **and a newer experiment** — i.e. new evidence, never an approval.

There is deliberately no override. Should one ever be added, it must move the signal into a separate,
explicitly labelled experimental state and must never mark compromised evidence as validated.

Two layers (see RESEARCH_METHODOLOGY.md §1a):

1. **Permanent principles** — the categories below (no survivorship bias, no look-ahead, no leakage,
   no test contamination, valid point-in-time data, realistic execution after the information was
   available, appropriate costs, adequate liquidity/capacity, sufficient statistical evidence,
   appropriate multiple-testing correction). They are fixed in code.
2. **Configurable thresholds** — the numbers that implement some principles (minimum effective N,
   FDR q, allowed execution timings, maximum participation). They come from the hypothesis's
   **pre-registered, versioned** :class:`~quantlab.research.acceptance.AcceptanceCriteria` (frozen
   into its spec hash; looser-than-default values need a registered justification), so they cannot
   be loosened after results are seen. Without a registered hypothesis the conservative defaults
   apply.

Two kinds of finding, both non-promotable:

* ``INTEGRITY_FAILURE`` — the evidence is methodologically compromised. Grade ``BIASED / …``
  (evidence is wrong) or ``PRELIMINARY / …`` (evidence is incomplete or optimistic), e.g.
  ``PRELIMINARY / SURVIVORSHIP-BIASED``.
* ``INSUFFICIENT_EVIDENCE`` — the method may be sound but the evidence is not yet enough (small
  effective sample, not significant after correction). Grade ``INSUFFICIENT_EVIDENCE / …``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import pandas as pd

from quantlab.provenance import DataFlag
from quantlab.research.acceptance import AcceptanceCriteria
from quantlab.research.registry import Registry


class IntegrityIssue(StrEnum):
    SURVIVORSHIP_BIAS = "SURVIVORSHIP_BIAS"
    LOOK_AHEAD_BIAS = "LOOK_AHEAD_BIAS"
    DATA_LEAKAGE = "DATA_LEAKAGE"
    TEST_CONTAMINATION = "TEST_CONTAMINATION"
    INCOMPLETE_UNIVERSE = "INCOMPLETE_UNIVERSE"
    PIT_DATA_ERROR = "PIT_DATA_ERROR"
    UNREALISTIC_EXECUTION = "UNREALISTIC_EXECUTION"
    MISSING_COSTS = "MISSING_COSTS"
    INADEQUATE_SAMPLE = "INADEQUATE_SAMPLE"
    CORPORATE_ACTION_ERROR = "CORPORATE_ACTION_ERROR"
    TIMESTAMP_INTEGRITY = "TIMESTAMP_INTEGRITY"
    UNRELIABLE_PRICES = "UNRELIABLE_PRICES"
    UNCORRECTED_MULTIPLE_TESTING = "UNCORRECTED_MULTIPLE_TESTING"
    OTHER_DEFECT = "OTHER_DEFECT"
    INSUFFICIENT_SIGNIFICANCE = "INSUFFICIENT_SIGNIFICANCE"


I = IntegrityIssue
LABELS: dict[IntegrityIssue, str] = {
    I.SURVIVORSHIP_BIAS: "SURVIVORSHIP-BIASED", I.LOOK_AHEAD_BIAS: "LOOK-AHEAD", I.DATA_LEAKAGE: "DATA-LEAKAGE",
    I.TEST_CONTAMINATION: "TEST-CONTAMINATED", I.INCOMPLETE_UNIVERSE: "INCOMPLETE-UNIVERSE",
    I.PIT_DATA_ERROR: "PIT-DATA-ERROR", I.UNREALISTIC_EXECUTION: "UNREALISTIC-EXECUTION",
    I.MISSING_COSTS: "MISSING-COSTS", I.INADEQUATE_SAMPLE: "INADEQUATE-SAMPLE",
    I.CORPORATE_ACTION_ERROR: "CORPORATE-ACTION-ERROR", I.TIMESTAMP_INTEGRITY: "TIMESTAMP-INTEGRITY",
    I.UNRELIABLE_PRICES: "UNRELIABLE-PRICES", I.UNCORRECTED_MULTIPLE_TESTING: "MULTIPLE-TESTING-UNCORRECTED",
    I.OTHER_DEFECT: "OTHER-DEFECT", I.INSUFFICIENT_SIGNIFICANCE: "NOT-SIGNIFICANT-AFTER-CORRECTION",
}
INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
# methodology may be sound; there is just not enough evidence yet
INSUFFICIENT_EVIDENCE_ISSUES = {I.INADEQUATE_SAMPLE, I.INSUFFICIENT_SIGNIFICANCE}
# integrity failures that mean the evidence is wrong (BIASED) rather than incomplete (PRELIMINARY)
BIASED_ISSUES = {I.LOOK_AHEAD_BIAS, I.DATA_LEAKAGE, I.TEST_CONTAMINATION, I.PIT_DATA_ERROR,
                 I.CORPORATE_ACTION_ERROR, I.TIMESTAMP_INTEGRITY, I.UNRELIABLE_PRICES}
RESEARCH_GRADE = "RESEARCH_GRADE"


def kind_of(category: str) -> str:
    return INSUFFICIENT_EVIDENCE if IntegrityIssue(category) in INSUFFICIENT_EVIDENCE_ISSUES else INTEGRITY_FAILURE

# automatic rules
FLAG_ISSUES: dict[str, IntegrityIssue] = {
    DataFlag.SURVIVORSHIP_BIASED_UNIVERSE.value: I.SURVIVORSHIP_BIAS,
    DataFlag.MODEL_PRICED.value: I.UNRELIABLE_PRICES,
}
VALIDATION_ERROR_ISSUES: dict[str, IntegrityIssue] = {
    "missing_available_at": I.TIMESTAMP_INTEGRITY, "non_trading_session": I.TIMESTAMP_INTEGRITY,
    "non_positive_or_missing_price": I.UNRELIABLE_PRICES, "ohlc_inconsistent": I.UNRELIABLE_PRICES,
    "negative_volume": I.UNRELIABLE_PRICES, "non_positive_value": I.CORPORATE_ACTION_ERROR,
    "unknown_action": I.CORPORATE_ACTION_ERROR, "schema": I.OTHER_DEFECT,
}
# PRINCIPLE: a trade can only execute after the information behind it was available. For each
# execution model we know, whether it satisfies that for daily bars (available after the close).
# Which of these a hypothesis may use is a configurable, pre-registered choice (allowed_executions).
EXECUTION_AFTER_INFORMATION: dict[str, bool] = {"next_open": True, "next_close": True, "same_close": False,
                                                "same_open": False}


@dataclass(frozen=True)
class Finding:
    category: str
    detail: str
    source: str = "automatic"  # "automatic" | "manual"
    severity: str = "MATERIAL"  # only MATERIAL findings block promotion
    finding_id: str | None = None

    @property
    def kind(self) -> str:
        return kind_of(self.category)

    def to_dict(self) -> dict:
        return {**asdict(self), "kind": self.kind}


@dataclass(frozen=True)
class Assessment:
    grade: str
    findings: tuple[Finding, ...]

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "MATERIAL"]

    @property
    def promotable(self) -> bool:
        return not self.blocking

    @property
    def integrity_failures(self) -> list[Finding]:
        return [f for f in self.blocking if f.kind == INTEGRITY_FAILURE]

    @property
    def insufficient_evidence(self) -> list[Finding]:
        return [f for f in self.blocking if f.kind == INSUFFICIENT_EVIDENCE]

    def reasons(self) -> list[str]:
        return [f"{f.kind} {f.category}: {f.detail}" for f in self.blocking]


# ---------------------------------------------------------------------------------------------
def automatic_findings(record: dict, crit: AcceptanceCriteria | None = None) -> list[Finding]:
    """Problems visible in one signal-record version (present-and-bad rules), judged against the
    pre-registered thresholds ``crit`` (conservative defaults when None)."""
    out: list[Finding] = []
    crit = crit or AcceptanceCriteria()
    for flag in sorted(set(record.get("Data_Flags") or [])):
        if flag in FLAG_ISSUES:
            out.append(Finding(FLAG_ISSUES[flag].value, f"data flag {flag}"))
    inputs: dict[str, Any] = record.get("Integrity_Inputs") or {}
    bad = {k: v for k, v in (inputs.get("lookahead_failures") or {}).items() if v}
    if bad:
        out.append(Finding(I.LOOK_AHEAD_BIAS.value, f"truncation test failed for {sorted(bad)}"))
    if inputs.get("test_contaminated"):
        out.append(Finding(I.TEST_CONTAMINATION.value, "untouched test partition was accessed more than once"))
    if inputs.get("leakage"):
        out.append(Finding(I.DATA_LEAKAGE.value, str(inputs["leakage"])))
    for issue in inputs.get("data_quality_issues") or []:
        cat = VALIDATION_ERROR_ISSUES.get(issue.get("check"), I.OTHER_DEFECT)
        out.append(Finding(cat.value, f"{issue.get('table')}: forced save with validation error "
                                      f"{issue.get('check')} ({issue.get('count')} rows)"))
    cm = inputs.get("cost_model")
    if cm:
        frictions = [cm.get(k, 0) or 0 for k in ("half_spread_bps", "slippage_bps", "impact_coef", "commission_per_share",
                                                  "commission_min", "commission_pct")]
        if not any(frictions):
            out.append(Finding(I.MISSING_COSTS.value, f"cost model {cm.get('name')!r} has no frictions"))
        if (cm.get("max_participation") or 0) > crit.max_participation:
            out.append(Finding(I.UNREALISTIC_EXECUTION.value,
                               f"assumed participation {cm['max_participation']:.0%} of ADV > pre-registered limit "
                               f"{crit.max_participation:.0%}"))
    ex = inputs.get("execution")
    if ex is not None:
        if not EXECUTION_AFTER_INFORMATION.get(ex, False):
            out.append(Finding(I.UNREALISTIC_EXECUTION.value,
                               f"execution {ex!r} is not known to occur after the decision information was available"))
        elif ex not in crit.allowed_executions:
            out.append(Finding(I.UNREALISTIC_EXECUTION.value,
                               f"execution {ex!r} not permitted by the pre-registered methodology "
                               f"{list(crit.allowed_executions)}"))
    n = record.get("Effective_Sample_Size")
    if n is not None and pd.notna(n) and n < crit.min_effective_n_inconclusive:
        out.append(Finding(I.INADEQUATE_SAMPLE.value,
                           f"effective N {n} < pre-registered minimum {crit.min_effective_n_inconclusive}"))
    p, q = record.get("P_Value"), record.get("Adjusted_P_Value")
    if p is not None and pd.notna(p) and (q is None or pd.isna(q)):
        out.append(Finding(I.UNCORRECTED_MULTIPLE_TESTING.value, "p-value without a multiple-testing adjustment"))
    elif q is not None and pd.notna(q) and q > crit.fdr_q:
        out.append(Finding(I.INSUFFICIENT_SIGNIFICANCE.value,
                           f"adjusted p (q) {q:.3g} > pre-registered FDR level {crit.fdr_q}"))
    return out


def grade_of(findings: list[Finding]) -> str:
    """``BIASED / …`` or ``PRELIMINARY / …`` for integrity failures, ``INSUFFICIENT_EVIDENCE / …`` for
    missing evidence; both parts when both apply (joined by ``; ``)."""
    present = {IntegrityIssue(f.category) for f in findings if f.severity == "MATERIAL"}
    cats = [c for c in IntegrityIssue if c in present]  # declaration order
    if not cats:
        return RESEARCH_GRADE
    failures = [c for c in cats if c not in INSUFFICIENT_EVIDENCE_ISSUES]
    lacking = [c for c in cats if c in INSUFFICIENT_EVIDENCE_ISSUES]
    parts = []
    if failures:
        cls = "BIASED" if any(c in BIASED_ISSUES for c in failures) else "PRELIMINARY"
        parts.append(f"{cls} / " + "+".join(LABELS[c] for c in failures))
    if lacking:
        parts.append(f"{INSUFFICIENT_EVIDENCE} / " + "+".join(LABELS[c] for c in lacking))
    return "; ".join(parts)


def _dedupe(findings: list[Finding]) -> tuple[Finding, ...]:
    seen, out = set(), []
    for f in findings:
        key = (f.category, f.detail, f.finding_id)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return tuple(out)


def assess_record(record: dict, crit: AcceptanceCriteria | None = None) -> Assessment:
    """Assessment of a single record version (no registry history, no manual findings)."""
    fs = _dedupe(automatic_findings(record, crit))
    return Assessment(grade_of(list(fs)), fs)


def registered_criteria(reg: Registry, versions: list[dict]) -> AcceptanceCriteria:
    """The thresholds that apply: those pre-registered with the signal's hypothesis (immutable, in its
    spec hash). Records without a registered hypothesis get the conservative defaults — never
    thresholds supplied by the record itself."""
    for v in versions:
        hid = v.get("Hypothesis_ID")
        if hid:
            try:
                rec = reg.get("hypotheses", hid)
            except KeyError:
                continue
            return AcceptanceCriteria(**rec["payload"]["spec"].get("acceptance", {}))
    return AcceptanceCriteria()


def assess(reg: Registry, signal_id: str, pending: dict | None = None) -> Assessment:
    """Full assessment: automatic findings over every stored version (plus ``pending``, a version
    about to be written), judged against the pre-registered thresholds, and open manual findings."""
    versions = [r["payload"] for r in reg.find("signal_records", signal_id=signal_id)]
    if pending is not None:
        versions.append(pending)
    crit = registered_criteria(reg, versions)
    fs = [f for v in versions for f in automatic_findings(v, crit)] + open_manual_findings(reg, signal_id)
    fs = _dedupe(fs)
    return Assessment(grade_of(list(fs)), fs)


# ---------------------------------------------------------------------------------------------
def raise_finding(reg: Registry, signal_id: str, category: IntegrityIssue | str, detail: str, raised_by: str,
                  severity: str = "MATERIAL") -> str:
    category = IntegrityIssue(category)
    if severity not in ("MATERIAL", "MINOR"):
        raise ValueError("severity must be MATERIAL or MINOR")
    if not detail.strip() or not raised_by.strip():
        raise ValueError("a finding needs a detail and who raised it")
    return reg.append("integrity_findings", {"signal_id": signal_id, "category": category.value, "action": "RAISED"},
                      {"detail": detail, "severity": severity, "raised_by": raised_by})


def resolve_finding(reg: Registry, finding_id: str, resolution: str, resolved_by: str,
                    evidence_experiment_id: str) -> str:
    """Close a manual finding. Requires an experiment recorded **after** the finding was raised:
    a problem is resolved by new evidence, not by approval."""
    f = reg.get("integrity_findings", finding_id)
    if f["action"] != "RAISED":
        raise ValueError(f"{finding_id} is not a raised finding")
    if any(r["payload"].get("finding_id") == finding_id
           for r in reg.find("integrity_findings", signal_id=f["signal_id"], action="RESOLVED")):
        raise ValueError(f"{finding_id} is already resolved")
    if not resolution.strip() or not resolved_by.strip():
        raise ValueError("a resolution needs a reason and who resolved it")
    ex = reg.get("experiments", evidence_experiment_id)
    if ex["seq"] <= _seq_after(reg, f):
        raise ValueError(f"{evidence_experiment_id} predates finding {finding_id}; resolution needs new evidence")
    return reg.append("integrity_findings", {"signal_id": f["signal_id"], "category": f["category"], "action": "RESOLVED"},
                      {"finding_id": finding_id, "resolution": resolution, "resolved_by": resolved_by,
                       "evidence_experiment_id": evidence_experiment_id})


def _seq_after(reg: Registry, finding_row: dict) -> int:
    """Highest experiments seq that existed when the finding was raised."""
    earlier = [r for r in reg.find("experiments") if r["created_at"] <= finding_row["created_at"]]
    return max((r["seq"] for r in earlier), default=0)


def open_manual_findings(reg: Registry, signal_id: str) -> list[Finding]:
    rows = reg.find("integrity_findings", signal_id=signal_id)
    resolved = {r["payload"]["finding_id"] for r in rows if r["action"] == "RESOLVED"}
    return [Finding(r["category"], r["payload"]["detail"], "manual", r["payload"]["severity"], r["id"])
            for r in rows if r["action"] == "RAISED" and r["id"] not in resolved]
