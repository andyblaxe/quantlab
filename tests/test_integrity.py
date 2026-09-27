"""Research-integrity promotion gate: every material integrity failure blocks promotion, approval
cannot override it, the refusal is recorded, and history cannot be rewritten to get around it."""

import pandas as pd
import pytest

from quantlab.paper.broker import PaperBroker, Quote
from quantlab.paper.forward import ForwardTester, NotPaperTradable, live_eligibility
from quantlab.research.catalog import InvalidTransition, SignalStatus, latest_record, transition, upsert_record
from quantlab.research.acceptance import AcceptanceCriteria, loosened_vs_defaults
from quantlab.research.integrity import (
    IntegrityIssue, assess, assess_record, raise_finding, resolve_finding,
)

CLEAN_COSTS = {"name": "retail_etf", "half_spread_bps": 1.0, "slippage_bps": 2.0, "impact_coef": 0.1,
               "commission_per_share": 0.0, "commission_min": 0.0, "commission_pct": 0.0, "max_participation": 0.01}
CLEAN = {"Name": "x", "Data_Flags": ["SURVIVORSHIP_RISK"], "Effective_Sample_Size": 250, "P_Value": 0.001,
         "Adjusted_P_Value": 0.01, "Integrity_Inputs": {"lookahead_failures": {"zret(h=1,vol_n=63)": []},
                                                         "cost_model": CLEAN_COSTS, "execution": "next_open",
                                                         "data_quality_issues": []}}

# one failure per integrity category the pipeline can detect automatically
FAILURES = {
    IntegrityIssue.SURVIVORSHIP_BIAS: {"Data_Flags": ["SURVIVORSHIP_BIASED_UNIVERSE"]},
    IntegrityIssue.UNRELIABLE_PRICES: {"Data_Flags": ["MODEL_PRICED"]},
    IntegrityIssue.LOOK_AHEAD_BIAS: {"Integrity_Inputs": {**CLEAN["Integrity_Inputs"],
                                                          "lookahead_failures": {"f": ["2010-01-04"]}}},
    IntegrityIssue.TEST_CONTAMINATION: {"Integrity_Inputs": {**CLEAN["Integrity_Inputs"], "test_contaminated": True}},
    IntegrityIssue.DATA_LEAKAGE: {"Integrity_Inputs": {**CLEAN["Integrity_Inputs"], "leakage": "label in features"}},
    IntegrityIssue.MISSING_COSTS: {"Integrity_Inputs": {**CLEAN["Integrity_Inputs"], "cost_model": {
        **CLEAN_COSTS, "half_spread_bps": 0.0, "slippage_bps": 0.0, "impact_coef": 0.0}}},
    IntegrityIssue.UNREALISTIC_EXECUTION: {"Integrity_Inputs": {**CLEAN["Integrity_Inputs"], "execution": "same_close"}},
    IntegrityIssue.INADEQUATE_SAMPLE: {"Effective_Sample_Size": 12},
    IntegrityIssue.UNCORRECTED_MULTIPLE_TESTING: {"Adjusted_P_Value": None},
    IntegrityIssue.TIMESTAMP_INTEGRITY: {"Integrity_Inputs": {**CLEAN["Integrity_Inputs"], "data_quality_issues": [
        {"table": "bars_daily", "check": "missing_available_at", "count": 3}]}},
    IntegrityIssue.CORPORATE_ACTION_ERROR: {"Integrity_Inputs": {**CLEAN["Integrity_Inputs"], "data_quality_issues": [
        {"table": "corporate_actions", "check": "unknown_action", "count": 1}]}},
}
# categories only a person (or a future check) can detect: recorded as manual findings
MANUAL_ONLY = [IntegrityIssue.INCOMPLETE_UNIVERSE, IntegrityIssue.PIT_DATA_ERROR, IntegrityIssue.OTHER_DEFECT]


def _validating(reg, sid, fields):
    upsert_record(reg, sid, fields, "evaluated")
    transition(reg, sid, SignalStatus.VALIDATING, "screened")


def test_clean_evidence_is_research_grade_and_promotable(registry):
    assert assess_record(CLEAN).grade == "RESEARCH_GRADE"
    _validating(registry, "SIG-1", CLEAN)
    transition(registry, "SIG-1", SignalStatus.ACCEPTED, "passed")


@pytest.mark.parametrize("issue", list(FAILURES), ids=lambda i: i.value)
def test_each_integrity_failure_blocks_promotion_even_with_approval(registry, issue):
    _validating(registry, "SIG-1", {**CLEAN, **FAILURES[issue]})
    rec = latest_record(registry, "SIG-1")
    assert rec["Evidence_Grade"] != "RESEARCH_GRADE"
    assert issue.value in {f["category"] for f in rec["Integrity_Findings"]}
    for to in (SignalStatus.ACCEPTED,):
        with pytest.raises(InvalidTransition, match="research-integrity gate"):
            transition(registry, "SIG-1", to, "promote", approved_by="owner")
    # the refusal is recorded with its reasons
    blocked = registry.find("journal", action="promotion_blocked")
    assert blocked and issue.value in {f["category"] for f in blocked[-1]["payload"]["blocking_findings"]}
    assert latest_record(registry, "SIG-1")["Status"] == "VALIDATING"  # stored, analysable, not promoted


@pytest.mark.parametrize("issue", MANUAL_ONLY, ids=lambda i: i.value)
def test_manual_findings_block_until_resolved_with_new_evidence(registry, issue):
    _validating(registry, "SIG-1", CLEAN)
    fid = raise_finding(registry, "SIG-1", issue, "found during review", raised_by="owner")
    with pytest.raises(InvalidTransition, match=issue.value):
        transition(registry, "SIG-1", SignalStatus.ACCEPTED, "promote", approved_by="owner")
    old = registry.append("experiments", {"hypothesis_id": "H-1", "kind": "development", "status": "PASS"}, {})
    # an experiment that existed before the finding cannot resolve it (that would be an approval)
    registry.append("integrity_findings", {"signal_id": "SIG-0", "category": "OTHER_DEFECT", "action": "RAISED"},
                    {"detail": "unrelated", "severity": "MINOR", "raised_by": "x"})
    fid2 = raise_finding(registry, "SIG-1", issue, "second look", raised_by="owner")
    with pytest.raises(ValueError, match="new evidence"):
        resolve_finding(registry, fid2, "looks fine", "owner", old)
    new = registry.append("experiments", {"hypothesis_id": "H-1", "kind": "development", "status": "PASS"}, {})
    resolve_finding(registry, fid, "re-run on corrected data", "owner", new)
    resolve_finding(registry, fid2, "re-run on corrected data", "owner", new)
    with pytest.raises(ValueError, match="already resolved"):
        resolve_finding(registry, fid, "again", "owner", new)
    transition(registry, "SIG-1", SignalStatus.ACCEPTED, "promote")


def test_minor_findings_are_recorded_but_do_not_block(registry):
    _validating(registry, "SIG-1", CLEAN)
    raise_finding(registry, "SIG-1", IntegrityIssue.OTHER_DEFECT, "cosmetic label typo", "owner", severity="MINOR")
    assert len(assess(registry, "SIG-1").findings) == 1
    transition(registry, "SIG-1", SignalStatus.ACCEPTED, "promote")


def test_history_cannot_be_rewritten_to_pass_the_gate(registry):
    _validating(registry, "SIG-1", {**CLEAN, **FAILURES[IntegrityIssue.SURVIVORSHIP_BIAS]})
    upsert_record(registry, "SIG-1", {"Data_Flags": ["SURVIVORSHIP_RISK"]}, "remove the flag")
    assert latest_record(registry, "SIG-1")["Evidence_Grade"] == "PRELIMINARY / SURVIVORSHIP-BIASED"
    with pytest.raises(InvalidTransition):
        transition(registry, "SIG-1", SignalStatus.ACCEPTED, "promote", approved_by="owner")
    for field in ("Evidence_Grade", "Integrity_Findings"):
        with pytest.raises(ValueError, match="derived"):
            upsert_record(registry, "SIG-1", {field: "RESEARCH_GRADE"}, "sneaky")


def test_grades_separate_integrity_failure_from_insufficient_evidence():
    small = assess_record({**CLEAN, **FAILURES[IntegrityIssue.INADEQUATE_SAMPLE]})
    assert small.grade == "INSUFFICIENT_EVIDENCE / INADEQUATE-SAMPLE"  # not "biased"
    assert small.insufficient_evidence and not small.integrity_failures and not small.promotable
    weak = assess_record({**CLEAN, "Adjusted_P_Value": 0.3})
    assert weak.grade == "INSUFFICIENT_EVIDENCE / NOT-SIGNIFICANT-AFTER-CORRECTION"
    both = {**CLEAN, "Data_Flags": ["SURVIVORSHIP_BIASED_UNIVERSE"], "Effective_Sample_Size": 12}
    assert assess_record(both).grade == "PRELIMINARY / SURVIVORSHIP-BIASED; INSUFFICIENT_EVIDENCE / INADEQUATE-SAMPLE"
    worst = {**both, **FAILURES[IntegrityIssue.LOOK_AHEAD_BIAS]}
    assert assess_record(worst).grade.startswith("BIASED / ")
    assert {f.kind for f in assess_record(worst).findings} == {"INTEGRITY_FAILURE", "INSUFFICIENT_EVIDENCE"}


def _registered_signal(reg, acceptance: dict, record: dict) -> str:
    from quantlab.research.hypotheses import HypothesisSpec, Mechanism, register_hypothesis
    spec = HypothesisSpec(name="t", question="q", statement="s", mechanism=Mechanism.LIQUIDITY, rationale="r",
                          family="f", universe="u", strategy="threshold_event",
                          params={"conditions": [["zret(h=1,vol_n=63)", "<", -2.0]]},
                          acceptance={**AcceptanceCriteria().to_dict(), **acceptance})
    hid = register_hypothesis(reg, spec)
    _validating(reg, "SIG-1", {**record, "Hypothesis_ID": hid})
    return hid


def test_thresholds_come_from_the_preregistered_methodology(registry):
    # a methodology registered (with justification) for 20 independent observations passes at N=25 ...
    _registered_signal(registry, {"min_effective_n_inconclusive": 20, "justification": "annual rebalancing study"},
                       {**CLEAN, "Effective_Sample_Size": 25})
    assert assess(registry, "SIG-1").promotable
    transition(registry, "SIG-1", SignalStatus.ACCEPTED, "passed")


def test_stricter_preregistered_thresholds_are_enforced(registry):
    _registered_signal(registry, {"max_participation": 0.005, "allowed_executions": ["next_open"]},
                       {**CLEAN, "Integrity_Inputs": {**CLEAN["Integrity_Inputs"], "execution": "next_close"}})
    reasons = " ".join(assess(registry, "SIG-1").reasons())
    assert "participation" in reasons and "not permitted by the pre-registered methodology" in reasons


def test_thresholds_cannot_be_loosened_without_registered_justification_or_after_the_fact(registry):
    from quantlab.research.hypotheses import HypothesisSpec, Mechanism, register_hypothesis
    loose = {**AcceptanceCriteria().to_dict(), "fdr_q": 0.5}
    spec = HypothesisSpec(name="t", question="q", statement="s", mechanism=Mechanism.LIQUIDITY, rationale="r",
                          family="f", universe="u", strategy="threshold_event", params={}, acceptance=loose)
    with pytest.raises(ValueError, match="justification"):
        register_hypothesis(registry, spec)
    assert loosened_vs_defaults(AcceptanceCriteria(fdr_q=0.5, allowed_executions=("next_open", "same_close"))) == [
        "fdr_q", "allowed_executions"]
    # after results: thresholds a record carries are ignored; the registered (default) ones apply
    _registered_signal(registry, {}, {**CLEAN, "Effective_Sample_Size": 25})
    upsert_record(registry, "SIG-1", {"Integrity_Inputs": {**CLEAN["Integrity_Inputs"],
                                                            "validation_config": {"min_effective_n_inconclusive": 5}}},
                  "try to loosen")
    assert not assess(registry, "SIG-1").promotable


def test_execution_before_information_is_an_integrity_failure_whatever_the_config():
    crit = AcceptanceCriteria(allowed_executions=("same_close",), justification="x")
    a = assess_record({**CLEAN, "Integrity_Inputs": {**CLEAN["Integrity_Inputs"], "execution": "same_close"}}, crit)
    assert a.integrity_failures and "after the decision information" in a.reasons()[0]


def test_finding_after_promotion_stops_paper_trading_and_live_eligibility(registry):
    _validating(registry, "SIG-1", CLEAN)
    transition(registry, "SIG-1", SignalStatus.ACCEPTED, "passed")
    transition(registry, "SIG-1", SignalStatus.PAPER_TRADING, "approved", approved_by="owner")
    raise_finding(registry, "SIG-1", IntegrityIssue.CORPORATE_ACTION_ERROR, "split applied twice", "owner")
    b = PaperBroker(starting_cash=10_000)
    b.set_quote(Quote("SPY", 99.98, 100.02, pd.Timestamp("2026-01-05 15:00", tz="UTC"), source="test"))
    from quantlab.risk.gate import RiskGate
    with pytest.raises(NotPaperTradable, match="integrity"):
        ForwardTester(registry, b, RiskGate()).open("SIG-1", "SPY", 1, "fired", {}, {})
    le = live_eligibility(registry, "SIG-1", pd.Series([0.01] * 50), modelled_slippage=0.0003)
    assert not le["research_integrity"]["passed"] and not le["eligible"]
    with pytest.raises(InvalidTransition):
        transition(registry, "SIG-1", SignalStatus.LIVE_ELIGIBLE, "promote", approved_by="owner")
