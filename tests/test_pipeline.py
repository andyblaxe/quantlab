"""End-to-end pipeline tests on SIMULATED markets with known ground truth."""

import pytest

from quantlab.data.providers.synthetic import SyntheticConfig, SyntheticMarket
from quantlab.research.artifacts import ArtifactStore
from quantlab.research.catalog import latest_record
from quantlab.research.data import ResearchData
from quantlab.research.hypotheses import (
    HypothesisSpec, HypothesisStatus, Mechanism, hypothesis_status, register_hypothesis,
)
from quantlab.research.pipeline import PipelineError, ResearchPipeline
from quantlab.research.registry import Registry
from quantlab.research.splits import SplitPlan, register_split_plan

PLAN = SplitPlan("2013-12-31", "2017-12-31", embargo_sessions=10)


def _pipeline(tmp_path, ar1, seed=3):
    m = SyntheticMarket(SyntheticConfig(n_symbols=8, start="2006-01-01", end="2021-12-31", seed=seed, idio_ar1=ar1))
    reg = Registry(tmp_path / "r.sqlite")
    register_split_plan(reg, PLAN, "test")
    pipe = ResearchPipeline(reg, ResearchData.from_synthetic(m), PLAN, ArtifactStore(tmp_path / "art"),
                            n_boot=200, n_perm=200, mc_paths=200, wf_min_train=504, wf_test=252)
    return reg, pipe


def _spec(threshold=-2.0, **kw):
    base = dict(name="1d reversal", question="Do -2 sigma days revert?", statement="zret<-2 → positive next-day return",
                mechanism=Mechanism.LIQUIDITY, rationale="liquidity provision", family="mean_reversion",
                universe="synthetic_all", strategy="threshold_event",
                params={"conditions": [["zret(h=1,vol_n=63)", "<", threshold]]},
                param_neighbors=[{"conditions": [["zret(h=1,vol_n=63)", "<", threshold + 0.5]]},
                                 {"conditions": [["zret(h=1,vol_n=63)", "<", threshold - 0.5]]}],
                holding_period=1, cost_profile="retail_etf")
    base.update(kw)
    return HypothesisSpec(**base)


def test_planted_edge_is_detected_and_survives_untouched_test(tmp_path):
    reg, pipe = _pipeline(tmp_path, ar1=-0.45)
    hid = register_hypothesis(reg, _spec())
    out = pipe.evaluate(hid)
    assert out["signal_status"] == "VALIDATING", out["conclusion"]
    assert hypothesis_status(reg, hid) == HypothesisStatus.FROZEN
    rec = latest_record(reg, out["signal_id"])
    assert rec["Data_Label"] == "SIMULATED" and rec["Sample_Size"] > 100
    assert rec["Expected_Return"] > 0 and rec["P_Value"] < 0.05
    test = pipe.run_untouched_test(hid)
    assert test["signal_status"] == "ACCEPTED"
    assert pipe.vault.total_accesses() == 1
    assert all(not v for v in reg.verify_chain().values())
    with pytest.raises(PipelineError):
        pipe.run_untouched_test(hid)  # the test partition is one-shot


def test_null_market_is_not_accepted_and_vault_untouched(tmp_path):
    reg, pipe = _pipeline(tmp_path, ar1=0.0, seed=5)
    hid = register_hypothesis(reg, _spec())
    out = pipe.evaluate(hid)
    assert out["signal_status"] in ("REJECTED", "EXPERIMENTAL")
    assert pipe.vault.total_accesses() == 0  # rejected ideas never consume the holdout
    with pytest.raises(PipelineError):
        pipe.run_untouched_test(hid)
    # the failure is permanently on record
    ex = reg.find("experiments", hypothesis_id=hid, kind="development")
    assert len(ex) == 1 and ex[0]["status"] in ("FAIL", "INCONCLUSIVE")
    assert reg.find("journal", action="evaluate_hypothesis")[0]["payload"]["conclusion"]


def test_evaluation_runs_once_and_requires_registration(tmp_path):
    reg, pipe = _pipeline(tmp_path, ar1=0.0)
    hid = register_hypothesis(reg, _spec())
    pipe.evaluate(hid)
    with pytest.raises(PipelineError):
        pipe.evaluate(hid)
    with pytest.raises(KeyError):
        pipe.evaluate("H-999999")


def test_family_burden_raises_adjusted_p(tmp_path):
    reg, pipe = _pipeline(tmp_path, ar1=-0.45)
    # nine other hypotheses registered in the same family (not yet tested → count as p=1)
    for k in range(9):
        register_hypothesis(reg, _spec(threshold=-3.0 - k * 0.1, name=f"variant {k}"))
    hid = register_hypothesis(reg, _spec())
    out = pipe.evaluate(hid)
    mt = out["results"]["multiple_testing"]
    assert mt["family_size"] == 10
    assert mt["q_value"] > out["results"]["train"]["p_primary"]


def test_experiment_is_reproducible(tmp_path):
    r1, p1 = _pipeline(tmp_path / "a", ar1=-0.45)
    r2, p2 = _pipeline(tmp_path / "b", ar1=-0.45)
    h1, h2 = register_hypothesis(r1, _spec()), register_hypothesis(r2, _spec())
    a, b = p1.evaluate(h1)["results"], p2.evaluate(h2)["results"]
    assert a["data"]["dataset_hashes"] == b["data"]["dataset_hashes"]
    for part in ("train", "validation"):
        assert a[part]["trades_net"] == b[part]["trades_net"]
        assert a[part]["p_primary"] == b[part]["p_primary"]
    assert a["monte_carlo"]["historical"]["1000"]["p_ruin"] == b["monte_carlo"]["historical"]["1000"]["p_ruin"]


def test_dsr_not_inflated_by_heterogeneous_bad_trials(tmp_path):
    """Regression (2026-09-27): cost-dominated, very negative-Sharpe trials must not raise the DSR bar
    for a genuine effect. The criterion uses the null sampling dispersion of the Sharpe ratio."""
    reg, pipe = _pipeline(tmp_path, ar1=-0.45)
    losers = []
    for seg in range(5):
        s = _spec(name=f"overnight loser {seg}", strategy="segment_hold", family="calendar",
                  params={"segment": "overnight", "conditions": [["zret(h=1,vol_n=63)", ">", -5.0 + seg]]},
                  param_neighbors=[], cost_profile="per_share_broker")
        losers.append(register_hypothesis(reg, s))
    for h in losers:
        pipe.evaluate(h)
    hid = register_hypothesis(reg, _spec())
    out = pipe.evaluate(hid)
    dsr = out["results"]["deflated_sharpe"]
    assert dsr["dsr"] >= 0.95, dsr
    assert "diagnostic_empirical_dispersion" in dsr
    assert out["signal_status"] == "VALIDATING", out["conclusion"]
