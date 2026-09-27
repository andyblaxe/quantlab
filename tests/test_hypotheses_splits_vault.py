import pandas as pd
import pytest

from quantlab.research.hypotheses import (
    DuplicateHypothesis, HypothesisSpec, HypothesisStatus, Mechanism, UnregisteredHypothesis,
    advance_hypothesis, family_size, hypothesis_status, load_hypothesis, register_hypothesis, require_registered,
)
from quantlab.research.splits import (
    SplitPlan, active_split_plan, partition_of, register_split_plan, walk_forward_folds,
)
from quantlab.research.vault import TestVault, VaultSealed


def spec(**kw):
    base = dict(name="reversal", question="Do extreme drops revert?", statement="zret<-2 → positive 5d return",
                mechanism=Mechanism.LIQUIDITY, rationale="liquidity provision", family="mean_reversion",
                universe="etf_core", strategy="threshold_event", params={"feature": "zret(h=1)", "threshold": -2.0})
    base.update(kw)
    return HypothesisSpec(**base)


def test_registration_is_required_and_exact(registry):
    s = spec()
    hid = register_hypothesis(registry, s, reason="test")
    assert hypothesis_status(registry, hid) == HypothesisStatus.REGISTERED
    require_registered(registry, hid, s)
    tweaked = spec(params={"feature": "zret(h=1)", "threshold": -2.5})
    with pytest.raises(UnregisteredHypothesis):
        require_registered(registry, hid, tweaked)  # variant must be registered separately
    with pytest.raises(UnregisteredHypothesis):
        require_registered(registry, "H-999999", s)
    assert load_hypothesis(registry, hid) == s


def test_duplicate_registration_refused_and_variants_counted(registry):
    register_hypothesis(registry, spec())
    with pytest.raises(DuplicateHypothesis):
        register_hypothesis(registry, spec())
    register_hypothesis(registry, spec(params={"feature": "zret(h=1)", "threshold": -3.0}))
    register_hypothesis(registry, spec(family="trend", name="x"))
    assert family_size(registry, "mean_reversion") == 2
    assert family_size(registry) == 3


def test_acceptance_criteria_frozen_into_spec_hash():
    s1 = spec()
    s2 = spec(acceptance={**s1.acceptance, "fdr_q": 0.2})
    assert s1.spec_hash() != s2.spec_hash()


def test_lifecycle_only_moves_forward(registry):
    hid = register_hypothesis(registry, spec())
    advance_hypothesis(registry, hid, HypothesisStatus.EVALUATED, "done")
    with pytest.raises(ValueError):
        advance_hypothesis(registry, hid, HypothesisStatus.REGISTERED, "rewind")
    advance_hypothesis(registry, hid, HypothesisStatus.FROZEN, "locked")


def test_split_boundaries_have_embargo():
    sessions = pd.bdate_range("2010-01-01", "2020-12-31")
    plan = SplitPlan(train_end="2015-12-31", validation_end="2018-12-31", embargo_sessions=10)
    b = plan.boundaries(sessions)
    assert b["train"][1] == pd.Timestamp("2015-12-31")
    gap = sessions[(sessions > b["train"][1]) & (sessions < b["validation"][0])]
    assert len(gap) == 10
    gap2 = sessions[(sessions > b["validation"][1]) & (sessions < b["test"][0])]
    assert len(gap2) == 10 and b["test"][1] == sessions[-1]


def test_trades_straddling_boundaries_are_purged():
    sessions = pd.bdate_range("2010-01-01", "2020-12-31")
    b = SplitPlan("2015-12-31", "2018-12-31", embargo_sessions=5).boundaries(sessions)
    entry = pd.Series(pd.to_datetime(["2015-06-01", "2015-12-28", "2017-01-03", "2019-06-03"]))
    exit_ = pd.Series(pd.to_datetime(["2015-06-08", "2016-01-05", "2017-01-10", "2019-06-10"]))
    assert partition_of(entry, exit_, b).tolist() == ["train", "embargo", "validation", "test"]


def test_split_plan_changes_are_recorded(registry):
    register_split_plan(registry, SplitPlan("2015-12-31", "2018-12-31"), "initial")
    register_hypothesis(registry, spec())
    register_split_plan(registry, SplitPlan("2014-12-31", "2018-12-31"), "moved")
    assert active_split_plan(registry).train_end == "2014-12-31"
    last = registry.find("journal", action="register_split_plan")[-1]
    assert last["payload"]["changed_after_research_started"] is True


def test_walk_forward_folds_never_overlap_and_respect_purge():
    sessions = pd.bdate_range("2010-01-01", "2018-12-31")
    folds = walk_forward_folds(sessions, min_train_sessions=500, test_sessions=250, purge_sessions=10,
                               end=pd.Timestamp("2017-12-31"))
    assert len(folds) >= 4
    for f in folds:
        assert f.train_end < f.test_start and f.test_end <= pd.Timestamp("2017-12-31")
        assert len(sessions[(sessions > f.train_end) & (sessions < f.test_start)]) == 10
    for a, b in zip(folds, folds[1:]):
        assert a.test_end < b.test_start


def test_vault_requires_frozen_and_logs_contamination(registry, small_panel):
    plan = SplitPlan("2018-12-31", "2020-06-30", embargo_sessions=5)
    vault = TestVault(registry, plan)
    dev = vault.development_view(small_panel)
    assert all(v.index.max() <= pd.Timestamp("2020-06-30") for v in dev.values())
    hid = register_hypothesis(registry, spec())
    with pytest.raises(VaultSealed):
        vault.unseal(small_panel, hid, None, "peek")  # not frozen
    advance_hypothesis(registry, hid, HypothesisStatus.EVALUATED, "e")
    advance_hypothesis(registry, hid, HypothesisStatus.FROZEN, "f")
    full = vault.unseal(small_panel, hid, "E-000001", "final test")
    assert full["close"].index.max() > pd.Timestamp("2020-06-30")
    advance_hypothesis(registry, hid, HypothesisStatus.TESTED, "t")
    with pytest.raises(VaultSealed):
        vault.unseal(small_panel, hid, "E-000002", "again")
    vault.unseal(small_panel, hid, "E-000002", "again", acknowledge_contamination=True)
    assert vault.is_contaminated(hid) and vault.total_accesses() == 2
