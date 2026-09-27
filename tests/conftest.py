import pytest

from quantlab.data.adjust import build_panel
from quantlab.data.providers.synthetic import SyntheticConfig, SyntheticMarket
from quantlab.research.registry import Registry


@pytest.fixture(scope="session")
def small_market():
    """SIMULATED market, 6 symbols + market, 2016-2021 (session-scoped: generated once)."""
    return SyntheticMarket(SyntheticConfig(n_symbols=6, start="2016-01-01", end="2021-12-31", seed=11,
                                           split_threshold=150.0))


@pytest.fixture(scope="session")
def small_panel(small_market):
    return build_panel(small_market.bars().frame, small_market.corporate_actions().frame)


@pytest.fixture()
def registry(tmp_path):
    reg = Registry(tmp_path / "registry.sqlite")
    yield reg
    reg.close()


# --- shared research "world" for analyst and dashboard tests (SIMULATED) ---------------------------
from quantlab.analyst.facts import FactBase  # noqa: E402
from quantlab.research.artifacts import ArtifactStore  # noqa: E402
from quantlab.research.data import ResearchData  # noqa: E402
from quantlab.research.hypotheses import HypothesisSpec, Mechanism, register_hypothesis  # noqa: E402
from quantlab.research.pipeline import ResearchPipeline  # noqa: E402
from quantlab.research.splits import SplitPlan, register_split_plan  # noqa: E402


def _spec(name, thr, family="mean_reversion", mech=Mechanism.LIQUIDITY):
    z = "zret(h=1,vol_n=63)"
    return HypothesisSpec(name=name, question=f"Does z<{thr} revert?", statement=f"z<{thr} → positive next-day return",
                          mechanism=mech, rationale="liquidity provision", family=family, universe="synthetic_all",
                          strategy="threshold_event", params={"conditions": [[z, "<", thr]]},
                          param_neighbors=[{"conditions": [[z, "<", thr + 0.5]]}, {"conditions": [[z, "<", thr - 0.5]]}],
                          holding_period=1, cost_profile="retail_etf")


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analyst")
    m = SyntheticMarket(SyntheticConfig(n_symbols=8, start="2006-01-01", end="2021-12-31", seed=3, idio_ar1=-0.45))
    reg = Registry(tmp / "r.sqlite")
    plan = SplitPlan("2013-12-31", "2017-12-31", embargo_sessions=10)
    register_split_plan(reg, plan, "t")
    art = ArtifactStore(tmp / "art")
    pipe = ResearchPipeline(reg, ResearchData.from_synthetic(m), plan, art, n_boot=200, n_perm=200, mc_paths=200,
                            wf_min_train=504, wf_test=252)
    good = register_hypothesis(reg, _spec("reversal -2", -2.0))
    bad = register_hypothesis(reg, _spec("buy strength (bollinger)", 3.0, mech=Mechanism.UNKNOWN))  # nonsense: z<3 ≈ always
    queued = register_hypothesis(reg, _spec("queued idea", -3.0))
    pipe.evaluate(good)
    pipe.run_untouched_test(good)
    pipe.evaluate(bad)
    return {"reg": reg, "fb": FactBase(reg, art), "good": good, "bad": bad, "queued": queued, "tmp": tmp}


