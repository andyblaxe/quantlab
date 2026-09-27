"""Calibrate the research machinery itself on SIMULATED markets with known truth.

* **Size / false-positive rate** — markets with *no* planted effect: how often does a hypothesis get
  to FROZEN (passes all development criteria) or ACCEPTED (also passes the untouched test)?
  Should be well below 5%.
* **Power** — markets with a planted short-term reversal large enough to survive costs: how often
  is it detected?

The result is recorded as a project event so reports can cite the machinery's measured error rates.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from quantlab.data.providers.synthetic import SyntheticConfig, SyntheticMarket
from quantlab.research.artifacts import ArtifactStore
from quantlab.research.data import ResearchData
from quantlab.research.hypotheses import HypothesisSpec, Mechanism, register_hypothesis
from quantlab.research.pipeline import ResearchPipeline
from quantlab.research.registry import Registry
from quantlab.research.splits import SplitPlan, register_split_plan

PLAN = SplitPlan("2013-12-31", "2017-12-31", embargo_sessions=10)


def _spec() -> HypothesisSpec:
    z = "zret(h=1,vol_n=63)"
    return HypothesisSpec(
        name="calibration reversal", question="calibration", statement="z<-2 → positive next-day net return",
        mechanism=Mechanism.LIQUIDITY, rationale="calibration", family="calibration", universe="synthetic_all",
        strategy="threshold_event", params={"conditions": [[z, "<", -2.0]]},
        param_neighbors=[{"conditions": [[z, "<", -1.5]]}, {"conditions": [[z, "<", -2.5]]}],
        holding_period=1, cost_profile="retail_etf")


def _one(seed: int, ar1: float, root: Path) -> dict:
    m = SyntheticMarket(SyntheticConfig(n_symbols=8, start="2006-01-01", end="2021-12-31", seed=seed, idio_ar1=ar1))
    reg = Registry(root / f"r{seed}_{ar1}.sqlite")
    register_split_plan(reg, PLAN, "calibration")
    pipe = ResearchPipeline(reg, ResearchData.from_synthetic(m), PLAN, ArtifactStore(root / f"a{seed}_{ar1}"),
                            n_boot=200, n_perm=200, mc_paths=100, wf_min_train=504, wf_test=252)
    hid = register_hypothesis(reg, _spec())
    out = pipe.evaluate(hid)
    frozen = "run_untouched_test" in out["next_step"]
    final = pipe.run_untouched_test(hid)["signal_status"] if frozen else out["signal_status"]
    reg.close()
    return {"seed": seed, "ar1": ar1, "dev_status": out["signal_status"], "frozen": frozen, "final": final}


def run_calibration(ws, n_null: int = 10, n_power: int = 3, power_ar1: float = -0.45) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        null = [_one(1000 + k, 0.0, root) for k in range(n_null)]
        power = [_one(2000 + k, power_ar1, root) for k in range(n_power)]
    res = {
        "label": "SIMULATED",
        "null_runs": n_null, "null_frozen_rate": sum(r["frozen"] for r in null) / max(n_null, 1),
        "null_accepted_rate": sum(r["final"] == "ACCEPTED" for r in null) / max(n_null, 1),
        "power_runs": n_power, "power_ar1": power_ar1,
        "power_frozen_rate": sum(r["frozen"] for r in power) / max(n_power, 1),
        "power_accepted_rate": sum(r["final"] == "ACCEPTED" for r in power) / max(n_power, 1),
        "runs": null + power,
    }
    ws.registry.append("project_events", {"kind": "calibration"}, res)
    ws.registry.journal("calibration", reason="measure pipeline size and power on simulated markets",
                        results={k: v for k, v in res.items() if k != "runs"})
    return res
