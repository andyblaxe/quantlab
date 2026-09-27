import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.costs import get_cost_model
from quantlab.features import get_feature
from quantlab.research.generator import TEMPLATES, BudgetExceeded, generate, register_batch
from quantlab.research.hypotheses import family_size
from quantlab.research.measurements import conditional_forward, vrp
from quantlab.research.program import UNIVERSES_V1, program_v1
from quantlab.research.strategies import run_strategy, template_kind
from quantlab.research.data import ResearchData


def test_program_v1_is_well_formed():
    specs = program_v1()
    assert len(specs) == 10
    assert len({s.spec_hash() for s in specs}) == 10
    for s in specs:
        assert s.universe in UNIVERSES_V1
        template_kind(s.strategy)  # raises if unknown
        for cond in s.params.get("conditions", []):
            get_feature(cond[0])  # every feature spec resolves
        if template_kind(s.strategy) in ("event", "position"):
            assert s.param_neighbors, f"{s.name} must pre-declare neighbours for sensitivity"
        assert s.rationale and s.statement


def test_program_hashes_are_stable():
    """Specs are pre-registered in git; any edit must be deliberate (changes the hash)."""
    a = [s.spec_hash() for s in program_v1()]
    b = [s.spec_hash() for s in program_v1()]
    assert a == b


def test_generator_budget_and_registration(registry):
    specs = generate("extreme_move_reversal", "synthetic_all", budget=24)
    assert len(specs) == TEMPLATES["extreme_move_reversal"].size() == 12
    assert all(s.exploratory and s.generated_by.startswith("generator:") for s in specs)
    assert all(s.param_neighbors for s in specs)
    with pytest.raises(BudgetExceeded):
        generate("extreme_move_reversal", "synthetic_all", budget=5)
    ids = register_batch(registry, specs, "test batch")
    assert len(ids) == 12 and family_size(registry, "mean_reversion") == 12
    again = register_batch(registry, specs, "repeat")
    assert again == []  # duplicates skipped, not re-counted
    batches = registry.find("journal", action="generate_batch")
    assert batches[-1]["payload"]["duplicates_skipped"] and batches[-1]["payload"]["total_hypotheses_registered"] == 12


def test_segment_strategy_charges_a_round_trip_per_day(small_panel):
    run = run_strategy(small_panel, "segment_hold", {"segment": "overnight"}, ["SYNMKT"], get_cost_model("retail_etf"))
    tr = run.trades
    assert len(tr) > 1000
    assert (tr["net_ret"] < tr["gross_ret"]).all()
    # overnight gross = open[t+1] / close[t] - 1 on the adjusted series
    r0 = tr.iloc[0]
    c, o = small_panel["close"]["SYNMKT"], small_panel["open"]["SYNMKT"]
    assert np.isclose(r0["gross_ret"], o.loc[r0["exit_session"]] / c.loc[r0["entry_session"]] - 1)
    assert (tr["entry_session"] > tr["signal_session"]).all()


def test_measurements(small_market):
    data = ResearchData.from_synthetic(small_market)
    end = pd.Timestamp("2020-12-31")
    out = vrp(data, end, {"horizon": 21, "implied": "SYN_VIX"})
    assert out["status"] == "OK" and out["estimate"] > 0  # synthetic VIX embeds a premium by construction
    missing = vrp(data, end, {"implied": "VIX"})
    assert missing["status"] == "DATA_UNAVAILABLE"
    cf = conditional_forward(data, end, {"numerator": "SYN_VIX", "threshold": 25.0, "horizon": 21})
    assert cf["status"] == "OK" and cf["effective_n"] <= cf["n"]
    assert cf["details"]["fwd_vol_when_true"] > cf["details"]["fwd_vol_when_false"]  # GARCH persistence
