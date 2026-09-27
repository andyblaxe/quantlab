"""Constrained, budgeted, pre-registered hypothesis generation.

What this is NOT: a search over millions of feature combinations that reports the winners.

What it is:
* a small set of **templates**, each tied to a declared economic mechanism and a
  multiple-testing family;
* each template expands a **small, declared grid** (features × thresholds × horizons);
* a hard **budget** per batch (default 24) and a running global count;
* every generated spec is **registered before any evaluation** (``register_batch``), tagged
  ``exploratory=True`` and ``generated_by="generator:<template>"``;
* parameter neighbours are derived from grid adjacency, so every generated hypothesis carries its
  own sensitivity test;
* the batch itself is journaled (template, grid, budget, IDs) so reports can show how many
  hypotheses the machine produced, including all that failed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from quantlab.research.hypotheses import DuplicateHypothesis, HypothesisSpec, Mechanism, register_hypothesis
from quantlab.research.registry import Registry


class BudgetExceeded(ValueError):
    pass


@dataclass(frozen=True)
class Template:
    name: str
    mechanism: Mechanism
    family: str
    rationale: str
    strategy: str
    feature_grid: tuple[str, ...]
    thresholds: tuple[float, ...]
    op: str
    holding: tuple[int, ...]
    extra_conditions: tuple[tuple[str, str, float], ...] = ()
    execution: str = "next_open"
    direction: str = "long"
    cost_profile: str = "retail_etf"
    notes: dict[str, Any] = field(default_factory=dict)

    def size(self) -> int:
        return len(self.feature_grid) * len(self.thresholds) * len(self.holding)


TEMPLATES: dict[str, Template] = {
    "extreme_move_reversal": Template(
        "extreme_move_reversal", Mechanism.LIQUIDITY, "mean_reversion",
        "Large standardized declines may revert as liquidity providers are compensated.",
        "threshold_event", ("zret(h=1,vol_n=63)", "zret(h=3,vol_n=63)", "zret(h=5,vol_n=63)"),
        (-2.0, -2.5), "<", (1, 5)),
    "volume_confirmed_breakout": Template(
        "volume_confirmed_breakout", Mechanism.BEHAVIORAL, "trend",
        "Breakouts on heavy volume may signal information diffusing slowly (under-reaction).",
        "threshold_event", ("breakout(n=20)", "breakout(n=55)"), (0.0,), ">", (5, 21),
        extra_conditions=(("rel_volume(n=20)", ">", 1.5),)),
    "oversold_rsi": Template(
        "oversold_rsi", Mechanism.UNKNOWN, "mean_reversion",
        "Classic oscillator claim; tested as a candidate feature with no presumption of value.",
        "threshold_event", ("rsi(n=2)", "rsi(n=14)"), (10.0, 30.0), "<", (5,)),
    "trend_state": Template(
        "trend_state", Mechanism.BEHAVIORAL, "trend",
        "Trend persistence; tested against buy-and-hold on the same universe.",
        "state_position", ("sma_ratio(n=50)", "sma_ratio(n=100)", "sma_ratio(n=200)", "sma_slope(n=50,k=5)"),
        (0.0,), ">", (21,), execution="next_close"),
}


def _neighbors(values: tuple, v) -> list:
    i = values.index(v)
    return [values[j] for j in (i - 1, i + 1) if 0 <= j < len(values)]


def generate(template_name: str, universe: str, budget: int = 24) -> list[HypothesisSpec]:
    t = TEMPLATES[template_name]
    if t.size() > budget:
        raise BudgetExceeded(f"template {template_name} expands to {t.size()} hypotheses > budget {budget}; "
                             "narrow the grid instead of sampling it")
    specs = []
    for feat, thr, hold in itertools.product(t.feature_grid, t.thresholds, t.holding):
        cond = [[feat, t.op, thr]] + [list(c) for c in t.extra_conditions]
        neigh = [{"conditions": [[f2, t.op, thr]] + [list(c) for c in t.extra_conditions]}
                 for f2 in _neighbors(t.feature_grid, feat)]
        neigh += [{"conditions": [[feat, t.op, th2]] + [list(c) for c in t.extra_conditions]}
                  for th2 in _neighbors(t.thresholds, thr)]
        params: dict[str, Any] = {"conditions": cond}
        specs.append(HypothesisSpec(
            name=f"[gen:{t.name}] {feat} {t.op} {thr} hold {hold}",
            question=f"Does {feat} {t.op} {thr} predict {hold}-session returns on {universe}?",
            statement=f"Entries when {feat} {t.op} {thr} earn positive net {hold}-session returns beating random entries.",
            mechanism=t.mechanism, rationale=t.rationale, family=t.family, universe=universe, strategy=t.strategy,
            params=params, param_neighbors=neigh, direction=t.direction, holding_period=hold, execution=t.execution,
            primary_metric="excess_vs_buy_and_hold" if t.strategy == "state_position" else "mean_net_trade_return",
            cost_profile=t.cost_profile, generated_by=f"generator:{t.name}", exploratory=True,
        ))
    return specs


def register_batch(reg: Registry, specs: list[HypothesisSpec], reason: str) -> list[str]:
    """Register every spec (skipping exact duplicates already on record) and journal the batch."""
    ids, dups = [], []
    for s in specs:
        try:
            ids.append(register_hypothesis(reg, s, reason=reason))
        except DuplicateHypothesis as e:
            dups.append(e.existing_id)
    reg.journal("generate_batch", reason=reason, hypothesis_ids=ids, duplicates_skipped=dups,
                n_generated=len(specs), total_hypotheses_registered=reg.count("hypotheses"))
    return ids
