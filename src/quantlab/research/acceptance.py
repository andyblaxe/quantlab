"""Statistical acceptance / rejection criteria for signals.

The criteria object is **snapshotted into every hypothesis at registration** (and included in its
spec hash), so thresholds cannot be loosened after results are seen. Changing the defaults only
affects hypotheses registered afterwards.

See RESEARCH_METHODOLOGY.md for the rationale behind each threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AcceptanceCriteria:
    version: str = "2026-09-27.1"
    # sample adequacy
    min_effective_n_inconclusive: int = 30
    min_effective_n_accept: int = 100
    # screening (training data only)
    screen_p_value: float = 0.05
    # validation
    validation_sharpe_ratio_of_train: float = 0.5
    walk_forward_min_positive_fold_frac: float = 0.60
    fdr_q: float = 0.10  # Benjamini–Yekutieli over the whole family
    dsr_min_prob: float = 0.95  # deflated Sharpe ratio probability
    bootstrap_ci_level: float = 0.95
    cost_stress_multiplier: float = 2.0
    outlier_trim_frac: float = 0.05
    max_single_year_pnl_share: float = 0.40
    min_param_neighbor_positive_frac: float = 0.70
    # promotion to LIVE_ELIGIBLE (forward testing)
    live_min_paper_days: int = 60
    live_min_paper_trades: int = 30
    live_min_forward_percentile: float = 0.05
    live_max_slippage_ratio: float = 1.5

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_toml(path: str | Path) -> "AcceptanceCriteria":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return AcceptanceCriteria(**data.get("acceptance", data))


@dataclass
class CriterionResult:
    name: str
    passed: bool | None  # None = not evaluable (insufficient data) → never counts as a pass
    value: float | str | None
    threshold: float | str | None
    explanation: str


@dataclass
class AcceptanceDecision:
    stage: str
    outcome: str  # "PASS" | "FAIL" | "INCONCLUSIVE"
    results: list[CriterionResult] = field(default_factory=list)

    @property
    def failed(self) -> list[CriterionResult]:
        return [r for r in self.results if r.passed is False]

    @property
    def unevaluable(self) -> list[CriterionResult]:
        return [r for r in self.results if r.passed is None]

    def to_dict(self) -> dict:
        return {"stage": self.stage, "outcome": self.outcome,
                "results": [asdict(r) for r in self.results]}


def decide(stage: str, results: list[CriterionResult], inconclusive: bool = False) -> AcceptanceDecision:
    """All criteria must pass. Any failure → FAIL; missing evidence → INCONCLUSIVE (never PASS)."""
    if any(r.passed is False for r in results):
        outcome = "FAIL"
    elif inconclusive or any(r.passed is None for r in results):
        outcome = "INCONCLUSIVE"
    else:
        outcome = "PASS"
    return AcceptanceDecision(stage, outcome, results)
