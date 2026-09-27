"""Statistical acceptance / rejection criteria for signals.

The criteria object is **snapshotted into every hypothesis at registration** (and included in its
spec hash), so thresholds cannot be loosened after results are seen. Changing the defaults only
affects hypotheses registered afterwards.

These are the *configurable* numerical implementations of the permanent integrity principles
(RESEARCH_METHODOLOGY.md §1a). The class defaults are the conservative project defaults. A hypothesis
may pre-register different values suited to its methodology, but any value **looser** than the
default must carry a written ``justification`` at registration (:func:`loosened_vs_defaults`).

See RESEARCH_METHODOLOGY.md for the rationale behind each threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AcceptanceCriteria:
    version: str = "2026-09-27.2"
    # why this methodology needs values looser than the defaults (required if any are looser)
    justification: str = ""
    # sample adequacy (effective N already discounts autocorrelation/overlap; see stats.effective_n)
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
    # execution and liquidity assumptions
    allowed_executions: tuple[str, ...] = ("next_open", "next_close")
    max_participation: float = 0.10  # max fraction of ADV per trade assumed fillable

    def to_dict(self) -> dict:
        d = asdict(self)
        d["allowed_executions"] = list(self.allowed_executions)  # JSON-native, so stored specs round-trip
        return d

    @staticmethod
    def from_toml(path: str | Path) -> "AcceptanceCriteria":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return AcceptanceCriteria(**data.get("acceptance", data))


# direction in which each threshold becomes LESS strict: +1 = larger is looser, -1 = smaller is looser
LOOSER_DIRECTION: dict[str, int] = {
    "min_effective_n_inconclusive": -1, "min_effective_n_accept": -1, "screen_p_value": +1,
    "validation_sharpe_ratio_of_train": -1, "walk_forward_min_positive_fold_frac": -1, "fdr_q": +1,
    "dsr_min_prob": -1, "bootstrap_ci_level": -1, "cost_stress_multiplier": -1, "max_single_year_pnl_share": +1,
    "min_param_neighbor_positive_frac": -1, "live_min_paper_days": -1, "live_min_paper_trades": -1,
    "live_min_forward_percentile": -1, "live_max_slippage_ratio": +1, "max_participation": +1,
}


def loosened_vs_defaults(c: AcceptanceCriteria) -> list[str]:
    """Thresholds in ``c`` that are less strict than the conservative project defaults."""
    d = AcceptanceCriteria()
    out = [k for k, sign in LOOSER_DIRECTION.items() if sign * (getattr(c, k) - getattr(d, k)) > 0]
    if set(c.allowed_executions) - set(d.allowed_executions):
        out.append("allowed_executions")
    return out


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
