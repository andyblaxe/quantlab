"""The test vault: technical safeguards around the untouched out-of-sample partition.

* :meth:`TestVault.development_view` truncates every panel field at the development end date.
  All development-stage research (screening, validation, walk-forward, parameter selection) goes
  through this view, so test-period data is not merely "not looked at" — it is not in memory.
* :meth:`TestVault.unseal` returns the full panel only if the hypothesis is FROZEN (its final spec
  locked after model selection). Every unseal is written to the append-only ``vault_access`` log.
* A hypothesis may unseal once. A second unseal requires ``acknowledge_contamination=True`` and is
  permanently recorded as contaminated; reports surface it and the hypothesis can no longer be
  promoted on the strength of its test result.

These are guard rails, not cryptography: someone determined to peek can read the raw files. The
point is that the *default* path is clean, and any deviation leaves an indelible record.
"""

from __future__ import annotations

import pandas as pd

from quantlab.research.hypotheses import HypothesisStatus, hypothesis_status
from quantlab.research.registry import Registry
from quantlab.research.splits import SplitPlan


class VaultSealed(PermissionError):
    pass


def truncate_panel(panel: dict[str, pd.DataFrame], end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {k: v.loc[v.index <= end] for k, v in panel.items()}


class TestVault:
    __test__ = False  # not a pytest test class

    def __init__(self, reg: Registry, plan: SplitPlan) -> None:
        self.reg = reg
        self.plan = plan

    def development_view(self, panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return truncate_panel(panel, self.plan.development_end)

    def accesses(self, hypothesis_id: str) -> list[dict]:
        return self.reg.find("vault_access", hypothesis_id=hypothesis_id)

    def is_contaminated(self, hypothesis_id: str) -> bool:
        return any(a["contaminated"] == "True" for a in self.accesses(hypothesis_id))

    def unseal(self, panel: dict[str, pd.DataFrame], hypothesis_id: str, experiment_id: str | None,
               reason: str, acknowledge_contamination: bool = False) -> dict[str, pd.DataFrame]:
        status = hypothesis_status(self.reg, hypothesis_id)
        if status != HypothesisStatus.FROZEN and not (status == HypothesisStatus.TESTED and acknowledge_contamination):
            raise VaultSealed(
                f"{hypothesis_id} is {status}; the test partition opens only for FROZEN hypotheses "
                "(finish model selection and freeze the final spec first)")
        prior = self.accesses(hypothesis_id)
        contaminated = bool(prior)
        if contaminated and not acknowledge_contamination:
            raise VaultSealed(
                f"{hypothesis_id} already opened the vault ({prior[0]['id']}). Re-opening contaminates the "
                "test; pass acknowledge_contamination=True to proceed on the record")
        self.reg.append("vault_access",
                        {"hypothesis_id": hypothesis_id, "experiment_id": experiment_id, "contaminated": contaminated},
                        {"reason": reason, "plan": self.plan.to_dict(), "prior_accesses": [p["id"] for p in prior]})
        self.reg.journal("vault_unseal", reason=reason, hypothesis_id=hypothesis_id, experiment_id=experiment_id,
                         contaminated=contaminated)
        return panel

    def total_accesses(self) -> int:
        return self.reg.count("vault_access")
