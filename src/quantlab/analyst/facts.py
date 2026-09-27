"""The fact layer: the ONLY way the Research Analyst reads research results.

Rules that make hallucination structurally difficult:

* Every query reads the append-only registry (and hash-verified artifacts). Nothing is cached from
  conversation or model memory.
* Every value handed to a report or answer is a :class:`Fact` carrying the record ID it came from,
  so the rendered text can cite ``E-000012`` / ``SIG-000004`` / ``H-000004``.
* The fact layer is read-only: it has no method that writes results. (Reports are appended to the
  ``reports`` table by :mod:`quantlab.analyst.reports`, never experiments or signals.)
* "Not performed" is a first-class answer: queries return empty results, and callers must say so.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.research.artifacts import ArtifactStore
from quantlab.research.catalog import all_signals, latest_record
from quantlab.research.registry import Registry


@dataclass(frozen=True)
class Fact:
    value: Any
    source: str  # e.g. "experiments/E-000012:train.trades_net.mean"

    def __str__(self) -> str:
        return f"{self.value} [{self.source}]"


def dig(d: dict, path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


class FactBase:
    def __init__(self, reg: Registry, artifacts: ArtifactStore | None = None) -> None:
        self.reg = reg
        self.art = artifacts

    # --- inventory ---------------------------------------------------------------------------
    def counts(self) -> dict:
        hyps = self.reg.find("hypotheses")
        hstat = Counter(self.reg.current_status("hypothesis", h["id"]) for h in hyps)
        exps = [e for e in self.reg.find("experiments") if e["kind"] in ("development", "untouched_test")]
        sigs = all_signals(self.reg)
        meas = [e for e in exps if e["payload"].get("kind") == "measurement"]
        return {
            "hypotheses_registered": len(hyps),
            "hypotheses_generated_automatically": sum(1 for h in hyps if str(h["generated_by"]).startswith("generator:")),
            "hypotheses_by_family": dict(Counter(h["family"] for h in hyps)),
            "hypotheses_by_status": dict(hstat),
            "hypotheses_tested": len({e["hypothesis_id"] for e in exps}),
            "experiments_completed": len(exps),
            "experiments_running": 0,  # evaluations are synchronous; nothing is ever left running
            "untouched_tests_run": sum(1 for e in exps if e["kind"] == "untouched_test"),
            "vault_openings": self.reg.count("vault_access"),
            "contaminated_tests": sum(1 for v in self.reg.find("vault_access") if v["contaminated"] == "True"),
            "signals_by_status": dict(Counter(s["Status"] for s in sigs)),
            "measurements_by_outcome": dict(Counter(e["status"] for e in meas)),
            "data_labels_used": sorted({dig(e["payload"], "data.label") for e in exps if dig(e["payload"], "data.label")}),
            "reports_generated": self.reg.count("reports"),
        }

    def uses_only_simulated_data(self) -> bool:
        labels = self.counts()["data_labels_used"]
        return bool(labels) and set(labels) == {"SIMULATED"}

    # --- hypotheses / experiments ---------------------------------------------------------------
    def hypotheses(self) -> list[dict]:
        out = []
        for h in self.reg.find("hypotheses"):
            spec = h["payload"]["spec"]
            out.append({"id": h["id"], "name": h["name"], "family": h["family"], "generated_by": h["generated_by"],
                        "registered_at": h["created_at"], "status": self.reg.current_status("hypothesis", h["id"]),
                        "statement": spec["statement"], "mechanism": spec["mechanism"], "strategy": spec["strategy"],
                        "universe": spec["universe"], "exploratory": spec.get("exploratory", False)})
        return out

    def hypothesis(self, hid: str) -> dict:
        return self.reg.get("hypotheses", hid)

    def experiments_for(self, hid: str) -> list[dict]:
        return [e for e in self.reg.find("experiments", hypothesis_id=hid) if e["kind"] != "artifacts"]

    def development_experiment(self, hid: str) -> dict | None:
        rows = self.reg.find("experiments", order="DESC", limit=1, hypothesis_id=hid, kind="development")
        return rows[0] if rows else None

    def test_experiment(self, hid: str) -> dict | None:
        rows = self.reg.find("experiments", order="DESC", limit=1, hypothesis_id=hid, kind="untouched_test")
        return rows[0] if rows else None

    def artifacts_for(self, eid: str) -> dict:
        for row in self.reg.find("experiments", kind="artifacts"):
            if row["payload"].get("experiment_id") == eid:
                return row["payload"]["artifacts"]
        return {}

    def load_artifact(self, eid: str, name: str) -> pd.DataFrame | None:
        refs = self.artifacts_for(eid)
        if name not in refs or self.art is None:
            return None
        return self.art.load(refs[name])

    # --- signals ---------------------------------------------------------------------------------
    def signals(self, status: str | None = None) -> list[dict]:
        sigs = all_signals(self.reg)
        return [s for s in sigs if status is None or s["Status"] == status]

    def signal(self, sid: str) -> dict | None:
        return latest_record(self.reg, sid)

    def signal_history(self, sid: str) -> list[dict]:
        return self.reg.find("status_events", entity_type="signal", entity_id=sid)

    def resolve(self, ref: str) -> tuple[str | None, str | None]:
        """Accept 'H-000004', 'SIG-000004', '4', 'Strategy 4' → (hypothesis_id, signal_id)."""
        digits = "".join(ch for ch in ref if ch.isdigit())
        if not digits:
            return None, None
        n = int(digits)
        hid, sid = f"H-{n:06d}", f"SIG-{n:06d}"
        try:
            self.reg.get("hypotheses", hid)
        except KeyError:
            return None, None
        return hid, sid if self.signal(sid) else None

    def rejected(self) -> list[dict]:
        out = []
        for s in self.signals("REJECTED"):
            ev = [e for e in self.signal_history(s["Signal_ID"]) if e["to_status"] == "REJECTED"]
            out.append({"signal_id": s["Signal_ID"], "hypothesis_id": s.get("Hypothesis_ID"), "name": s.get("Name"),
                        "reason": ev[-1]["payload"]["reason"] if ev else None, "status_event": ev[-1]["id"] if ev else None})
        return out

    def ranked_signals(self) -> list[dict]:
        """Non-rejected signals, strongest evidence first (adjusted p, then validation mean)."""
        live = [s for s in self.signals() if s["Status"] not in ("REJECTED", "RETIRED")]
        order = {"LIVE_ELIGIBLE": 0, "PAPER_TRADING": 1, "ACCEPTED": 2, "VALIDATING": 3, "DEGRADED": 4, "EXPERIMENTAL": 5}
        return sorted(live, key=lambda s: (order.get(s["Status"], 9), s.get("Adjusted_P_Value") if s.get("Adjusted_P_Value") is not None else 1.0))

    def measurements(self) -> list[dict]:
        out = []
        for e in self.reg.find("experiments", kind="development"):
            if e["payload"].get("kind") == "measurement":
                h = self.reg.get("hypotheses", e["hypothesis_id"])
                out.append({"experiment_id": e["id"], "hypothesis_id": e["hypothesis_id"], "name": h["name"],
                            "outcome": e["status"], "estimate": dig(e["payload"], "train.estimate"),
                            "p_value": dig(e["payload"], "train.p_value"),
                            "q_value": dig(e["payload"], "multiple_testing.q_value"),
                            "status_detail": dig(e["payload"], "train.status")})
        return out

    # --- time --------------------------------------------------------------------------------------
    def changes_since(self, iso_ts: str | None) -> dict:
        def after(rows):
            return [r for r in rows if iso_ts is None or r["created_at"] > iso_ts]
        return {
            "since": iso_ts,
            "new_hypotheses": [{"id": r["id"], "name": r["name"]} for r in after(self.reg.find("hypotheses"))],
            "new_experiments": [{"id": r["id"], "hypothesis_id": r["hypothesis_id"], "kind": r["kind"], "outcome": r["status"]}
                                for r in after(self.reg.find("experiments")) if r["kind"] != "artifacts"],
            "status_changes": [{"id": r["id"], "entity": r["entity_id"], "to": r["to_status"],
                                "from": r["payload"].get("from_status"), "reason": r["payload"].get("reason")}
                               for r in after(self.reg.find("status_events")) if r["entity_type"] == "signal"],
            "vault_openings": [r["id"] for r in after(self.reg.find("vault_access"))],
            "split_plan_changes": [r["id"] for r in after(self.reg.find("project_events", kind="split_plan"))],
        }

    def last_report(self, kind: str = "research") -> dict | None:
        rows = self.reg.find("reports", order="DESC", limit=1, kind=kind)
        return rows[0] if rows else None

    def journal(self, limit: int = 50) -> list[dict]:
        return self.reg.find("journal", order="DESC", limit=limit)

    def calibration(self) -> dict | None:
        rows = self.reg.find("project_events", order="DESC", limit=1, kind="calibration")
        return {"id": rows[0]["id"], **rows[0]["payload"]} if rows else None

    # --- search / cross-signal -------------------------------------------------------------------
    def search(self, terms: list[str]) -> list[dict]:
        terms = [t.lower() for t in terms if t]
        hits = []
        for h in self.hypotheses():
            spec = self.reg.get("hypotheses", h["id"])["payload"]["spec"]
            blob = " ".join([h["name"], h["statement"], spec["question"], spec["rationale"], str(spec["params"])]).lower()
            if any(t in blob for t in terms):
                hits.append(h)
        return hits

    def return_correlations(self, signal_ids: list[str] | None = None) -> pd.DataFrame:
        """Correlation of development-period daily net returns between evaluated signals."""
        series = {}
        for s in self.signals():
            if signal_ids and s["Signal_ID"] not in signal_ids:
                continue
            eid = (s.get("Experiment_IDs") or [None])[0]
            df = self.load_artifact(eid, "daily") if eid else None
            if df is not None and "net" in df and df["net"].std() > 0:
                series[s["Signal_ID"]] = df["net"]
        if len(series) < 2:
            return pd.DataFrame()
        return pd.DataFrame(series).corr()

    def least_correlated_pairs(self, k: int = 5) -> list[tuple[str, str, float]]:
        c = self.return_correlations([s["Signal_ID"] for s in self.ranked_signals()])
        if c.empty:
            return []
        pairs = [(a, b, float(c.loc[a, b])) for i, a in enumerate(c.index) for b in c.columns[i + 1:]]
        return sorted(pairs, key=lambda x: abs(x[2]))[:k]


def fmt(x, kind: str = "num") -> str:
    """Deterministic number formatting used by every report/answer."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if kind == "pct":
        return f"{x * 100:.2f}%"
    if kind == "bp":
        return f"{x * 1e4:.1f} bp"
    if kind == "p":
        return f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}"
    if kind == "money":
        return f"${x:,.0f}"
    if isinstance(x, (int, np.integer)):
        return f"{x:,d}"
    return f"{x:.3g}" if abs(x) < 1000 else f"{x:,.0f}"
