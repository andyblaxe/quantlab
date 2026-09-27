"""Train / validation / untouched-test partitioning and walk-forward folds.

Time-ordered, never shuffled. Between partitions sits an **embargo** of at least the longest holding
period, so a trade entered at the end of training cannot realise its outcome inside validation.
Trades are assigned to a partition only if their entry *and* exit lie inside it (purging).

The split plan is registered once as a project event *before* research begins; changing it later
creates a new, visible project event that every subsequent report must mention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from quantlab.research.registry import Registry


@dataclass(frozen=True)
class SplitPlan:
    train_end: str
    validation_end: str
    test_end: str | None = None  # None = through the end of available data
    embargo_sessions: int = 21
    data_start: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def boundaries(self, sessions: pd.DatetimeIndex) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        """Inclusive (start, end) session bounds of each partition given available sessions."""
        s = pd.DatetimeIndex(sessions).sort_values()
        if self.data_start:
            s = s[s >= pd.Timestamp(self.data_start)]
        te, ve = pd.Timestamp(self.train_end), pd.Timestamp(self.validation_end)
        if not te < ve:
            raise ValueError("train_end must precede validation_end")
        train = s[s <= te]
        after_train = s[s > te]
        val = after_train[self.embargo_sessions:]
        val = val[val <= ve]
        after_val = s[s > ve][self.embargo_sessions:]
        test = after_val if self.test_end is None else after_val[after_val <= pd.Timestamp(self.test_end)]
        out = {}
        for name, part in (("train", train), ("validation", val), ("test", test)):
            if len(part) == 0:
                raise ValueError(f"split plan leaves the {name} partition empty for this data")
            out[name] = (part[0], part[-1])
        return out

    @property
    def development_end(self) -> pd.Timestamp:
        """Last session any development-stage research may see."""
        return pd.Timestamp(self.validation_end)


def partition_of(entry: pd.Series, exit_: pd.Series, bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    """Label each trade with the partition that contains both its entry and exit (else 'embargo')."""
    out = pd.Series("embargo", index=entry.index, dtype=object)
    for name, (a, b) in bounds.items():
        inside = (entry >= a) & (exit_ <= b)
        out[inside] = name
    return out


def register_split_plan(reg: Registry, plan: SplitPlan, reason: str) -> str:
    prev = active_split_plan(reg)
    pid = reg.append("project_events", {"kind": "split_plan"},
                     {"plan": plan.to_dict(), "reason": reason,
                      "replaces": prev.to_dict() if prev else None,
                      "hypotheses_registered_before": reg.count("hypotheses")})
    reg.journal("register_split_plan", reason=reason, plan=plan.to_dict(), project_event_id=pid,
                changed_after_research_started=bool(prev and reg.count("hypotheses")))
    return pid


def active_split_plan(reg: Registry) -> SplitPlan | None:
    rows = reg.find("project_events", order="DESC", limit=1, kind="split_plan")
    return SplitPlan(**rows[0]["payload"]["plan"]) if rows else None


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(
    sessions: pd.DatetimeIndex,
    min_train_sessions: int = 756,
    test_sessions: int = 252,
    purge_sessions: int = 21,
    expanding: bool = True,
    end: pd.Timestamp | None = None,
) -> list[Fold]:
    """Anchored (expanding) or rolling walk-forward folds over [sessions[0], end].

    Each fold's test window starts ``purge_sessions`` after its training window ends, so labels of
    late training samples cannot overlap the test window. Callers must pass ``end`` =
    development end — walk-forward never touches the untouched test partition.
    """
    s = pd.DatetimeIndex(sessions).sort_values()
    if end is not None:
        s = s[s <= pd.Timestamp(end)]
    folds = []
    k = 0
    train_end_i = min_train_sessions - 1
    while True:
        test_start_i = train_end_i + 1 + purge_sessions
        test_end_i = test_start_i + test_sessions - 1
        if test_end_i >= len(s):
            break
        train_start_i = 0 if expanding else max(0, train_end_i - min_train_sessions + 1)
        folds.append(Fold(k, s[train_start_i], s[train_end_i], s[test_start_i], s[test_end_i]))
        k += 1
        train_end_i += test_sessions
    return folds
