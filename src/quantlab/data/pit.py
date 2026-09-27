"""Point-in-time (PIT) access to data.

The contract: **a query "as of t" returns only rows whose ``available_at <= t``.** For revisable
series (macro data, fundamentals, estimates) where several vintages describe the same period, the
as-of view returns the latest vintage that was available at t — never a later revision.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class LookAheadError(AssertionError):
    """Raised when code attempts to use information not yet available at the decision time."""


def _ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        raise ValueError("as-of timestamps must be timezone-aware (UTC); a naive date is ambiguous"
                         " about *when* in the day the decision is made")
    return t.tz_convert("UTC")


@dataclass
class PointInTimeFrame:
    """Long-format table with an ``available_at`` column, queried only through as-of views."""

    frame: pd.DataFrame
    key: tuple[str, ...] = ()  # identity of a value across vintages, e.g. ("series_id", "observation_date")

    def __post_init__(self) -> None:
        if "available_at" not in self.frame:
            raise ValueError("PointInTimeFrame requires an available_at column")
        if self.frame["available_at"].isna().any():
            raise ValueError("rows without available_at cannot be used point-in-time")
        self.frame = self.frame.sort_values("available_at", kind="stable").reset_index(drop=True)

    def as_of(self, t) -> pd.DataFrame:
        """All rows known at t; for keyed tables, only the latest vintage of each key."""
        t = _ts(t)
        known = self.frame[self.frame["available_at"] <= t]
        if self.key:
            known = known.drop_duplicates(list(self.key), keep="last")
        return known.reset_index(drop=True)

    def assert_available(self, rows: pd.DataFrame, t) -> None:
        t = _ts(t)
        late = rows["available_at"] > t
        if late.any():
            raise LookAheadError(f"{int(late.sum())} rows used at {t} were not available until "
                                 f"{rows.loc[late, 'available_at'].min()}")


def asof_series(
    values: pd.DataFrame,
    decision_times: pd.Series | pd.DatetimeIndex,
    value_col: str = "value",
    period_col: str | None = "observation_date",
) -> pd.DataFrame:
    """For each decision time, the most recent value known at that time.

    ``values`` is a single series in long vintage format (``available_at``, ``value_col`` and
    optionally the period column). Returns a frame aligned with ``decision_times`` holding the value,
    the period it describes and when it became available (for audit).
    """
    dt = pd.DatetimeIndex(decision_times)
    if dt.tz is None:
        raise ValueError("decision_times must be tz-aware")
    v = values.sort_values("available_at", kind="stable")
    if period_col:
        # For revised series, a later vintage of an *older* period must not displace a newer period's
        # value: pick, at each decision time, the latest period known, at its latest vintage.
        v = v.sort_values(["available_at", period_col], kind="stable")
    left = pd.DataFrame({"decision_time": dt.tz_convert("UTC")}).reset_index(names="_pos")
    right = v.rename(columns={"available_at": "_avail"})
    right["_avail"] = right["_avail"].dt.tz_convert("UTC")
    if period_col:
        # running max of period so a late-arriving revision of an old period doesn't win
        right = right.assign(_cum_period=right[period_col].cummax())
        right = right[right[period_col] == right["_cum_period"]].drop(columns="_cum_period")
    out = pd.merge_asof(left.sort_values("decision_time"), right, left_on="decision_time",
                        right_on="_avail", direction="backward")
    out = out.sort_values("_pos").set_index(dt)
    cols = [value_col] + ([period_col] if period_col else []) + ["_avail"]
    return out[cols].rename(columns={"_avail": "value_available_at"})
