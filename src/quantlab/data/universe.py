"""Investment universes, point-in-time.

A universe answers "which securities could the strategy have considered at time t?". Using today's
index constituents for a 2008 backtest is survivorship bias: it silently excludes every company
that later failed or was removed. Two kinds are supported:

* :class:`MembershipUniverse` — built from a PIT membership table (start/end dates with announcement
  ``available_at``). Only as good as the data; free sources rarely have it.
* :class:`StaticUniverse` — a fixed list. Honest for ETF sets that existed throughout the test
  window; **flagged** ``SURVIVORSHIP_RISK`` when used for single stocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantlab.provenance import DataFlag


@dataclass
class StaticUniverse:
    name: str
    symbols: list[str]
    flags: frozenset[DataFlag] = field(default_factory=frozenset)
    inception: dict[str, pd.Timestamp] = field(default_factory=dict)  # optional first-available session

    def members(self, session) -> list[str]:
        session = pd.Timestamp(session)
        return [s for s in self.symbols if s not in self.inception or self.inception[s] <= session]

    def mask(self, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
        m = pd.DataFrame(False, index=index, columns=columns)
        for s in self.symbols:
            if s in m.columns:
                start = self.inception.get(s)
                m[s] = True if start is None else (index >= start)
        return m


@dataclass
class MembershipUniverse:
    """Membership intervals [start_date, end_date) known as of ``available_at``."""

    name: str
    membership: pd.DataFrame  # INDEX_MEMBERSHIP schema
    flags: frozenset[DataFlag] = field(default_factory=frozenset)

    def members(self, session, as_of: pd.Timestamp | None = None) -> list[str]:
        session = pd.Timestamp(session)
        m = self.membership
        if as_of is not None:
            m = m[m["available_at"] <= as_of]
        active = (m["start_date"] <= session) & (m["end_date"].isna() | (m["end_date"] > session))
        return sorted(m.loc[active, "symbol"].unique())

    def mask(self, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
        m = pd.DataFrame(False, index=index, columns=columns)
        for _, row in self.membership.iterrows():
            if row["symbol"] not in m.columns:
                continue
            end = row["end_date"] if pd.notna(row["end_date"]) else index.max() + pd.Timedelta(days=1)
            # Usable only once known: conservatively from the day after the announcement.
            known = row["available_at"].tz_convert("America/New_York").tz_localize(None).normalize()
            start = max(row["start_date"], known + pd.Timedelta(days=1))
            m.loc[(index >= start) & (index < end), row["symbol"]] = True
        return m
