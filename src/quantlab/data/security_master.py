"""Point-in-time security master: permanent identity, dated attributes, lifecycle.

A ticker is an attribute of a security that changes (PCLN → BKNG) and gets reused (DELL was Dell
Inc. until 2013 and Dell Technologies from 2018). Research therefore keys everything on a permanent
``security_id`` and uses the master to answer, as of a date:

* which security did ticker X mean?  (:meth:`resolve`)
* what was security S called / where was it listed?  (:meth:`attribute`)
* was S listed, and actually tradable?  (:meth:`listed_mask`, :meth:`tradable_mask`)
* which securities existed at all?  (:meth:`listed_on`) — the survivorship-free base universe
* how complete is this master?  (:meth:`coverage`) — feeds DATA_GAPS.md

Vendor independence: a provider adapter's only job is to fill the three canonical tables
(``securities``, ``security_identifiers``, ``security_events``; see ``data/schemas.py``) with its
permanent ids namespaced (``TIINGO:…``, ``NORGATE:…``, ``CRSP:…``) and to key bars/actions/events by
that id. The panel's columns are then security ids, and nothing downstream (features, backtests,
statistics, reports) changes when the vendor does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.data.schemas import SECURITIES, SECURITY_EVENTS, SECURITY_IDENTIFIERS

OPEN_END = pd.Timestamp("2262-01-01")


class AmbiguousIdentifier(LookupError):
    """More than one security carried the identifier on that date (the master is inconsistent)."""


class UnknownIdentifier(LookupError):
    """No security carried the identifier on that date."""


@dataclass
class SecurityMaster:
    securities: pd.DataFrame  # SECURITIES
    identifiers: pd.DataFrame  # SECURITY_IDENTIFIERS
    events: pd.DataFrame | None = None  # SECURITY_EVENTS

    def __post_init__(self) -> None:
        problems = self.validate()
        if problems:
            raise ValueError("invalid security master:\n  " + "\n  ".join(problems))
        ids = self.identifiers.copy()
        ids["valid_to"] = ids["valid_to"].fillna(OPEN_END)
        self._ids = ids
        sec = self.securities.set_index("security_id")
        self._first = sec["first_session"]
        self._last = sec["last_session"].fillna(OPEN_END)

    # --- consistency ---------------------------------------------------------------------------
    def validate(self) -> list[str]:
        problems = SECURITIES.validate(self.securities) + SECURITY_IDENTIFIERS.validate(self.identifiers)
        if self.events is not None:
            problems += SECURITY_EVENTS.validate(self.events)
        if problems:
            return problems
        known = set(self.securities["security_id"])
        for name, df in (("identifiers", self.identifiers), ("events", self.events)):
            if df is not None and not set(df["security_id"]) <= known:
                problems.append(f"{name} reference unknown security_ids {sorted(set(df['security_id']) - known)[:5]}")
        ids = self.identifiers.assign(valid_to=self.identifiers["valid_to"].fillna(OPEN_END))
        if (ids["valid_to"] < ids["valid_from"]).any():
            problems.append("identifier intervals with valid_to < valid_from")
        # a ticker may be reused, but never by two securities at the same time
        for (id_type, value), g in ids[ids["id_type"] == "ticker"].groupby(["id_type", "value"]):
            if g["security_id"].nunique() < 2:
                continue
            g = g.sort_values("valid_from")
            if (g["valid_from"].to_numpy()[1:] <= g["valid_to"].to_numpy()[:-1]).any():
                problems.append(f"ticker {value!r} held by two securities at once")
        sec = self.securities
        if (sec["last_session"].notna() & (sec["last_session"] < sec["first_session"])).any():
            problems.append("securities with last_session < first_session")
        return problems

    # --- identity ------------------------------------------------------------------------------
    def resolve(self, value: str, as_of, id_type: str = "ticker") -> str:
        """security_id that carried ``value`` on date ``as_of``."""
        d = pd.Timestamp(as_of).normalize()
        ids = self._ids
        hit = ids[(ids["id_type"] == id_type) & (ids["value"] == value)
                  & (ids["valid_from"] <= d) & (ids["valid_to"] >= d)]["security_id"].unique()
        if len(hit) == 0:
            raise UnknownIdentifier(f"no security had {id_type} {value!r} on {d.date()}")
        if len(hit) > 1:
            raise AmbiguousIdentifier(f"{id_type} {value!r} on {d.date()}: {sorted(hit)}")
        return str(hit[0])

    def attribute(self, security_id: str, as_of, id_type: str = "ticker") -> str | None:
        """Value of a dated attribute (ticker, name, exchange, ...) on ``as_of``; None if unknown."""
        d = pd.Timestamp(as_of).normalize()
        ids = self._ids
        hit = ids[(ids["security_id"] == security_id) & (ids["id_type"] == id_type)
                  & (ids["valid_from"] <= d) & (ids["valid_to"] >= d)]
        return None if hit.empty else str(hit.sort_values("valid_from")["value"].iloc[-1])

    def history(self, security_id: str, id_type: str = "ticker") -> pd.DataFrame:
        ids = self.identifiers
        return ids[(ids["security_id"] == security_id) & (ids["id_type"] == id_type)].sort_values("valid_from")

    # --- existence and tradability ----------------------------------------------------------------
    def listed_on(self, session) -> list[str]:
        """Every security listed on ``session`` — including ones that later delisted."""
        d = pd.Timestamp(session).normalize()
        return sorted(self._first.index[(self._first <= d) & (self._last >= d)])

    def listed_mask(self, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
        cols = [c for c in columns if c in self._first.index]
        first = self._first.reindex(cols).to_numpy(dtype="datetime64[ns]")
        last = self._last.reindex(cols).to_numpy(dtype="datetime64[ns]")
        idx = index.to_numpy(dtype="datetime64[ns]")[:, None]
        m = pd.DataFrame(False, index=index, columns=columns)
        m[cols] = (idx >= first) & (idx <= last)
        return m

    def tradable_mask(self, volume: pd.DataFrame) -> pd.DataFrame:
        """Listed, with a bar, positive volume, and not inside a recorded halt, on each session."""
        m = self.listed_mask(volume.index, volume.columns) & volume.fillna(0).gt(0)
        if self.events is not None and len(self.events):
            ev = self.events[self.events["event_type"].isin(["halt", "resume"])].sort_values("event_date")
            for sid, g in ev.groupby("security_id"):
                if sid not in m.columns:
                    continue
                halted_since = None
                for r in g.itertuples():
                    if r.event_type == "halt":
                        halted_since = r.event_date
                    elif halted_since is not None:
                        m.loc[(m.index >= halted_since) & (m.index < r.event_date), sid] = False
                        halted_since = None
                if halted_since is not None:
                    m.loc[m.index >= halted_since, sid] = False
        return m

    def delisting(self, security_id: str) -> dict:
        row = self.securities.set_index("security_id").loc[security_id]
        return {k: (None if (isinstance(v, float) and np.isnan(v)) or v is pd.NaT else v)
                for k, v in row.items() if k in ("last_session", "delisting_reason", "delisting_return",
                                                  "delisting_value")}

    # --- completeness ------------------------------------------------------------------------------
    def coverage(self) -> dict:
        """What this master can and cannot answer (reported, never guessed)."""
        sec = self.securities
        delisted = sec[sec["last_session"].notna()]
        types = set(self.identifiers["id_type"])
        ev_types = set(self.events["event_type"]) if self.events is not None and len(self.events) else set()

        def frac(col):
            return float(delisted[col].notna().mean()) if col in delisted and len(delisted) else 0.0

        return {
            "securities": int(len(sec)), "delisted": int(len(delisted)),
            "ticker_history": "ticker" in types, "name_history": "name" in types,
            "exchange_history": "exchange" in types,
            "delisting_reason_coverage": frac("delisting_reason"),
            "delisting_return_coverage": frac("delisting_return"),
            "lifecycle_event_types": sorted(ev_types),
            "earliest_delisting": None if delisted.empty else str(delisted["last_session"].min().date()),
        }
