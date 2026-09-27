"""Data validation. Bad data is reported, never silently "fixed".

Each check returns :class:`Issue` records with a severity:

* ``ERROR`` — data is unusable as-is (schema violations, impossible prices, timestamps that would
  permit look-ahead). The store refuses to save datasets with errors unless explicitly forced,
  and a forced save is recorded in the dataset manifest.
* ``WARNING`` — suspicious but possibly real (huge moves without a corporate action, stale prices,
  missing sessions). Recorded with the dataset so downstream reports can cite it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.calendar import get_calendar
from quantlab.data.schemas import ALL_SCHEMAS, BARS_DAILY, CORPORATE_ACTIONS, TableSchema


@dataclass(frozen=True)
class Issue:
    severity: str  # "ERROR" | "WARNING"
    check: str
    message: str
    count: int = 0
    examples: tuple = ()


@dataclass
class ValidationReport:
    table: str
    n_rows: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "n_rows": self.n_rows,
            "ok": self.ok,
            "issues": [
                {"severity": i.severity, "check": i.check, "message": i.message, "count": i.count,
                 "examples": [str(e) for e in i.examples]}
                for i in self.issues
            ],
        }

    def summary(self) -> str:
        lines = [f"{self.table}: {self.n_rows} rows, {len(self.errors)} errors, {len(self.warnings)} warnings"]
        for i in self.issues:
            lines.append(f"  [{i.severity}] {i.check}: {i.message}")
        return "\n".join(lines)


def _examples(df: pd.DataFrame, mask, cols, n=5) -> tuple:
    sub = df.loc[mask, cols].head(n)
    return tuple(tuple(r) for r in sub.itertuples(index=False))


def validate_schema(df: pd.DataFrame, schema: TableSchema) -> list[Issue]:
    return [Issue("ERROR", "schema", p) for p in schema.validate(df)]


def validate_bars(
    df: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    calendar_name: str = "XNYS",
    extreme_move: float = 0.40,
    stale_days: int = 5,
    min_delay_minutes: int = 0,
) -> ValidationReport:
    rep = ValidationReport("bars_daily", len(df))
    rep.issues += validate_schema(df, BARS_DAILY)
    if rep.errors:
        return rep
    cal = get_calendar(calendar_name)

    # --- impossible values -----------------------------------------------------------------------
    px = df[["open", "high", "low", "close"]]
    nonpos = (px <= 0).any(axis=1) | px.isna().any(axis=1)
    if nonpos.any():
        rep.issues.append(Issue("ERROR", "non_positive_or_missing_price", "OHLC <= 0 or NaN",
                                int(nonpos.sum()), _examples(df, nonpos, ["symbol", "session"])))
    tol = 1e-9
    bad_hl = (df["high"] + tol < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] - tol > df[["open", "close", "high"]].min(axis=1))
    if bad_hl.any():
        rep.issues.append(Issue("ERROR", "ohlc_inconsistent", "high/low do not bracket open/close",
                                int(bad_hl.sum()), _examples(df, bad_hl, ["symbol", "session"])))
    negvol = df["volume"] < 0
    if negvol.any():
        rep.issues.append(Issue("ERROR", "negative_volume", "volume < 0", int(negvol.sum()),
                                _examples(df, negvol, ["symbol", "session"])))

    # --- calendar / timestamp integrity --------------------------------------------------------
    sessions = pd.DatetimeIndex(df["session"].unique())
    valid_sessions = cal.sessions(sessions.min(), sessions.max()) if len(sessions) else sessions
    non_session = ~df["session"].isin(valid_sessions)
    if non_session.any():
        rep.issues.append(Issue("ERROR", "non_trading_session", "rows dated on non-trading days",
                                int(non_session.sum()), _examples(df, non_session, ["symbol", "session"])))
    ok_sess = df.loc[~non_session, "session"]
    if len(ok_sess):
        closes = cal.session_closes(pd.DatetimeIndex(ok_sess))
        min_avail = pd.Series(closes, index=ok_sess.index) + pd.Timedelta(minutes=min_delay_minutes)
        early = df.loc[~non_session, "available_at"] < min_avail
        if early.any():
            rep.issues.append(Issue(
                "ERROR", "available_before_close",
                "available_at precedes the session close (+ required delay): would allow look-ahead",
                int(early.sum()), _examples(df.loc[~non_session], early, ["symbol", "session", "available_at"])))

    # --- per-symbol continuity -----------------------------------------------------------------
    ratio_lookup = set()
    if actions is not None and len(actions):
        ratio_lookup = set(zip(actions["symbol"], actions["ex_date"]))
    gaps_total, gap_examples = 0, []
    extreme_rows, stale_rows = [], []
    for sym, g in df.sort_values("session").groupby("symbol"):
        g_sessions = pd.DatetimeIndex(g["session"])
        expected = cal.sessions(g_sessions.min(), g_sessions.max())
        missing = expected.difference(g_sessions)
        if len(missing):
            gaps_total += len(missing)
            gap_examples += [(sym, d.date()) for d in missing[:2]]
        c = g["close"].to_numpy()
        if len(c) > 1:
            r = c[1:] / c[:-1] - 1
            idx = np.where(np.abs(r) > extreme_move)[0] + 1
            for k in idx:
                d = g["session"].iloc[k]
                if (sym, d) not in ratio_lookup:
                    extreme_rows.append((sym, d.date(), round(float(r[k - 1]), 4)))
        if len(c) > stale_days:
            same = np.r_[False, c[1:] == c[:-1]]
            run = pd.Series(same).groupby((~pd.Series(same)).cumsum()).cumsum().to_numpy()
            if (run >= stale_days).any():
                stale_rows.append((sym, int(run.max())))
    if gaps_total:
        rep.issues.append(Issue("WARNING", "missing_sessions",
                                "trading sessions absent between a symbol's first and last bar",
                                gaps_total, tuple(gap_examples[:5])))
    if extreme_rows:
        rep.issues.append(Issue("WARNING", "extreme_move_without_action",
                                f"|close-to-close move| > {extreme_move:.0%} with no corporate action (possible bad"
                                " print or missing split)", len(extreme_rows), tuple(extreme_rows[:5])))
    if stale_rows:
        rep.issues.append(Issue("WARNING", "stale_prices", f">= {stale_days} consecutive identical closes",
                                len(stale_rows), tuple(stale_rows[:5])))
    zero_vol = df["volume"] == 0
    if zero_vol.any():
        rep.issues.append(Issue("WARNING", "zero_volume", "zero-volume bars (no fills will be simulated on them)",
                                int(zero_vol.sum()), _examples(df, zero_vol, ["symbol", "session"])))
    return rep


def validate_actions(actions: pd.DataFrame, calendar_name: str = "XNYS") -> ValidationReport:
    rep = ValidationReport("corporate_actions", len(actions))
    rep.issues += validate_schema(actions, CORPORATE_ACTIONS)
    if rep.errors or not len(actions):
        return rep
    cal = get_calendar(calendar_name)
    bad_kind = ~actions["action"].isin(["split", "dividend"])
    if bad_kind.any():
        rep.issues.append(Issue("ERROR", "unknown_action", "action must be split|dividend", int(bad_kind.sum())))
    nonpos = actions["value"] <= 0
    if nonpos.any():
        rep.issues.append(Issue("ERROR", "non_positive_value", "split ratio / dividend must be > 0",
                                int(nonpos.sum()), _examples(actions, nonpos, ["symbol", "ex_date"])))
    opens = cal.session_opens(pd.DatetimeIndex(actions["ex_date"]))
    late = pd.Series(np.asarray(actions["available_at"] > opens), index=actions.index)
    if late.any():
        rep.issues.append(Issue(
            "WARNING", "announced_after_ex_date",
            "action first known after the ex-date open; strategies could not have anticipated it",
            int(late.sum()), _examples(actions, late, ["symbol", "ex_date"])))
    return rep


def validate_generic(df: pd.DataFrame, table: str) -> ValidationReport:
    """Schema + available_at sanity for tables without bespoke checks."""
    schema = ALL_SCHEMAS[table]
    rep = ValidationReport(table, len(df))
    rep.issues += validate_schema(df, schema)
    if not rep.errors and "available_at" in df and df["available_at"].isna().any():
        rep.issues.append(Issue("ERROR", "missing_available_at",
                                "rows without an availability timestamp cannot be used point-in-time",
                                int(df["available_at"].isna().sum())))
    return rep
