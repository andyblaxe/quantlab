"""Canonical table schemas.

Every table carries point-in-time metadata:

* ``available_at`` — UTC timestamp at which the system could first have known the row.
  This is the column every look-ahead check keys on.
* ``source`` — provider identifier.

Conventions
-----------
* Daily bars are stored **unadjusted** (as traded). Split/dividend adjustment is computed from the
  ``corporate_actions`` table (see :mod:`quantlab.data.adjust`) so that retroactively revised
  vendor-adjusted histories never leak into research.
* ``session`` is a tz-naive trading date; all other timestamps are tz-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: dict[str, str]  # column -> kind: "str" | "float" | "int" | "date" | "ts" | "bool"
    key: tuple[str, ...]
    optional: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Return a list of schema violations (empty list = valid)."""
        problems: list[str] = []
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            problems.append(f"{self.name}: missing columns {missing}")
            return problems
        for col, kind in {**self.columns, **{k: v for k, v in self.optional.items() if k in df.columns}}.items():
            s = df[col]
            if kind == "float" and not pd.api.types.is_numeric_dtype(s):
                problems.append(f"{self.name}.{col}: expected numeric, got {s.dtype}")
            elif kind == "int" and not pd.api.types.is_integer_dtype(s) and not pd.api.types.is_float_dtype(s):
                problems.append(f"{self.name}.{col}: expected integer, got {s.dtype}")
            elif kind == "date":
                if not pd.api.types.is_datetime64_any_dtype(s):
                    problems.append(f"{self.name}.{col}: expected datetime, got {s.dtype}")
                elif getattr(s.dt, "tz", None) is not None:
                    problems.append(f"{self.name}.{col}: session dates must be tz-naive")
            elif kind == "ts":
                if not isinstance(s.dtype, pd.DatetimeTZDtype):
                    problems.append(f"{self.name}.{col}: expected tz-aware UTC timestamp, got {s.dtype}")
                elif str(s.dt.tz) != "UTC":
                    problems.append(f"{self.name}.{col}: timestamps must be UTC, got {s.dt.tz}")
            elif kind == "bool" and not pd.api.types.is_bool_dtype(s):
                problems.append(f"{self.name}.{col}: expected bool, got {s.dtype}")
        if self.key and len(df):
            dup = df.duplicated(list(self.key)).sum()
            if dup:
                problems.append(f"{self.name}: {dup} duplicate rows on key {self.key}")
        return problems


BARS_DAILY = TableSchema(
    name="bars_daily",
    columns={
        "symbol": "str",
        "session": "date",
        "open": "float",
        "high": "float",
        "low": "float",
        "close": "float",
        "volume": "float",
        "available_at": "ts",
        "source": "str",
    },
    optional={"vwap": "float", "trades": "int"},
    key=("symbol", "session"),
    description="Unadjusted daily OHLCV per trading session.",
)

BARS_INTRADAY = TableSchema(
    name="bars_intraday",
    columns={
        "symbol": "str",
        "bar_start": "ts",
        "bar_end": "ts",
        "open": "float",
        "high": "float",
        "low": "float",
        "close": "float",
        "volume": "float",
        "available_at": "ts",
        "source": "str",
    },
    key=("symbol", "bar_start"),
    description="Unadjusted intraday bars. available_at >= bar_end.",
)

CORPORATE_ACTIONS = TableSchema(
    name="corporate_actions",
    columns={
        "symbol": "str",
        "ex_date": "date",
        "action": "str",  # "split" | "dividend"
        "value": "float",  # split: new shares per old share; dividend: cash per share
        "available_at": "ts",  # announcement time (conservatively: ex_date open if unknown)
        "source": "str",
    },
    key=("symbol", "ex_date", "action"),
)

EARNINGS_EVENTS = TableSchema(
    name="earnings_events",
    columns={
        "symbol": "str",
        "fiscal_period": "str",
        "announce_time": "ts",  # actual release timestamp
        "timing": "str",  # "BMO" | "AMC" | "DURING" | "UNKNOWN"
        "available_at": "ts",  # == announce_time for realised events
        "source": "str",
    },
    optional={"scheduled_available_at": "ts", "eps_actual": "float", "revenue_actual": "float"},
    key=("symbol", "fiscal_period"),
    description="Realised earnings announcements. Expected dates known in advance are a separate concept"
    " (scheduled_available_at) and must not be confused with the realised timestamp.",
)

EARNINGS_ESTIMATES = TableSchema(
    name="earnings_estimates",
    columns={
        "symbol": "str",
        "fiscal_period": "str",
        "metric": "str",  # "EPS" | "REVENUE"
        "mean": "float",
        "stdev": "float",
        "n_estimates": "int",
        "available_at": "ts",  # snapshot time of the consensus
        "source": "str",
    },
    key=("symbol", "fiscal_period", "metric", "available_at"),
)

OPTION_QUOTES_EOD = TableSchema(
    name="option_quotes_eod",
    columns={
        "underlying": "str",
        "session": "date",
        "expiration": "date",
        "strike": "float",
        "right": "str",  # "C" | "P"
        "bid": "float",
        "ask": "float",
        "volume": "float",
        "open_interest": "float",
        "underlying_price": "float",
        "quote_time": "ts",  # time the quote snapshot was taken
        "available_at": "ts",
        "source": "str",
    },
    optional={"iv": "float", "delta": "float", "gamma": "float", "theta": "float", "vega": "float",
              "style": "str", "multiplier": "float"},
    key=("underlying", "session", "expiration", "strike", "right"),
    description="Historical end-of-day option quotes. Vendor IV/Greeks are optional; we recompute.",
)

MACRO_SERIES = TableSchema(
    name="macro_series",
    columns={
        "series_id": "str",
        "observation_date": "date",  # period the value describes
        "value": "float",
        "available_at": "ts",  # vintage / release time
        "source": "str",
    },
    key=("series_id", "observation_date", "available_at"),
    description="Macro, rates and index-level series stored as vintages (one row per revision).",
)

INDEX_MEMBERSHIP = TableSchema(
    name="index_membership",
    columns={
        "index_id": "str",
        "symbol": "str",
        "start_date": "date",
        "end_date": "date",  # NaT = still a member
        "available_at": "ts",  # when the membership change was announced
        "source": "str",
    },
    key=("index_id", "symbol", "start_date"),
)

CLASSIFICATIONS = TableSchema(
    name="classifications",
    columns={
        "symbol": "str",
        "scheme": "str",  # e.g. "SIC", "GICS"
        "level": "str",  # e.g. "sector", "industry"
        "code": "str",
        "start_date": "date",
        "end_date": "date",
        "available_at": "ts",
        "source": "str",
    },
    key=("symbol", "scheme", "level", "start_date"),
)

SHORT_INTEREST = TableSchema(
    name="short_interest",
    columns={
        "symbol": "str",
        "settlement_date": "date",
        "short_interest": "float",
        "available_at": "ts",  # FINRA publication, typically ~7 business days after settlement
        "source": "str",
    },
    optional={"avg_daily_volume": "float", "days_to_cover": "float"},
    key=("symbol", "settlement_date"),
)

FUNDAMENTALS = TableSchema(
    name="fundamentals",
    columns={
        "symbol": "str",
        "fiscal_period": "str",
        "period_end": "date",
        "metric": "str",
        "value": "float",
        "available_at": "ts",  # filing acceptance time (EDGAR), not period end
        "source": "str",
    },
    key=("symbol", "fiscal_period", "metric", "available_at"),
)

ALL_SCHEMAS: dict[str, TableSchema] = {
    s.name: s
    for s in [
        BARS_DAILY, BARS_INTRADAY, CORPORATE_ACTIONS, EARNINGS_EVENTS, EARNINGS_ESTIMATES,
        OPTION_QUOTES_EOD, MACRO_SERIES, INDEX_MEMBERSHIP, CLASSIFICATIONS, SHORT_INTEREST, FUNDAMENTALS,
    ]
}
