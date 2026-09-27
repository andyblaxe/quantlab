"""Load user-supplied files (e.g. purchased Norgate/ORATS exports) into canonical schemas.

The user declares the provenance: files are ``REAL`` data, and the user must state known
limitations (flags). Column names are mapped explicitly — nothing is guessed silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quantlab.calendar import get_calendar
from quantlab.data.providers.base import DataProvider, ProviderInfo
from quantlab.provenance import DataFlag, DataLabel, Dataset, Provenance


@dataclass
class LocalCsvBarsProvider(DataProvider):
    """One CSV per symbol (``<directory>/<SYMBOL>.csv``) or one long CSV with a symbol column."""

    directory: Path
    column_map: dict[str, str] = field(default_factory=lambda: {
        "Date": "session", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    flags: frozenset[DataFlag] = frozenset()
    source: str = "local_csv"
    delay_minutes: int = 15
    unadjusted: bool = True  # declare whether the file holds raw (as-traded) prices

    def __post_init__(self) -> None:
        flags = set(self.flags)
        if not self.unadjusted:
            flags.add(DataFlag.VENDOR_ADJUSTED)
        self.flags = frozenset(flags)
        self.info = ProviderInfo(self.source, DataLabel.REAL, self.flags, notes=f"files in {self.directory}")

    def get_daily_bars(self, symbols, start, end):
        cal = get_calendar()
        frames = []
        for sym in symbols:
            path = Path(self.directory) / f"{sym}.csv"
            if not path.exists():
                continue  # absent data is absent — never fabricated
            raw = pd.read_csv(path).rename(columns=self.column_map)
            raw["session"] = pd.to_datetime(raw["session"]).dt.normalize()
            raw = raw[(raw["session"] >= pd.Timestamp(start)) & (raw["session"] <= pd.Timestamp(end))]
            raw = raw[raw["session"].map(cal.is_session)]
            raw["symbol"] = sym
            raw["available_at"] = cal.daily_bar_available_at(pd.DatetimeIndex(raw["session"]), self.delay_minutes)
            raw["source"] = self.source
            frames.append(raw[["symbol", "session", "open", "high", "low", "close", "volume", "available_at", "source"]])
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["symbol", "session", "open", "high", "low", "close", "volume", "available_at", "source"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return Dataset("bars_daily", df, Provenance(DataLabel.REAL, self.source, self.flags))
