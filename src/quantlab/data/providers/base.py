"""Abstract data-provider interfaces.

Research code depends only on these interfaces, never on a vendor. Replacing free data with an
institutional vendor means writing one adapter class — nothing else changes.

Every provider method returns a :class:`~quantlab.provenance.Dataset`, i.e. a frame that conforms
to a canonical schema (:mod:`quantlab.data.schemas`) *and* carries its provenance label and flags.
Providers must never fabricate rows: missing data is simply absent (and surfaced by validation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from quantlab.provenance import DataFlag, DataLabel, Dataset


class ProviderNotConfigured(RuntimeError):
    """Raised when a provider needs a credential or local file that has not been configured."""


class CapabilityNotSupported(NotImplementedError):
    """Raised when a provider cannot supply a requested data type (never silently faked)."""


@dataclass(frozen=True)
class ProviderInfo:
    provider_id: str
    label: DataLabel
    flags: frozenset[DataFlag] = field(default_factory=frozenset)
    paid: bool = False
    notes: str = ""


class DataProvider(ABC):
    """Base class; subclasses implement only the capabilities they genuinely have."""

    info: ProviderInfo

    # --- prices ----------------------------------------------------------------------------------
    def get_daily_bars(self, symbols: list[str], start: str, end: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide daily bars")

    def get_intraday_bars(self, symbols: list[str], start: str, end: str, freq: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide intraday bars")

    def get_corporate_actions(self, symbols: list[str], start: str, end: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide corporate actions")

    # --- events / fundamentals ------------------------------------------------------------------
    def get_earnings_events(self, symbols: list[str], start: str, end: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide earnings events")

    def get_earnings_estimates(self, symbols: list[str], start: str, end: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide earnings estimates")

    def get_fundamentals(self, symbols: list[str], metrics: list[str], start: str, end: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide fundamentals")

    def get_short_interest(self, symbols: list[str], start: str, end: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide short interest")

    # --- options -----------------------------------------------------------------------------------
    def get_option_quotes_eod(self, underlying: str, start: str, end: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide historical option quotes")

    # --- macro / reference -------------------------------------------------------------------------
    def get_macro_series(self, series_id: str, start: str, end: str, vintages: bool = False) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide macro series")

    def get_index_membership(self, index_id: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide index membership")

    def get_classifications(self, symbols: list[str], scheme: str) -> Dataset:
        raise CapabilityNotSupported(f"{self.info.provider_id} does not provide classifications")

    # --- helpers -----------------------------------------------------------------------------------
    def capabilities(self) -> list[str]:
        """Names of the methods this provider overrides (i.e. genuinely supports)."""
        caps = []
        for name in [
            "get_daily_bars", "get_intraday_bars", "get_corporate_actions", "get_earnings_events",
            "get_earnings_estimates", "get_fundamentals", "get_short_interest", "get_option_quotes_eod",
            "get_macro_series", "get_index_membership", "get_classifications",
        ]:
            if getattr(type(self), name) is not getattr(DataProvider, name):
                caps.append(name)
        return caps


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
