"""Exchange calendar and information-availability rules.

The single most common source of look-ahead bias in daily research is treating a bar's close as
known *at* the close. Official closing prints (closing auction) are disseminated after the bell,
and vendor EOD files later still. We therefore define:

    available_at(daily bar for session D) = session_close(D) + publication_delay

using the real NYSE calendar, so early-close days (e.g. 13:00 ET the day after Thanksgiving) are
handled correctly. All timestamps are timezone-aware UTC.
"""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xc
import numpy as np
import pandas as pd

CALENDAR_START = "1990-01-01"


@lru_cache(maxsize=4)
def _calendar(name: str) -> xc.ExchangeCalendar:
    return xc.get_calendar(name, start=CALENDAR_START)


class TradingCalendar:
    def __init__(self, name: str = "XNYS") -> None:
        self.name = name
        self._cal = _calendar(name)

    # -- sessions -------------------------------------------------------------------------------
    def sessions(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
        """Trading sessions (tz-naive dates) in [start, end]."""
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        start = max(start, self._cal.first_session)
        end = min(end, self._cal.last_session)
        return self._cal.sessions_in_range(start, end)

    def is_session(self, date: str | pd.Timestamp) -> bool:
        return bool(self._cal.is_session(pd.Timestamp(date).normalize()))

    def next_session(self, date: str | pd.Timestamp) -> pd.Timestamp:
        return self._cal.next_session(pd.Timestamp(date).normalize())

    def previous_session(self, date: str | pd.Timestamp) -> pd.Timestamp:
        return self._cal.previous_session(pd.Timestamp(date).normalize())

    # -- times ----------------------------------------------------------------------------------
    def session_open(self, date: str | pd.Timestamp) -> pd.Timestamp:
        return self._cal.session_open(pd.Timestamp(date).normalize())

    def session_close(self, date: str | pd.Timestamp) -> pd.Timestamp:
        return self._cal.session_close(pd.Timestamp(date).normalize())

    def session_opens(self, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._cal.opens.reindex(sessions).values, tz="UTC")

    def session_closes(self, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._cal.closes.reindex(sessions).values, tz="UTC")

    def is_early_close(self, date: str | pd.Timestamp) -> bool:
        d = pd.Timestamp(date).normalize()
        return d in self._cal.early_closes

    def daily_bar_available_at(self, sessions: pd.DatetimeIndex, delay_minutes: int = 15) -> pd.DatetimeIndex:
        """When the completed daily bar for each session is first usable by the system."""
        closes = self.session_closes(pd.DatetimeIndex(sessions))
        if closes.isna().any():
            bad = pd.DatetimeIndex(sessions)[np.asarray(closes.isna())]
            raise ValueError(f"not trading sessions on {self.name}: {list(bad[:5])}")
        return closes + pd.Timedelta(minutes=delay_minutes)


@lru_cache(maxsize=4)
def get_calendar(name: str = "XNYS") -> TradingCalendar:
    return TradingCalendar(name)
