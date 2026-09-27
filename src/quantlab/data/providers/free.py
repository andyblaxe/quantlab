"""Adapters for free data sources.

Status: **implemented and unit-tested against recorded response fixtures; not exercised against
the live services in the build environment** (its network policy blocks these hosts). Run
``quantlab data smoke-test`` on a machine with internet access before trusting them.

Each adapter separates ``fetch`` (HTTP) from ``parse`` (pure function), and states its known
limitations as :class:`~quantlab.provenance.DataFlag` values that propagate into every result.
"""

from __future__ import annotations

import io
import time

import httpx
import pandas as pd

from quantlab.calendar import get_calendar
from quantlab.config import get_settings
from quantlab.data.providers.base import DataProvider, ProviderInfo, ProviderNotConfigured
from quantlab.provenance import DataFlag, DataLabel, Dataset, Provenance

NY = "America/New_York"


def http_get(url: str, params: dict | None = None, headers: dict | None = None, retries: int = 3) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=30.0, follow_redirects=True)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.HTTPError,) as e:  # pragma: no cover - network
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")  # pragma: no cover


def _bars_frame(df: pd.DataFrame, symbol: str, source: str, delay_minutes: int) -> pd.DataFrame:
    cal = get_calendar()
    df = df.copy()
    df["session"] = pd.to_datetime(df["session"]).dt.normalize()
    df = df[df["session"].map(cal.is_session)].sort_values("session")
    df["symbol"] = symbol
    df["available_at"] = cal.daily_bar_available_at(pd.DatetimeIndex(df["session"]), delay_minutes)
    df["source"] = source
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df[["symbol", "session", "open", "high", "low", "close", "volume", "available_at", "source"]].reset_index(drop=True)


# =============================================================================================
# Stooq — free daily bars, no key. Prices are vendor-adjusted; no delisted symbols.
# =============================================================================================
class StooqProvider(DataProvider):
    info = ProviderInfo(
        "stooq", DataLabel.REAL,
        frozenset({DataFlag.VENDOR_ADJUSTED, DataFlag.SURVIVORSHIP_RISK, DataFlag.UNVERIFIED_SOURCE}),
        notes="Free CSV; adjusted history is revised retroactively; corporate actions not supplied.",
    )
    URL = "https://stooq.com/q/d/l/"

    @staticmethod
    def parse(text: str, symbol: str, delay_minutes: int = 15) -> pd.DataFrame:
        raw = pd.read_csv(io.StringIO(text))
        if "Date" not in raw.columns:
            raise ValueError(f"unexpected Stooq response for {symbol}: {text[:80]!r}")
        raw = raw.rename(columns={"Date": "session", "Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"})
        if "volume" not in raw:
            raw["volume"] = float("nan")
        return _bars_frame(raw, symbol, "stooq", delay_minutes)

    def get_daily_bars(self, symbols, start, end):
        delay = get_settings().bar_publication_delay_min
        frames = []
        for sym in symbols:
            r = http_get(self.URL, params={"s": f"{sym.lower()}.us", "i": "d",
                                           "d1": pd.Timestamp(start).strftime("%Y%m%d"),
                                           "d2": pd.Timestamp(end).strftime("%Y%m%d")})
            frames.append(self.parse(r.text, sym, delay))
        return Dataset("bars_daily", pd.concat(frames, ignore_index=True),
                       Provenance(DataLabel.REAL, "stooq", self.info.flags))


# =============================================================================================
# Tiingo — free tier with key; raw OHLCV plus dividends and split factors.
# =============================================================================================
class TiingoProvider(DataProvider):
    info = ProviderInfo(
        "tiingo", DataLabel.REAL, frozenset({DataFlag.SURVIVORSHIP_RISK}),
        notes="Raw prices + divCash + splitFactor, so PIT adjustment is possible. Delisted coverage partial.",
    )
    URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

    def _key(self) -> str:
        k = get_settings().tiingo_api_key
        if not k:
            raise ProviderNotConfigured("TIINGO_API_KEY is not set")
        return k.get_secret_value()

    @staticmethod
    def parse(records: list[dict], symbol: str, delay_minutes: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = pd.DataFrame(records)
        if raw.empty:
            return _bars_frame(pd.DataFrame(columns=["session", "open", "high", "low", "close", "volume"]),
                               symbol, "tiingo", delay_minutes), pd.DataFrame()
        raw["session"] = pd.to_datetime(raw["date"], utc=True).dt.tz_localize(None).dt.normalize()
        bars = _bars_frame(raw, symbol, "tiingo", delay_minutes)
        cal = get_calendar()
        acts = []
        for _, r in raw.iterrows():
            if float(r.get("splitFactor", 1.0) or 1.0) != 1.0:
                acts.append((r["session"], "split", float(r["splitFactor"])))
            if float(r.get("divCash", 0.0) or 0.0) > 0:
                acts.append((r["session"], "dividend", float(r["divCash"])))
        actions = pd.DataFrame(acts, columns=["ex_date", "action", "value"])
        # explicit dtypes so symbols with no actions don't turn the concatenated columns into object
        actions["ex_date"] = pd.to_datetime(actions["ex_date"])
        actions["value"] = actions["value"].astype(float)
        actions["symbol"] = symbol
        # announcement time unknown: conservatively treat as known at the ex-date open
        actions["available_at"] = (cal.session_opens(pd.DatetimeIndex(actions["ex_date"])) if len(actions)
                                   else pd.Series(dtype="datetime64[ns, UTC]"))
        actions["source"] = "tiingo"
        return bars, actions[["symbol", "ex_date", "action", "value", "available_at", "source"]]

    def _fetch(self, sym, start, end):
        # bars and corporate actions come from the same response: fetch once per (symbol, range)
        cache = self.__dict__.setdefault("_cache", {})
        if (sym, start, end) not in cache:
            r = http_get(self.URL.format(ticker=sym.lower()),
                         params={"startDate": start, "endDate": end, "token": self._key()})
            cache[(sym, start, end)] = self.parse(r.json(), sym, get_settings().bar_publication_delay_min)
        return cache[(sym, start, end)]

    def get_daily_bars(self, symbols, start, end):
        frames = [self._fetch(s, start, end)[0] for s in symbols]
        return Dataset("bars_daily", pd.concat(frames, ignore_index=True),
                       Provenance(DataLabel.REAL, "tiingo", self.info.flags))

    def get_corporate_actions(self, symbols, start, end):
        frames = [self._fetch(s, start, end)[1] for s in symbols]
        return Dataset("corporate_actions", pd.concat(frames, ignore_index=True),
                       Provenance(DataLabel.REAL, "tiingo", self.info.flags))


# =============================================================================================
# FRED / ALFRED — macro, rates, VIX close. Vintages via realtime_start/realtime_end.
# =============================================================================================
class FredProvider(DataProvider):
    info = ProviderInfo("fred", DataLabel.REAL, notes="FRED API; ALFRED vintages when vintages=True.")
    URL = "https://api.stlouisfed.org/fred/series/observations"

    def _key(self) -> str:
        k = get_settings().fred_api_key
        if not k:
            raise ProviderNotConfigured("FRED_API_KEY is not set")
        return k.get_secret_value()

    @staticmethod
    def parse(payload: dict, series_id: str, vintages: bool, lag_days: int = 1) -> pd.DataFrame:
        obs = pd.DataFrame(payload.get("observations", []))
        if obs.empty:
            return pd.DataFrame(columns=["series_id", "observation_date", "value", "available_at", "source"])
        obs = obs[obs["value"] != "."]  # FRED marks missing values with "."
        out = pd.DataFrame({
            "series_id": series_id,
            "observation_date": pd.to_datetime(obs["date"]),
            "value": pd.to_numeric(obs["value"]).astype(float),
        })
        if vintages:
            # realtime_start is the vintage date; FRED gives no time of day, so assume end of that day (ET).
            vint = pd.to_datetime(obs["realtime_start"])
            out["available_at"] = (vint + pd.Timedelta(hours=23, minutes=59)).dt.tz_localize(NY).dt.tz_convert("UTC")
        else:
            # Without vintages: conservatively available `lag_days` after the observation, 17:00 ET.
            out["available_at"] = (out["observation_date"] + pd.Timedelta(days=lag_days, hours=17)
                                   ).dt.tz_localize(NY).dt.tz_convert("UTC")
        out["source"] = "alfred" if vintages else "fred"
        return out.reset_index(drop=True)

    def get_macro_series(self, series_id, start, end, vintages=False):
        params = {"series_id": series_id, "api_key": self._key(), "file_type": "json",
                  "observation_start": start, "observation_end": end}
        if vintages:
            params.update({"realtime_start": "1776-07-04", "realtime_end": "9999-12-31"})
        df = self.parse(http_get(self.URL, params=params).json(), series_id, vintages)
        flags = frozenset() if vintages else frozenset({DataFlag.REVISABLE_NO_VINTAGE})
        return Dataset("macro_series", df, Provenance(DataLabel.REAL, "alfred" if vintages else "fred", flags))


# =============================================================================================
# Cboe — daily index history CSVs (VIX, VIX9D, VIX3M, VVIX, SKEW ...). Free, no key.
# =============================================================================================
class CboeIndexProvider(DataProvider):
    info = ProviderInfo("cboe", DataLabel.REAL, notes="Cboe public index history CSVs.")
    URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv"
    # VIX-family indices keep calculating until 16:15 ET; treat the close as known 30 min after the equity close.
    DELAY_MIN = 30

    @staticmethod
    def parse(text: str, index: str, delay_minutes: int = 30) -> pd.DataFrame:
        raw = pd.read_csv(io.StringIO(text))
        raw.columns = [c.strip().upper() for c in raw.columns]
        date_col = "DATE"
        value_col = "CLOSE" if "CLOSE" in raw.columns else index.upper()
        if value_col not in raw.columns:
            value_col = [c for c in raw.columns if c != date_col][-1]
        cal = get_calendar()
        d = pd.to_datetime(raw[date_col], format="mixed").dt.normalize()
        keep = d.map(cal.is_session)
        d = d[keep]
        out = pd.DataFrame({
            "series_id": index.upper(),
            "observation_date": d.values,
            "value": pd.to_numeric(raw.loc[keep, value_col], errors="coerce").astype(float).values,
        })
        out["available_at"] = cal.daily_bar_available_at(pd.DatetimeIndex(out["observation_date"]), delay_minutes)
        out["source"] = "cboe"
        return out.dropna(subset=["value"]).reset_index(drop=True)

    def get_macro_series(self, series_id, start, end, vintages=False):
        r = http_get(self.URL.format(index=series_id.upper()))
        df = self.parse(r.text, series_id, self.DELAY_MIN)
        df = df[(df["observation_date"] >= pd.Timestamp(start)) & (df["observation_date"] <= pd.Timestamp(end))]
        return Dataset("macro_series", df.reset_index(drop=True), Provenance(DataLabel.REAL, "cboe"))
