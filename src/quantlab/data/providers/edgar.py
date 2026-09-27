"""SEC EDGAR adapter: earnings announcement timestamps from 8-K Item 2.02 filings.

An 8-K with Item 2.02 ("Results of Operations and Financial Condition") is how US issuers furnish
their earnings release. EDGAR records the filing's acceptance time to the second, so pre-market,
intraday and after-close releases can be told apart.

Rules (see DATA.md):

* ``announce_time`` = ``available_at`` = the 8-K **acceptance time** (true UTC in the submissions
  JSON; verified against issuers' published release times). The wire press release usually precedes
  the filing by minutes, occasionally by hours, so the timestamp is never early (no look-ahead) but
  can be late: an event may be assigned a day-0 session one day after the true reaction.
* Only original ``8-K`` filings (not ``8-K/A``). Item 2.02 exists from 2004-08-23; earlier releases
  (old Item 12) are not captured.
* Issuers also file Item 2.02 for pre-announcements, restatements and investor days. Fiscal
  periods are the period ends (``reportDate``) of the issuer's 10-Q/10-K filings. A period's
  candidates are the Item 2.02 filings after its end and no later than the next period's end (at
  most ``MATCH_WINDOW_DAYS``); the one whose lag after period end is closest to the issuer's usual
  lag (median over the nearest periods with a single candidate) is the release. Other Item 2.02 filings are
  dropped and counted. The usual lag uses the issuer's whole history: this only decides *which*
  filing is the quarterly release (a reader of the filing knows that), never its timestamp.
  The latest quarter appears only once its 10-Q/10-K is on file.
* The ticker → CIK map is today's (``company_tickers.json``): delisted and renamed issuers are
  missing, hence ``SURVIVORSHIP_RISK``.

Fair access: SEC requires a descriptive User-Agent (``EDGAR_USER_AGENT``) and at most 10 requests/s.
"""

from __future__ import annotations

import time

import pandas as pd

from quantlab.calendar import get_calendar
from quantlab.config import get_settings
from quantlab.data.providers.base import DataProvider, ProviderInfo, ProviderNotConfigured
from quantlab.data.providers.free import NY, http_get
from quantlab.provenance import DataFlag, DataLabel, Dataset, Provenance

EVENT_COLUMNS = ["symbol", "fiscal_period", "announce_time", "timing", "available_at", "source",
                 "cik", "accession"]
PERIODIC_FORMS = frozenset({"10-Q", "10-K", "10-K405", "10-QT", "10-KT"})
ITEM_202_START = pd.Timestamp("2004-08-23")
MATCH_WINDOW_DAYS = 75
USUAL_LAG_NEIGHBOURS = 8


def classify_timing(announce_time: pd.Timestamp) -> str:
    """BMO / DURING / AMC relative to the NYSE session on the announcement's ET date.

    Announcements on non-session days are BMO: they precede the next session's open.
    """
    cal = get_calendar()
    day = announce_time.tz_convert(NY).tz_localize(None).normalize()
    if not cal.is_session(day):
        return "BMO"
    if announce_time < cal.session_open(day):
        return "BMO"
    if announce_time >= cal.session_close(day):
        return "AMC"
    return "DURING"


class EdgarProvider(DataProvider):
    info = ProviderInfo(
        "edgar", DataLabel.REAL, frozenset({DataFlag.SURVIVORSHIP_RISK}),
        notes="8-K Item 2.02 acceptance timestamps (from 2004-08-23); current ticker→CIK map only.",
    )
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
    MIN_INTERVAL_S = 0.15  # stay well under SEC's 10 requests/s

    def __init__(self) -> None:
        self._last_request = 0.0
        self._cik: dict[str, int] | None = None

    # --- HTTP ----------------------------------------------------------------------------------
    def _headers(self) -> dict:
        ua = get_settings().edgar_user_agent
        if not ua:
            raise ProviderNotConfigured("EDGAR_USER_AGENT is not set (SEC requires a contact User-Agent)")
        return {"User-Agent": ua}

    def _get_json(self, url: str):
        wait = self.MIN_INTERVAL_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            return http_get(url, headers=self._headers()).json()
        finally:
            self._last_request = time.monotonic()

    # --- pure parsers --------------------------------------------------------------------------
    @staticmethod
    def parse_ticker_map(payload: dict) -> dict[str, int]:
        """``company_tickers.json`` → {TICKER: CIK}. Tiingo-style class shares use '-' (BRK-B)."""
        return {str(r["ticker"]).upper().replace(".", "-"): int(r["cik_str"]) for r in payload.values()}

    @staticmethod
    def parse_filings(pages: list[dict]) -> pd.DataFrame:
        """Columnar filing pages (``filings.recent`` and the older ``files`` pages) → one frame."""
        cols = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "items"]
        frames = [pd.DataFrame({c: p.get(c, [None] * len(p.get("accessionNumber", []))) for c in cols})
                  for p in pages]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        df = df.drop_duplicates("accessionNumber")
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        df["acceptance"] = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce")
        df["items"] = df["items"].fillna("").astype(str)
        return df.sort_values("acceptance", kind="stable").reset_index(drop=True)

    @staticmethod
    def parse_earnings(filings: pd.DataFrame, symbol: str, cik: int) -> tuple[pd.DataFrame, dict]:
        """Select one earnings release per fiscal period. Returns (events, counts)."""
        f = filings
        is_202 = (f["form"] == "8-K") & f["items"].str.split(",").map(lambda xs: "2.02" in [x.strip() for x in xs])
        rel = f[is_202 & f["acceptance"].notna()
                & (f["acceptance"] >= ITEM_202_START.tz_localize(NY))].sort_values("acceptance")
        rel = rel.assign(day=rel["acceptance"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize())
        ends = pd.to_datetime(f.loc[f["form"].isin(PERIODIC_FORMS), "reportDate"], errors="coerce")
        ends = pd.DatetimeIndex(ends.dropna().unique()).sort_values()
        # candidates for a period: releases after it ends and no later than the next period's end
        cands = []
        for i, pe in enumerate(ends):
            nxt = ends[i + 1] if i + 1 < len(ends) else pe + pd.Timedelta(days=MATCH_WINDOW_DAYS)
            c = rel[(rel["day"] > pe) & (rel["day"] <= min(nxt, pe + pd.Timedelta(days=MATCH_WINDOW_DAYS)))]
            if len(c):
                cands.append((pe, c.assign(lag=(c["day"] - pe).dt.days)))
        # the issuer's usual reporting lag around each period, from nearby single-candidate periods
        # (issuers' lags drift over the years, e.g. AAPL ~18 days in 2006, ~31 in 2019)
        single = pd.Series({pe: c["lag"].iloc[0] for pe, c in cands if len(c) == 1}, dtype=float)

        def usual_lag(pe: pd.Timestamp) -> float:
            if single.empty:
                return 30.0
            near = single.drop(pe, errors="ignore")
            near = near.iloc[abs((near.index - pe).days).argsort()[:USUAL_LAG_NEIGHBOURS]] if len(near) else single
            return float(near.median())

        rows, used = [], set()
        for pe, c in cands:
            c = c[~c["accessionNumber"].isin(used)]
            if c.empty:
                continue
            r = c.loc[(c["lag"] - usual_lag(pe)).abs().idxmin()]  # ties → earliest (stable order)
            used.add(r["accessionNumber"])
            ts = r["acceptance"]
            rows.append({"symbol": symbol, "fiscal_period": pe.strftime("%Y-%m-%d"), "announce_time": ts,
                         "timing": classify_timing(ts), "available_at": ts, "source": "edgar",
                         "cik": int(cik), "accession": r["accessionNumber"]})
        events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
        events["announce_time"] = pd.to_datetime(events["announce_time"], utc=True)
        events["available_at"] = pd.to_datetime(events["available_at"], utc=True)
        events["cik"] = events["cik"].astype("int64")
        counts = {"item_202_filings": int(len(rel)), "matched": int(len(events)),
                  "unmatched": int(len(rel) - len(used)),
                  "ambiguous_periods": int(sum(len(c) > 1 for _, c in cands))}
        return events, counts

    # --- provider API --------------------------------------------------------------------------
    def cik_map(self) -> dict[str, int]:
        if self._cik is None:
            self._cik = self.parse_ticker_map(self._get_json(self.TICKERS_URL))
        return self._cik

    def filings(self, cik: int) -> pd.DataFrame:
        sub = self._get_json(self.SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
        pages = [sub["filings"]["recent"]]
        for extra in sub["filings"].get("files", []):
            pages.append(self._get_json(self.SUBMISSIONS_URL.format(name=extra["name"])))
        return self.parse_filings(pages)

    def get_earnings_events(self, symbols, start, end):
        cmap = self.cik_map()
        missing = [s for s in symbols if s.upper() not in cmap]
        if missing:
            raise KeyError(f"no CIK in SEC's current ticker map for {missing} (delisted or renamed?)")
        frames, self.last_counts = [], {}
        for sym in symbols:
            ev, counts = self.parse_earnings(self.filings(cmap[sym.upper()]), sym, cmap[sym.upper()])
            day = ev["announce_time"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
            frames.append(ev[(day >= pd.Timestamp(start)) & (day <= pd.Timestamp(end))])
            self.last_counts[sym] = counts
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVENT_COLUMNS)
        return Dataset("earnings_events", df, Provenance(DataLabel.REAL, "edgar", self.info.flags))
