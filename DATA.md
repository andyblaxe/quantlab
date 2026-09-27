# DATA.md — Data model, sources and point-in-time rules

## Provenance labels

Every dataset is a `Dataset(name, frame, provenance)`; the label cannot be separated from the frame.

| Label | Meaning |
|---|---|
| `REAL` | Fetched from an external source as-is (or loaded from a purchased file). |
| `DERIVED` | Computed from REAL inputs only. |
| `SIMULATED` | Generated, or computed from anything simulated. **Simulation is contagious** — no amount of processing turns it into DERIVED or REAL. |

Known biases travel with the data as flags (union on derivation): `SURVIVORSHIP_RISK`,
`NO_PIT_MEMBERSHIP`, `VENDOR_ADJUSTED`, `PARTIAL_VOLUME`, `MODEL_PRICED`, `REVISABLE_NO_VINTAGE`,
`UNVERIFIED_SOURCE`.

## Timestamps

Every table has `available_at` (tz-aware UTC): **when the system could first have known the row.**

| Data | `available_at` rule |
|---|---|
| Daily bar for session D | NYSE close of D (13:00 ET on early-close days) + publication delay (default 15 min, `QUANTLAB_BAR_PUBLICATION_DELAY_MIN`) |
| Cboe VIX-family close | close + 30 min (VIX calculates until 16:15 ET) |
| FRED series without vintages | observation date + 1 day, 17:00 ET (conservative; flagged `REVISABLE_NO_VINTAGE`) |
| ALFRED vintages | vintage (realtime_start) date, 23:59 ET (FRED publishes no time of day) |
| Corporate actions (Tiingo) | ex-date session open (true announcement is earlier; conservative) |
| Earnings events (EDGAR) | 8-K Item 2.02 acceptance time (to the second, UTC). Never early; can be late when the wire release precedes the filing, which can push day 0 one session late |
| Index membership | announcement time; mask begins the day after announcement at the earliest |

`session` columns are tz-naive trading dates validated against the XNYS calendar (`exchange_calendars`,
history from 1990).

## Tables (canonical schemas, `src/quantlab/data/schemas.py`)

`bars_daily`, `bars_intraday`, `corporate_actions`, `earnings_events`, `earnings_estimates`,
`option_quotes_eod`, `macro_series` (vintage rows), `index_membership`, `classifications`,
`short_interest`, `fundamentals`. Schemas exist for all of them; **only `bars_daily`,
`corporate_actions`, `earnings_events` and `macro_series` have working producers today** (see
PROJECT_STATE.md).

## Security master (identity is permanent, tickers are dated attributes)

`data/security_master.py`, schemas `securities`, `security_identifiers`, `security_events`.

* `security_id` is a vendor's permanent id, namespaced (`TIINGO:US000000000038`, `NORGATE:<assetid>`,
  `CRSP:<permno>`, `SHARADAR:<permaticker>`). It never changes and is never reused.
* Ticker, name, exchange, CIK, CUSIP and FIGI are rows in `security_identifiers` with validity
  intervals, so reuse (DELL: Dell Inc. to 2013, Dell Technologies from 2018) and renames (PCLN →
  BKNG) resolve by date: `SecurityMaster.resolve("DELL", "2010-06-01")`.
* `securities` holds first/last listed session, delisting reason/value/return where a source has
  them. `security_events` holds listings, delistings, M&A, bankruptcies, spin-offs, ticker/name/
  exchange changes and halts. Splits and dividends stay in `corporate_actions`.
* `listed_on(date)` is the survivorship-free base universe (everything listed then, including later
  delistings). `tradable_mask(volume)` = listed ∧ bar with volume > 0 ∧ not halted.
* Historical index membership stays in `index_membership`, keyed by `security_id`.
* Once a master exists, the `symbol` column of every other canonical table holds the `security_id`;
  the panel's columns are security ids, so no engine downstream depends on tickers.
* **Status:** schemas, class and tests exist; **no source fills it yet** (Tiingo cannot supply
  pre-2013 delisted prices, ticker history or membership — see DATA_GAPS.md G1).
* Universe kinds (`research/data.universe_flags_from_kinds`): `etf`, `stock_pit`, `stock_current`.
  `stock_current` (and any undeclared universe) carries `SURVIVORSHIP_BIASED_UNIVERSE`, which makes
  results PRELIMINARY and blocks promotion.

## Data quality and the research-integrity gate (permanent rule)

Data problems follow the data into every result and are enforced by the promotion gate
(RESEARCH_METHODOLOGY.md §1a, `research/integrity.py`):

* `SURVIVORSHIP_BIASED_UNIVERSE` ⇒ survivorship finding; `MODEL_PRICED` ⇒ unreliable-prices finding.
* A dataset saved with `force=True` despite validation ERRORs contributes a finding to every result
  that uses it: timestamp/session errors ⇒ timestamp integrity, price errors ⇒ unreliable prices,
  action errors ⇒ corporate-action error (`research/session.forced_validation_errors`).
* Provider-level caveats that do not by themselves invalidate an edge (`SURVIVORSHIP_RISK` on ETF
  sets, `VENDOR_ADJUSTED`, `REVISABLE_NO_VINTAGE`) stay as flags and are reported, but do not block.
* These are integrity *failures*, distinct from insufficient evidence (small samples, lack of
  significance), which is judged against each hypothesis's pre-registered thresholds.
* Known gaps and what would close them: DATA_GAPS.md.

## Earnings announcements (SEC EDGAR)

`data/providers/edgar.py`, CLI `quantlab data fetch-earnings --symbols AAPL,MSFT,...`
(needs `EDGAR_USER_AGENT`; ~2 requests per company, throttled under SEC's 10/s).

* Source: the issuer's submissions JSON; only original `8-K`s listing Item 2.02 (from 2004-08-23,
  when Item 2.02 was introduced). `8-K/A` amendments are ignored.
* One release per fiscal period. Periods are the period ends of the issuer's 10-Q/10-K filings.
  Candidates are Item 2.02 filings after the period end and no later than the next period end
  (≤ 75 days); the chosen one has the lag closest to the issuer's usual lag, the median over the 8
  nearest unambiguous periods. This drops pre-announcements, restatements and second filings
  (checked on AAPL 2005/2006/2008/2019, MSFT 2004, JPM 2012). Dropped filings are counted per symbol.
* `timing`: BMO / DURING / AMC against that day's NYSE open and close (early closes handled);
  filings on non-session days are BMO for the next session.
* The newest quarter appears only once its 10-Q/10-K is filed.
* Ticker → CIK uses SEC's **current** ticker map, so delisted/renamed issuers are missing
  (`SURVIVORSHIP_RISK`; DATA_GAPS.md G3). With a security master, fetch by CIK instead.

## Prices: raw storage, point-in-time adjustment

Bars are stored **unadjusted**. `data/adjust.build_panel` constructs a forward total-return index:

```
TRI_t = TRI_{t-1} * (close_t * split_ratio_t + dividend_t) / close_{t-1}
```

This uses only information at or before t, so adjusted history never changes when later splits or
dividends arrive (tested: `test_adjustment_is_point_in_time`). Vendor back-adjusted series (e.g.
Stooq) are accepted but flagged `VENDOR_ADJUSTED`: ratio features remain valid, level features
(price filters, dollar volume) do not.

Conventions: split ratio = new shares per old share; dividend per pre-split share if a split and
dividend share an ex-date. After a data gap the next return spans the gap from the last valid
close (never silently zero).

## Validation (`data/validation.py`)

Errors block saving (unless forced, which is recorded): schema violations, non-positive prices,
high/low not bracketing open/close, negative volume, non-session dates, `available_at` earlier
than the session close + required delay.
Warnings are recorded in the manifest: missing sessions, |move| > 40% with no corporate action,
stale prices (≥5 identical closes), zero-volume bars, corporate actions announced after ex-date.

## Store (`data/store.py`)

`<data_dir>/store/<table>/<sha256>.parquet` + manifest (provenance, validation report, forced flag).
Write-once: a version file is never overwritten; content hash is verified on every load. DuckDB
queries run directly over Parquet.

## Providers

| Provider | Class | Status | Label / flags |
|---|---|---|---|
| Synthetic market | `SyntheticProvider` | Working, tested | SIMULATED |
| Local CSV (purchased data) | `LocalCsvBarsProvider` | Working | REAL + user-declared flags |
| Stooq daily | `StooqProvider` | Parser tested on fixtures; **live fetch untested (blocked in build env)** | REAL; VENDOR_ADJUSTED, SURVIVORSHIP_RISK, UNVERIFIED_SOURCE |
| Tiingo daily + actions | `TiingoProvider` | Parser tested; live untested; needs `TIINGO_API_KEY` | REAL; SURVIVORSHIP_RISK |
| FRED / ALFRED | `FredProvider` | Parser tested; live untested; needs `FRED_API_KEY` | REAL (flag REVISABLE_NO_VINTAGE without vintages) |
| Cboe indices (VIX, VIX3M, …) | `CboeIndexProvider` | Parser tested; live untested | REAL |
| Options EOD, intraday, estimates, PIT membership | — | **Interface only.** Paid data required (see PLAN.md §6) | — |

Providers raise `CapabilityNotSupported` for anything they cannot supply and
`ProviderNotConfigured` when a key is missing. They never fabricate rows.

## Synthetic market (SIMULATED)

GARCH(1,1) market with Student-t shocks and a calm/stressed Markov regime; sector and idiosyncratic
returns; OHLC with overnight/intraday split; volume rising with |return|; quarterly dividends;
splits; quarterly earnings jumps; a VIX-like index. Optional planted effects (`idio_ar1`,
`momentum_loading`) exist solely to measure the research pipeline's power and false-positive rate.
