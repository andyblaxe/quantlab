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
| Earnings events | actual announcement timestamp (EDGAR acceptance time when implemented) |
| Index membership | announcement time; mask begins the day after announcement at the earliest |

`session` columns are tz-naive trading dates validated against the XNYS calendar (`exchange_calendars`,
history from 1990).

## Tables (canonical schemas, `src/quantlab/data/schemas.py`)

`bars_daily`, `bars_intraday`, `corporate_actions`, `earnings_events`, `earnings_estimates`,
`option_quotes_eod`, `macro_series` (vintage rows), `index_membership`, `classifications`,
`short_interest`, `fundamentals`. Schemas exist for all of them; **only `bars_daily`,
`corporate_actions`, `earnings_events` and `macro_series` have working producers today** (see
PROJECT_STATE.md).

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
