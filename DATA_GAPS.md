# DATA_GAPS.md — data quality and coverage gaps

_Created 2026-09-27. One entry per known gap: what is missing, what it biases, how the platform
contains it today, and what would close it. Update an entry when the gap changes; never delete one —
mark it CLOSED with the date and the fix._

Severity: **BLOCKING** (research of that kind cannot produce promotable evidence) · **MATERIAL**
(biases results; flagged) · **MINOR** (documented, conservative handling in place).

## G1. Survivorship-free stock universe — BLOCKING for single-stock research

**Gap.** No point-in-time list of which US stocks existed (and were index members) on each past date,
with prices for the ones that later delisted.

**What Tiingo (our current access) provides — measured 2026-09-27:**

| Item | Finding |
|---|---|
| Delisted stocks with prices | `supported_tickers.zip` lists 13,147 US stocks whose data ends before Sep 2026, but almost all ended **2013 or later** (by end year: 2008: 5, 2009: 48, 2010: 49, 2011: 81, 2012: 66, then hundreds to ~1,800 per year). |
| Pre-2013 exits | Of 20 large companies that left the S&P 500 in 2004–2012 (Bear Stearns, Lehman, Wachovia, Countrywide, National City, Merrill Lynch, Wyeth, Schering-Plough, TXU, KeySpan, Dow Jones, Ambac, WorldCom, Enron, …), **none** has full price history; Merrill Lynch has 2006-12 → 2008-12 only. |
| Identity of delisted companies | The fundamentals metadata endpoint lists 20,325 companies (12,518 inactive, incl. Lehman, Enron, WorldCom) with a permanent `permaTicker`, but on the free tier sector/industry/SIC/location fields read "Field not available for free/evaluation", and a permaTicker with no price history (Lehman) returns no bars. |
| permaTicker | Works as the ticker in `/tiingo/daily/<id>/prices` (undocumented; documented only as a search-response "placeholder"). |
| Reused tickers | 1,958 US-stock rows share a ticker. `/tiingo/daily/dell/prices` for 2010 returns **nothing**: the ticker resolves to today's Dell Technologies; the old Dell Inc. is unreachable by ticker. Wrong-company risk: `G` is Genpact (not Gillette), `WB` Weibo (not Wachovia), `CC` Chemours (not Circuit City). |
| Renames | The new ticker carries the full history (BKNG from 1999; META from 2012) **and** the old ticker's row may remain (PCLN 1999–2018): the same security can appear twice. |
| Ticker / name / exchange history | Not provided (only the current ticker per permaTicker). |
| Delisting date | Implied by the last bar / `endDate`. |
| Delisting reason, delisting return | Not provided. Metadata description starts "DELISTED -" (free text). The final bar can be a flat zero-volume placeholder (TWTR 2022-10-28 at $53.70 vs the $54.20 cash-out). |
| Historical index membership | Not provided. |
| Fundamentals | Free tier: 3 years of the DOW 30 only. |
| Search | Returns active securities only; no parameter to include delisted ones. |

**Conclusion.** Tiingo cannot materially reduce survivorship bias for any study whose sample starts
before ~2013 (the 2008–09 failures are exactly the missing ones), and has no index-membership history
at all. PEAD's EDGAR events start in 2004 and its training period ends 2012.

**Containment in the platform (enforced, 2026-09-27; part of the permanent research-integrity gate,
RESEARCH_METHODOLOGY.md §1a):**
* Universe-level flag `SURVIVORSHIP_BIASED_UNIVERSE`. Every universe must declare its kind
  (`etf`, `stock_pit`, `stock_current`); an undeclared universe is treated as `stock_current`.
* Records carrying the flag have `Evidence_Grade = "PRELIMINARY / SURVIVORSHIP-BIASED"`. The catalog
  refuses ACCEPTED, PAPER_TRADING and LIVE_ELIGIBLE for them regardless of approval, and the pipeline
  withholds the one-shot untouched test so the vault is not spent on biased data.

**To close.** A source with (a) delisted US stocks with prices back to at least 1998–2004, (b)
point-in-time index membership (S&P 500 at minimum; S&P 100 / Russell for other studies), (c) a
permanent security id with ticker history, loaded into the security master (DATA.md). See
DATA_PROVIDERS.md for the evaluation. Ideally also (d) delisting returns (CRSP only).

## G2. Delisting returns — MATERIAL for strategies that hold stocks into a delisting

**Gap.** The value a holder actually received when a stock delisted (cash-out, final OTC price,
zero in liquidation). Without it, a backtest that holds a stock through a bankruptcy delisting
silently books the last exchange close instead of a large loss.
**Available from:** CRSP (DLRET / delisting price). Norgate: explicitly not provided. Sharadar: delist
events and acquisition counterparties, not returns.
**Containment:** the security master has `delisting_return`/`delisting_value` columns (NaN until a
source fills them). Planned rule: when unknown, bankruptcy/regulatory delistings are booked at −100%
for the remaining position in stress tests, and results report the share of trades affected.

## G3. EDGAR ticker → CIK map is current-only — MATERIAL (depends on G1)

**Gap.** `company_tickers.json` maps today's tickers, so delisted issuers' earnings events cannot be
fetched by ticker. The EDGAR submissions themselves exist for delisted CIKs.
**To close:** resolve CIK from the security master (CIK as a dated identifier; Sharadar exposes SEC
filing links per permaticker) and fetch by CIK instead of ticker.

## G4. ETF universes chosen with hindsight — MINOR

The v1 ETF sets are funds that exist today. ETFs that closed are absent. The sector SPDRs and the
index ETFs existed throughout their samples, so the bias is small. Tiingo's provider-level
`SURVIVORSHIP_RISK` flag remains on these results, but they are not promotion-blocked.

## G5. Earnings estimates / surprises — BLOCKING for estimate-based signals

No consensus estimates, so PEAD must proxy surprise by the announcement-window abnormal return.
Needs I/B/E/S, Zacks, Estimize or similar.

## G6. Historical options data — BLOCKING for option strategies

Unchanged from PLAN.md §11. Model-priced chains are SIMULATED and never evidence.

## G7. Announcement timing precision — MINOR

EDGAR acceptance time can trail the wire release by minutes to hours; day 0 may be one session late
(never early). Event windows should start at day −1.
