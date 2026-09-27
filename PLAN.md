# QuantLab — Architecture Proposal and Research Plan

This document answers the 18 pre-implementation questions from the project brief. It was written
**before** substantial code. Where later implementation changed a decision, `PROJECT_STATE.md` and
`ARCHITECTURE.md` hold the current answer; this file is kept as the original plan of record.

---

## 0. Repository inspection (what was here)

The repository `credit-research-lab` contains an unrelated, deployed **Next.js / Supabase credit-analyst
training app** (TypeScript, `src/`, `schema.sql`, root `README.md` / `ARCHITECTURE.md`). None of it is
reusable for quantitative research, and it must not be disturbed.

**Decision:** the quant platform lives in a self-contained Python project in `quantlab/`, with its own
`pyproject.toml`, `.env.example`, docs, tests and data directory. The only change outside `quantlab/` is
a one-line pointer in the root README. It can later be split into its own repository with no code
changes.

**Environment constraint discovered:** this build environment's network policy blocks the free market
data hosts (Yahoo, Stooq, FRED). Adapters for them are written and unit-tested against recorded
fixtures, but **no real market data has been downloaded here**. All numbers produced in this
environment come from **SIMULATED** data and are labeled so. They are machinery checks, not research
findings.

---

## 1. Proposed system architecture

A layered pipeline. Each layer depends only on layers below it, and communicates through typed,
provenance-labeled data objects rather than shared global state.

```
                         ┌────────────────────────────────────────────┐
  presentation           │ Dashboard (FastAPI+Jinja+Plotly)  CLI      │  read-only over stores
                         │ Research Assistant (intent → fact queries)  │
                         └───────────────▲────────────────────────────┘
  explanation            ┌───────────────┴────────────────────────────┐
                         │ Research Analyst: facts → reports           │  cannot write results
                         └───────────────▲────────────────────────────┘
  research governance    ┌───────────────┴────────────────────────────┐
                         │ Registry (append-only SQLite): hypotheses,  │
                         │ experiments, journal, signal catalog,       │
                         │ test-vault access log, reports              │
                         └───────────────▲────────────────────────────┘
  research engine        ┌───────────────┴────────────────────────────┐
                         │ Validation pipeline: splits → backtest →    │
                         │ falsification → WF → stress → Monte Carlo → │
                         │ acceptance rules → catalog status           │
                         ├─────────────────────────────────────────────┤
                         │ Backtest (portfolio + trade/event engines)  │
                         │ Cost model   Stats   Options   Monte Carlo  │
                         │ Signal combination   Portfolio construction │
                         └───────────────▲────────────────────────────┘
  independent risk       ┌───────────────┴──────────┐  ┌─────────────────┐
                         │ Risk engine + RiskGate   │  │ Paper trading /  │
                         │ (can veto any trade)     │──│ Broker interface │
                         └───────────────▲──────────┘  └─────────────────┘
  features               ┌───────────────┴────────────────────────────┐
                         │ Point-in-time feature library (each feature │
                         │ carries available_at; truncation-tested)    │
                         └───────────────▲────────────────────────────┘
  data                   ┌───────────────┴────────────────────────────┐
                         │ Providers (abstract) → validation → versioned│
                         │ Parquet/DuckDB store, PIT views, universes   │
                         └─────────────────────────────────────────────┘
```

Key architectural rules:

1. **Every datum carries `available_at`** — the timestamp at which the system could first have
   known it — and a provenance label `REAL | DERIVED | SIMULATED`. Derived data inherits the
   *latest* `available_at` of its inputs and the *weakest* label of its inputs (anything touched
   by simulated data is simulated).
2. **The research engine writes results; nothing else does.** The analyst/report/dashboard layers
   have read-only access to the registry.
3. **Risk is independent of signals.** The RiskGate receives a proposed trade plus portfolio state
   and returns APPROVE / REJECT with reasons. It does not import any strategy code.
4. **Vendors and brokers are behind interfaces** (`MarketDataProvider`, `OptionsDataProvider`,
   `FundamentalsProvider`, `Broker`), so swapping free for paid data changes configuration, not
   research code.
5. **The registry is append-only**, enforced by SQLite triggers that abort `UPDATE`/`DELETE`, and
   each row is hash-chained to the previous one so tampering is detectable.

## 2. Technology stack (and why)

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Scientific ecosystem; required by brief. |
| Data frames | pandas 3 + NumPy 2 | Ubiquitous, mature time-series tooling. Polars was considered; pandas is kept for ecosystem compatibility (statsmodels, sklearn). |
| Analytical store | Parquet files + DuckDB | Columnar, compressed, zero-server, SQL over Parquet, handles tens of GB on a laptop. |
| Research registry | SQLite (stdlib) | Transactional, supports triggers (DuckDB does not) → append-only enforcement. Single file, trivially backed up. |
| Stats | SciPy, statsmodels | HAC/Newey-West errors, regressions, distributions. Bootstrap/permutation/FDR/DSR implemented in-house (small, tested, auditable). |
| ML (later) | scikit-learn, LightGBM | Baselines first (logistic/linear), then trees. Deep learning only if justified. |
| Calendar | exchange_calendars (XNYS) | Correct sessions, holidays and **early closes** — needed to timestamp closing data correctly. |
| Config | pydantic-settings + env vars | Typed config, secrets only from environment, `.env.example` provided. |
| CLI | Typer | Scriptable, reproducible research runs. |
| Dashboard | FastAPI + Jinja2 + Plotly | Local, server-rendered, no JS build step; charts are interactive; read-only. Streamlit was considered but couples UI and compute. |
| Tests | pytest + hypothesis | Property-based tests suit invariants (accounting identities, no-look-ahead). |

## 3. Database architecture

Two stores with different integrity requirements:

**A. Market data store (`data/store/`, Parquet + DuckDB catalog)**

- Tables (logical): `bars_daily`, `bars_intraday`, `corporate_actions`, `earnings_events`,
  `earnings_estimates`, `option_quotes_eod`, `macro_series` (with vintages), `index_membership`,
  `classifications`, `short_interest`, `fundamentals`.
- Every table has: `event_time` (what period the value describes), `available_at` (when
  knowable), `ingested_at` (when we stored it), `source` (provider id), `label`
  (REAL/DERIVED/SIMULATED).
- **Raw prices are stored unadjusted**; adjustments are computed from the corporate-actions table
  *as known at the as-of time*. Vendor "adjusted close" series are retroactively revised every
  dividend and are therefore a leakage vector.
- **Datasets are versioned by content hash** (SHA-256 of canonical Parquet bytes + schema +
  provider metadata). Experiments record the dataset hash; a changed hash means a different
  dataset.
- Revisable series (macro, fundamentals) are stored as **vintages** (`available_at` per revision) so
  "value as known at t" is queryable.

**B. Research registry (`data/registry.sqlite`, append-only)**

- `hypotheses` (pre-registered specs, hashed, timestamped *before* evaluation)
- `experiments` (runs: code version, dataset hash, params, seeds, results JSON)
- `journal` (every research action)
- `signal_records` (catalog; new *versions* are appended, never updated)
- `status_events` (status transitions with reasons)
- `vault_access` (every access to untouched test data)
- `reports` (every generated report, stored verbatim)
- Each row stores `prev_hash` and `row_hash`; `verify_chain()` detects edits made outside the API.

## 4. UI / dashboard architecture

A local FastAPI app (`quantlab dashboard`) serving server-rendered pages with embedded Plotly charts,
bound to `127.0.0.1` by default. Pages: Portfolio, Strategies, Signals, Opportunities, Research,
Options, Risk, Monte Carlo, Reports, Assistant. All pages are **read-only views** of the registry and
data store; the dashboard cannot start experiments or change statuses (that stays in the CLI where
it is journaled). Every number shown links to its source record ID. Every chart is labeled with
the data provenance of its inputs (a red **SIMULATED** banner when applicable).

## 5. Data providers for the MVP (free)

| Data | Provider | Notes |
|---|---|---|
| Daily OHLCV (ETFs, large caps) | Stooq CSV; Yahoo (unofficial); Tiingo free tier (API key) | Free, but no delisted securities (Tiingo has some), occasional bad prints, vendor-adjusted fields revised retroactively. |
| VIX, VIX9D, VIX3M, VVIX, SKEW index | Cboe public CSVs; FRED `VIXCLS` | Real, free, daily. |
| Rates / Treasury | FRED (DGS3MO, DGS2, DGS10, DFF, T10Y2Y); **ALFRED** for vintages | Free API key. ALFRED gives true point-in-time revisions. |
| Earnings announcement timestamps | SEC EDGAR 8-K Item 2.02 acceptance timestamps | Free, and — importantly — timestamped to the second, so pre/post-market timing is knowable. |
| Fundamentals (PIT) | SEC EDGAR XBRL `companyfacts` with `filed` dates | Free and point-in-time by filing date; messy tagging. |
| Factors | Ken French data library | Free; for residualizing returns. |
| Short interest | FINRA bi-monthly short interest; RegSHO daily short volume | Free; publication lag must be modeled. |
| Classification | SEC SIC codes | Free proxy for sectors; GICS is proprietary. |

## 6. Providers that would materially improve research (mostly paid)

Prices are approximate as of the plan date and must be checked before buying.

| Need | Provider(s) | Why it matters |
|---|---|---|
| Survivorship-free US equities + historical index membership | **Norgate Data** (retail, low hundreds USD/yr); CRSP via WRDS (institutional) | Without delisted stocks and PIT membership, cross-sectional equity results are biased upward. **This is the single most valuable purchase for equity research.** |
| Historical option EOD bid/ask, IV, Greeks | **ORATS**, Cboe DataShop/LiveVol, historicaloptiondata.com, OptionMetrics (WRDS) | Required for any options/earnings-volatility strategy to be tested honestly. |
| Historical option NBBO intraday | ThetaData, Polygon/Massive (upper tiers), Databento (OPRA) | Needed to test execution near open/close and earnings timing. |
| Intraday equity bars/quotes | Polygon/Massive, Databento, FirstRate Data | Intraday mean reversion, open/close effects, realistic fills. |
| Earnings estimates, revisions, surprise | I/B/E/S (WRDS), Zacks, Estimize, FactSet | EPS surprise / revision signals; free data cannot support them honestly. |
| GICS sectors (PIT) | S&P / MSCI via vendors, Norgate (partial) | Sector-neutral construction. |
| Float / borrow cost | Vendor data (e.g., Ortex, S3, IBKR borrow feed) | Short selling feasibility and cost. |

## 7. Free vs paid limitations

- **Survivorship bias:** free equity data covers today's survivors → any cross-sectional stock study
  is biased. Mitigation in MVP: prefer **ETF universes** (sector SPDRs, broad index ETFs), which
  have far less survivorship issue, and flag all single-stock cross-sectional results
  `SURVIVORSHIP_RISK`.
- **Index membership:** no free, reliable PIT constituent history. Wikipedia change logs are
  error-prone. Membership-dependent studies are blocked until paid data.
- **Options:** no free historical bid/ask. Options *analytics* (pricing, Greeks, IV, payoff) work
  now; options *backtests* are blocked until paid data. Simulating option prices from a pricing
  model and then "discovering" option edges would be circular — the platform refuses to label
  such results as research findings.
- **Adjusted prices:** free vendors revise adjusted history; we store raw and recompute.
- **Earnings estimates/revisions:** unavailable free.
- **Intraday:** free intraday history is short (days to ~2 years) and often IEX-only (a small
  fraction of volume) → not representative for fills.
- **Data quality:** free feeds contain bad ticks, missing days and split errors; validation layer
  catches the common ones but cannot guarantee correctness.

## 8. Repository / folder structure

```
quantlab/
  pyproject.toml  .env.example  .gitignore
  README.md ARCHITECTURE.md DATA.md RESEARCH_METHODOLOGY.md RISK_MANAGEMENT.md PROJECT_STATE.md PLAN.md
  config/                      acceptance criteria, universes, cost profiles (TOML)
  src/quantlab/
    config.py provenance.py calendar.py
    data/        schemas, providers/, store, validation, pit, universe, corporate_actions
    features/    registry + library (all PIT, truncation-tested)
    research/    registry (append-only), hypotheses, generator, splits, vault, pipeline,
                 catalog, acceptance, experiments
    backtest/    costs, engine (portfolio), trades (event), metrics, benchmarks, event_study
    stats/       bootstrap, permutation, multiple_testing, sharpe (PSR/DSR), hac, bayes
    montecarlo/  resampling, capital simulation, ruin
    risk/        metrics (VaR/ES), sizing (fractional Kelly), gate, exposures
    options/     black_scholes, implied_vol, binomial (American), strategies, surface, events
    portfolio/   signal combination, construction
    analyst/     facts, reports, assistant (no free-form generation)
    dashboard/   FastAPI app + templates
    paper/       broker interface, paper broker, forward-test ledger
    cli.py
  tests/                       mirrors src layout
  data/                        (git-ignored) store/, registry.sqlite, reports/
```

## 9. Phase-by-phase roadmap

| Phase | Scope | Exit criterion |
|---|---|---|
| **1. Foundations** | Config, provenance, calendar/availability rules, data schemas, provider interface + synthetic/CSV/Stooq/FRED/Cboe adapters, validation, versioned store, PIT views, append-only registry, hypothesis pre-registration, splits + test vault, PIT feature library with automatic look-ahead test | All look-ahead, timestamp, registry-immutability and vault tests pass |
| **2. Backtesting & statistics** | Cost model, portfolio + trade engines (next-bar fills, liquidity caps, integer shares), metrics (gross vs net), benchmarks, event study, bootstrap, permutation, HAC, FDR/Holm/BY, PSR/DSR | Engine accounting identities hold; planted-edge and null-data calibration tests pass |
| **3. Validation pipeline** | Train/val/test, walk-forward, stress (cost ×2, parameter sensitivity, outlier removal, sub-periods), Monte Carlo, acceptance rules, signal catalog + status machine, first hypothesis batch | End-to-end run on simulated data: null signals rejected at ≈α, planted signals detected |
| **4. Options analytics** | BS, IV, Greeks, CRR American, parity, expected/event move, payoffs, multi-leg, surface/skew/term, option trade engine (bid/ask, multiplier, expiry, assignment) | Pricing identities and known values reproduced |
| **5. Risk & portfolio** | VaR/ES, exposures, fractional Kelly with caps, ruin estimates, RiskGate, signal combination with correlation shrinkage, portfolio construction | RiskGate vetoes verified; no double-counting of correlated signals in tests |
| **6. Analyst, reports, dashboard, assistant** | Fact layer, research & strategy reports (plain/quant modes), change-since-last-report, dashboard pages, NL assistant over stored records | Every rendered number traceable to a record ID |
| **7. Paper trading** | Broker ABC, paper broker, forward ledger, forward-vs-expected monitoring, decay detection | Live execution impossible without explicit config + safety review file |
| **8. Real data research** | Run the hypothesis program on real data (needs network access / data purchase) | Findings recorded, including failures |
| **9. Cross-sectional, stat-arb, ML** | Ranking, residual momentum, pairs/cointegration/PCA, ML models vs baselines | Complex models must beat baselines OOS after costs |

## 10. What can realistically be researched with free data

- Time-series trend/momentum and MA rules on **index and sector ETFs** (long histories, low
  survivorship bias).
- Short-horizon mean reversion on ETFs, conditional on trend and VIX regime (daily bars).
- Calendar effects (turn-of-month, day-of-week, pre-holiday, OPEX week, FOMC days).
- Overnight vs intraday decomposition (daily open/close suffices).
- Volatility risk premium *measurement* (VIX vs subsequent realized SPX vol) and VIX term-structure
  regimes (Cboe indices).
- Sector ETF cross-sectional momentum/reversal.
- Post-earnings drift using EDGAR announcement timestamps and announcement-window returns
  (large-cap survivors only → flagged).
- Macro regime conditioning (rates, curve) with ALFRED vintages.

## 11. What requires paid data

- Any **options strategy backtest** (earnings straddles/strangles, spreads, VRP harvesting, skew,
  surface relative value, put-call parity violations) → historical option bid/ask.
- Implied earnings move vs realized → option data.
- Intraday mean reversion, open/close auction effects, realistic intraday fills → intraday data.
- Credible single-stock cross-sectional factors → survivorship-free data + PIT membership.
- Earnings surprise/revisions/dispersion → estimates data.
- Short interest / float effects with daily granularity → vendor data.

## 12. First 10 hypotheses to test, and why

Chosen because they (a) are testable with free data, (b) have a published or plausible mechanism,
(c) span different alpha sources (trend, reversal, calendar, volatility), and (d) include the user's
own proposals so they can be **tested for falsification**, not assumed.

| # | Hypothesis | Why test it | Primary way it could be false |
|---|---|---|---|
| H1 | 12-1 month time-series momentum on liquid ETFs predicts next-month returns | Documented across asset classes (Moskowitz–Ooi–Pedersen 2012) — a benchmark "real" effect | Post-publication decay; driven by 2008 only |
| H2 | Holding SPY only when above its 200-day MA improves risk-adjusted net return vs buy-and-hold | User-proposed; popular claim | Lower drawdown but equal/lower Sharpe; whipsaw costs |
| H3 | 50/200 "golden/death cross" contains incremental information beyond H2 | User-proposed | Few trades → inconclusive; subsumed by H2 |
| H4 | Extreme negative 1–5 day standardized returns in index ETFs predict positive next 1–5 day returns, conditional on long-term uptrend | Well-known short-horizon reversal / liquidity provision | Regime-specific (post-2000 only), outliers (2020) |
| H5 | Bollinger-band position adds predictive information **incremental to** a plain z-scored return (nested test) | User asked whether BBs matter | Likely redundant: BB position ≈ z-score of price vs MA |
| H6 | SPY overnight returns exceed intraday returns on average | Documented "overnight drift" | Real but untradable after 2 round trips/day of cost — tests the executability filter |
| H7 | The volatility risk premium (VIX² − subsequent realized variance) is positive on average and fat-left-tailed | Foundational for any option-selling idea | Premium present but tail losses dominate geometric growth |
| H8 | VIX term structure inversion (VIX > VIX3M) predicts higher future realized vol and different forward SPY return distribution | Regime signal for sizing/risk, not necessarily alpha | Contemporaneous, not predictive; small sample of inversions |
| H9 | Cross-sectional 12-1 momentum among sector ETFs predicts relative returns | Sector momentum literature; survivorship-light universe | Only 9–11 assets → low breadth, weak power |
| H10 | Turn-of-the-month (last day to day +3) returns exceed other days on SPY | Documented calendar anomaly | Decayed post-publication; family of calendar tests must be corrected jointly |

The first paid-data hypothesis (queued, blocked): **long pre-earnings straddles lose money on
average after spreads because implied earnings moves exceed realized moves** (the user's
"buy options before earnings" idea, framed so it can be rejected).

## 13. Statistical acceptance / rejection criteria

Encoded in `config/acceptance.toml` and `research/acceptance.py`; frozen per hypothesis at
registration so criteria cannot be loosened after seeing results.

**Sample adequacy**
- Effective independent observations < 30 → `INCONCLUSIVE` (status stays EXPERIMENTAL, flagged).
- ≥ 100 effective observations required for ACCEPTED.
- Effective N accounts for overlapping holding periods and same-day clustering.

**Screening (EXPERIMENTAL → VALIDATING), training data only**
- Net mean return > 0 after conservative costs.
- HAC (Newey-West) one-sided p < 0.05 on training.

**Validation (VALIDATING → ACCEPTED)** — all must hold:
1. Validation net mean > 0 and validation Sharpe ≥ 0.5 × training Sharpe.
2. Walk-forward: ≥ 60% of folds net-positive; pooled WF OOS net Sharpe > 0.
3. Multiple testing: Benjamini–Yekutieli q < 0.10 over **all hypotheses ever registered** in the
   family, and Deflated Sharpe Ratio probability ≥ 0.95 given the number of trials.
4. Stationary block-bootstrap 95% CI for net mean excludes 0.
5. Cost stress: still net-positive at 2× costs.
6. Outliers: still net-positive after removing top 5% of trades.
7. Concentration: no single calendar year contributes > 40% of net P&L.
8. Parameter sensitivity: ≥ 70% of pre-declared neighboring parameter sets are net-positive.
9. Untouched test (opened once, after 1–8): same sign, net mean > 0; degradation vs validation
   reported. Failure → REJECTED (not retuned).

**Promotion to PAPER_TRADING** requires ACCEPTED + risk review + manual approval (journaled).

**LIVE_ELIGIBLE** requires ≥ 60 trading days and ≥ 30 paper trades, forward mean within the
historical 90% predictive interval (not below the 5th percentile), realized slippage ≤ 1.5× modeled,
and RiskGate compliance. Real-money execution additionally requires the config flag + signed safety
review.

**DEGRADED / RETIRED**: monitoring via rolling performance vs predictive distribution; persistent
breach → DEGRADED; not recovered within review window → RETIRED.

## 14. Preventing look-ahead bias and test-set contamination

- `available_at` on every row, computed from **exchange calendar session close + publication delay**
  (early closes handled), vendor-specific lags for fundamentals/short interest, filing acceptance
  timestamps for EDGAR.
- `PointInTimeFrame.as_of(t)` filters by `available_at ≤ t`; engines only see as-of views.
- **Automatic truncation test** for every registered feature: compute on full history vs history
  truncated at t; values at t must match. A feature that fails cannot be registered.
- Execution is **strictly after** decision: default fill at next session open (or next close);
  fill must lie in that bar's [low, high]; no fill on missing/zero-volume bars.
- Raw unadjusted prices + as-of corporate actions (no revised adjusted closes).
- PIT universes (membership with start/end dates); survivorship flag when unavailable.
- **Test vault:** data after the holdout start date is stripped from default research loaders.
  Access requires `vault.unseal(hypothesis_id, reason)`, which is permitted **once per hypothesis**
  and only after the hypothesis is `FROZEN` (spec + parameters + criteria hashed). Every access
  is logged in the append-only registry; re-access is recorded as contamination and reported.
- Purge + embargo gaps between train/validation/test and walk-forward folds equal to the label
  horizon.
- Label/target construction is separated from features in code (features cannot import labels).

## 15. Automated hypothesis generation without data-mining

- **Grammar, not search:** hypotheses are generated from a small set of templates
  (e.g., "feature F in top/bottom quantile predicts H-day forward return on universe U in regime R"),
  each tied to a declared **mechanism class** (risk premium, behavioral, liquidity, structural).
- **Budget:** each generation batch has a hard cap; the global count of tests is tracked.
- **Pre-registration:** each generated hypothesis is hashed and written to the registry *before*
  any evaluation code touches it. Results for unregistered specs are refused.
- **Family-wise accounting:** p-values are corrected over the whole family including all past
  failures (BY for arbitrary dependence; Romano–Wolf style permutation max-stat where feasible).
- **Generated ≠ confirmed:** auto-generated survivors are tagged `EXPLORATORY`; they must pass the
  untouched test and walk-forward as a separate confirmatory step.
- **Everything is reported:** reports show counts of all tested hypotheses, not just survivors.

## 16. How the Research Analyst avoids hallucination

- A **fact layer** (`analyst/facts.py`) runs typed queries against the registry and returns
  `Fact(value, source_table, record_id, row_hash)` objects.
- Reports and assistant answers are rendered by deterministic templates from facts only. Each
  number is emitted with its record ID.
- The assistant maps questions to query intents (rule-based); unknown intents return "no analysis
  has been performed on this" plus the list of what exists.
- If an LLM is added later, it may only rephrase a retrieved fact bundle, and a validator rejects
  any output containing numbers or IDs that do not appear in the bundle.
- The analyst layer has no write access to results; it can only append `reports`.

## 17. Computational requirements

- **MVP (free daily data):** ~500 symbols × 25 years ≈ 3M rows (<1 GB Parquet). Any modern laptop
  (4+ cores, 8–16 GB RAM). Bootstrap/permutation tests (10k resamples × hundreds of hypotheses)
  run in minutes on 4 cores.
- **Options EOD (paid):** full US chains ≈ 1–3M contract-rows/day → hundreds of GB over 15 years.
  Needs 32–64 GB RAM, 1–2 TB NVMe, partitioned Parquet by date/underlying, DuckDB queries.
- **Intraday quotes:** TB scale; process per-symbol-day streams; consider a workstation with 64 GB+
  and multi-TB storage, or restrict to a focused universe.
- **ML:** gradient boosting on CPU is sufficient; GPU only if deep learning is ever justified.

## 18. Flaws and tensions in the specification

1. **$100–$1,000 capital vs. the strategy set.** One option contract commonly costs more than the
   whole account; diversification across many signals is impossible; per-trade commissions and
   spreads dominate; short selling, pairs trading and most spreads require a margin account
   (brokers typically require ≥ $2,000 for margin), and pattern-day-trading rules may restrict
   frequent trading (check current FINRA rules). Many researched edges will be **real but not
   executable at that size** — the capacity analysis will say so explicitly.
2. **Free data cannot answer many requested questions** (options, estimates, index membership,
   intraday fills). Interfaces are built; findings are blocked until data exists.
3. **"Completely untouched" test data erodes over a long program.** Each hypothesis that opens the
   vault uses it up a little for the *program* as a whole. Paper trading on genuinely future data
   is the only truly clean out-of-sample test; the vault access count is reported.
4. **Regime analysis has tiny samples.** Since 1993 there are only a handful of bear markets and
   recessions; regime-conditional statistics will often be inconclusive, and the reports must say
   so rather than slicing until something looks significant.
5. **Regime labels can leak.** NBER recession dates are announced months later. Ex-post regimes are
   allowed for *description*; trading rules may only use regimes computable in real time.
6. **Earnings samples are clustered** in four seasons per year; effective N is far smaller than the
   event count.
7. **Monte Carlo from historical resampling cannot produce crises that have not happened.** It is a
   lower bound on tail risk; results are presented with that caveat and with stressed variants.
8. **FDR procedures assume dependence structures** that correlated hypotheses violate; BY and
   permutation max-statistics are used for that reason, at the cost of power.
9. **Taxes and wash-sale rules are absent from the brief** but materially affect net geometric
   growth for high-turnover strategies in taxable accounts. Planned as an optional cost layer.
10. **Paper trading needs a real-time data feed**, which is not free at quality; delayed free feeds
    distort fill simulation.
11. **"Probability of ruin"** needs a definition; the default here is equity falling below a
    configurable fraction (default 20%) of starting capital, or below the minimum capital needed to
    place the strategy's smallest trade.
