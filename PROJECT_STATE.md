# PROJECT_STATE.md

_Last updated: 2026-09-27, second session (Phase 8 started on this Mac: real data ingested, program v1
evaluated at development stage; EDGAR adapter built; survivorship audit done — PEAD paused pending
the owner's data decision)._

**Permanent rule (RESEARCH_METHODOLOGY.md §1a):** nothing with unresolved material research-integrity
problems — survivorship bias above all — may be promoted to ACCEPTED, PAPER_TRADING or LIVE_ELIGIBLE.
It is enforced in code (`research/integrity.py`); do not remove, weaken or bypass it. The principles
are permanent; the numeric thresholds are pre-registered per hypothesis and cannot be loosened after
results are seen.

This is the hand-off document. A new session should be able to continue from here without the
conversation history. Vocabulary: **IMPLEMENTED** (code exists) · **TESTED** (automated tests cover
it) · **SIMULATED** (only exercised on synthetic data) · **PLANNED** (not built).

## 1. Research findings

Program v1 (10 pre-registered hypotheses, `research/program.py`, PLAN.md §12) was registered
(H-000001…H-000010) and evaluated at **development stage only** on REAL data (Tiingo bars + Cboe VIX,
see §7) on 2026-09-27. The untouched test partition (after 2018 + embargo) was **not** opened for any
hypothesis.

| ID | Hypothesis | Result | Where it stopped |
|---|---|---|---|
| H-000001 | Time-series momentum (multi-asset ETFs) | REJECTED | screening: train p-value |
| H-000002 | SPY 200-day MA filter | REJECTED | screening: train p-value |
| H-000003 | SPY golden/death cross | REJECTED | screening: train p-value |
| H-000004 | Short-term reversal in index ETFs within an uptrend | REJECTED | passed screening; failed walk-forward and BY multiple testing |
| H-000005 | Bollinger bands beyond z-scored returns | REJECTED | screening: net mean ≤ 0, p-value |
| H-000006 | SPY overnight drift after costs | REJECTED | screening: net mean ≤ 0, p-value |
| H-000007 | Volatility risk premium (measurement) | SUPPORTED | estimate 0.01144, BY q = 0.00036, same sign over the development period |
| H-000008 | VIX term-structure inversion → higher realised vol (measurement) | INCONCLUSIVE | only 20 independent observations (VIX3M starts ~2009) |
| H-000009 | Sector ETF cross-sectional momentum | REJECTED | screening: net mean ≤ 0, p-value |
| H-000010 | Turn-of-the-month in SPY | REJECTED | screening: train p-value |

Reading: no trading hypothesis in v1 shows an after-cost edge on these ETFs. H-000007 is a measurement;
a positive average premium is **not** evidence that selling options is profitable (needs option bid/ask
data). Caveats: 20-symbol static universe of today's ETFs (SURVIVORSHIP_RISK flag).

## 2. What is built

| Component | Module | Status |
|---|---|---|
| Config from env vars, `.env.example` | `config.py` | IMPLEMENTED, TESTED |
| Provenance labels/flags, dataset hashing | `provenance.py` | IMPLEMENTED, TESTED |
| NYSE calendar, bar availability timestamps (early closes) | `calendar.py` | IMPLEMENTED, TESTED |
| Canonical schemas (11 tables) | `data/schemas.py` | IMPLEMENTED (bars/actions/earnings/macro/options exercised) |
| Provider interface | `data/providers/base.py` | IMPLEMENTED, TESTED |
| Synthetic market (SIMULATED; GARCH, regimes, splits, dividends, earnings, VIX-like; planted effects) | `data/providers/synthetic.py` | IMPLEMENTED, TESTED |
| Stooq / Tiingo / FRED-ALFRED / Cboe adapters | `data/providers/free.py` | parsers TESTED on fixtures; **live fetch never run** |
| Local CSV loader (for purchased data) | `data/providers/local_csv.py` | IMPLEMENTED, untested |
| Validation | `data/validation.py` | IMPLEMENTED, TESTED |
| Write-once versioned store + DuckDB | `data/store.py` | IMPLEMENTED, TESTED |
| PIT forward total-return adjustment | `data/adjust.py` | IMPLEMENTED, TESTED (matches simulator truth to 1e-10) |
| PIT frames / as-of joins (vintages) | `data/pit.py` | IMPLEMENTED, TESTED |
| PIT universes | `data/universe.py` | IMPLEMENTED, TESTED |
| Feature library (24 families) + automatic truncation test | `features/` | IMPLEMENTED, TESTED |
| Cost model, portfolio engine, event-trade engine, metrics, benchmarks | `backtest/` | IMPLEMENTED, TESTED |
| Event study | `backtest/event_study.py` | IMPLEMENTED, lightly tested; no earnings data source yet |
| Statistics (HAC, bootstrap, permutation, random-entry, effective N, BH/BY/Holm, PSR/DSR, IC, Bayes) | `stats/` | IMPLEMENTED, TESTED incl. size/power calibration |
| Append-only hash-chained registry (thread-safe) | `research/registry.py` | IMPLEMENTED, TESTED |
| Hypothesis pre-registration + lifecycle; acceptance criteria frozen into hash | `research/hypotheses.py`, `acceptance.py` | IMPLEMENTED, TESTED |
| Split plan, embargo, purging, walk-forward; test vault | `research/splits.py`, `vault.py` | IMPLEMENTED, TESTED |
| Strategy templates (threshold/quantile events, state positions, overnight/intraday segments) | `research/strategies.py` | IMPLEMENTED, TESTED |
| Validation pipeline + one-shot untouched test | `research/pipeline.py` | IMPLEMENTED, TESTED (SIMULATED) |
| Measurement hypotheses (VRP, conditional forward) | `research/measurements.py` | IMPLEMENTED, TESTED |
| Signal catalog + enforced status machine | `research/catalog.py` | IMPLEMENTED, TESTED |
| Research program v1; budgeted generator; calibration | `research/program.py`, `generator.py`, `calibration.py` | IMPLEMENTED, TESTED |
| Relative-value toolkit (rolling corr, Engle–Granger, PIT spreads/z, half-life, PCA residuals, clustering) | `research/relative_value.py` | IMPLEMENTED, TESTED — not yet wired into hypothesis templates |
| ML benchmarks (walk-forward logistic / gradient boosting, OOS evaluation vs baselines, calibration, permutation importance, PDP) | `research/models.py` | IMPLEMENTED, TESTED — not yet wired into the pipeline |
| Monte Carlo ($100–$100k; fixed costs, min size, ruin, stressed) | `montecarlo/simulate.py` | IMPLEMENTED, TESTED |
| Options: BSM, Greeks, IV (no-arb bounds), CRR American, parity (bid/ask), expected & event moves, RND, payoffs, multi-leg, chain analytics, conservative option trade simulator | `options/` | IMPLEMENTED, TESTED on model/simulated chains; **no historical option data** |
| Risk: VaR/ES/CF, exposures, fractional Kelly on lower-bound edge, ruin, RiskGate | `risk/` | IMPLEMENTED, TESTED |
| Signal combination (no double counting) + fractional-Kelly portfolio | `portfolio/combine.py` | IMPLEMENTED, TESTED |
| Current opportunities (NO TRADE explicit) | `research/opportunities.py` | IMPLEMENTED, exercised via dashboard (SIMULATED) |
| Research Analyst: fact layer, research & 14-section strategy reports (plain/quant), change-since-last, assistant | `analyst/` | IMPLEMENTED, TESTED |
| Dashboard (11 pages, local, read-only except report generation) | `dashboard/app.py` | IMPLEMENTED, smoke-TESTED, screenshots checked light/dark |
| Paper broker, forward ledger, forward-vs-history, CUSUM degradation, LIVE_ELIGIBLE check, live lock | `paper/` | IMPLEMENTED, TESTED; no real-time quote feed |
| CLI | `cli.py` | IMPLEMENTED; `data fetch`, `fetch-macro`, `fetch-earnings` run live |
| EDGAR 8-K Item 2.02 earnings timestamps (release matched by usual reporting lag) | `data/providers/edgar.py` | IMPLEMENTED, TESTED on fixtures, run live (AAPL/MSFT/JPM/BRK-B) |
| Security master: permanent ids, dated tickers/names/exchanges, lifecycle events, listed/tradable masks | `data/security_master.py` | IMPLEMENTED, TESTED; not populated (no source) |
| **Research-integrity promotion gate (PERMANENT RULE)**: permanent principles in code; numeric thresholds pre-registered per hypothesis (`AcceptanceCriteria` v2026-09-27.2; looser-than-default needs a registered justification); INTEGRITY_FAILURE vs INSUFFICIENT_EVIDENCE; automatic + manual findings; union over record history; no override; enforced in catalog, pipeline (vault withheld), paper trading and live eligibility | `research/integrity.py`, `acceptance.py`, `catalog.py`, `pipeline.py`, `paper/forward.py` | IMPLEMENTED, TESTED (`tests/test_integrity.py`) |

Tests: `.venv/bin/python -m pytest` (from the repo root) → 218 passed (~3 min; the pipeline tests dominate).

Environment (this Mac): Python 3.12 venv at `.venv`, created with uv (`~/.local/bin/uv venv --python 3.12`,
then `uv pip install -e '.[dev,ml]'`). Resolved to **pandas 3.0.6**; the full suite passes on it.

## 3. Machinery calibration and demo (SIMULATED)

* `quantlab calibrate` (after the DSR fix): null markets 0/10 accepted; planted 1-day reversal
  (AR(1) = −0.45) 3/3 detected and accepted after the untouched test. An earlier run (0/20 nulls,
  5/5 power) predates the DSR fix but nulls failed at screening there too. 0/N bounds the
  false-positive rate only loosely (0/20 ⇒ ≤ ~14% at 95%); run more for a tight bound.
* `quantlab demo` (synthetic market with a planted 1-day idiosyncratic reversal): the two generated
  hypotheses matching the planted mechanism (1-day z-score, 1-day hold) were ACCEPTED; the 10
  mismatched variants and all eight program-v1 trading hypotheses were REJECTED; VRP measurement
  SUPPORTED (true by construction); VIX term structure INCONCLUSIVE (no VIX3M in the demo).
  Note: this synthetic path (seed 42) fell ~70% over 26 years, so "market drawdown > 20%" dominates
  its regime table — a property of that path, not a bug.
* A weaker planted effect (−0.15) is real but below costs and is correctly rejected; −0.3 failed
  screening once because a stressed regime in TRAIN hurt it. The pipeline errs toward false negatives.

## 4. What is simulated / what uses real data

Simulated: §3 (calibration, demo). Real data: program v1 development-stage results in §1.

## 5. Bugs found and fixed this session (recorded for auditability)

1. **Deflated Sharpe benchmark inflated** by heterogeneous, cost-crushed trials (empirical cross-trial
   Sharpe dispersion). Fixed: criterion uses the null sampling dispersion; empirical version kept as a
   diagnostic. Regression test added. The pre-fix demo registry was archived, not deleted
   (`data/demo_archive_dsr_bug_20260927`, local only).
2. Option trade simulator dropped trades with a missing exit quote (survivorship). Fixed: exit is
   delayed (recorded) or settled at expiry.
3. IVs from min-tick and deep-ITM quotes were trusted. Fixed: `iv_reliable` flag (bid > 0, spread <
   25%, OTM/near-ATM); surfaces use reliable quotes only.
4. Segment (overnight) strategy too slow (Python loop); vectorised.
5. Registry not usable from the dashboard's thread pool; now thread-safe.
6. **`src/quantlab/data/` was never committed** (first session): the unanchored `.gitignore` rule
   `data/` matched it, so fresh clones could not import. Recovered from
   `credit-research-lab@recover-data-package`; rule anchored to `/data/` (commit e2cc99d).
7. Tiingo adapter: symbols without corporate actions returned untyped empty frames ⇒ concatenated
   `ex_date`/`value` became object and failed validation. Fixed; also one request per symbol (5bb4ba6).
8. Store integrity check failed on data it had just saved under pandas 3 (object string columns reload
   as `str`, changing the content hash). The store now hashes the frame as Parquet returns it; regression
   test added (5bb4ba6). The pre-fix store was archived at `data/store_archive_hash_bug_20260927`.

## 6. Known limitations / open issues

- Stooq is blocked by a bot challenge (use Tiingo). FRED adapter has a working key but no v1 series use it.
- Cboe VIX3M history starts ~2009, which starves H-000008 (INCONCLUSIVE).
- **Survivorship (BLOCKING for single stocks):** Tiingo has almost no pre-2013 delisted prices and no
  index-membership history (DATA_GAPS.md G1). Current-members stock universes are enforced as
  PRELIMINARY / SURVIVORSHIP-BIASED: no promotion, untouched test withheld. Provider evaluation and
  recommendation in DATA_PROVIDERS.md. **Do not buy data without the owner's approval.**
- Security master (`data/security_master.py`): schemas/class/tests only; no source fills it yet.
- EDGAR adapter maps tickers via SEC's current map (G3); switch to CIK from the security master.
- No historical options data ⇒ option strategies cannot be backtested (analytics only). Model-priced
  chains are labelled SIMULATED and must never be used as evidence of option mispricing.
- Relative-value and ML modules are tools; no hypothesis templates use them yet.
- Event-study bootstrap treats events as independent (clustered bootstrap planned).
- Effective-N is a calendar-block heuristic.
- `turn_of_month` uses today's calendar (unscheduled historical closures were not known in advance).
- Tiingo corporate-action announcement times unknown (conservative: ex-date open).
- VIX series are timestamped close+30 min while bars are close+15 min, so VIX used at a session's
  decision time is the previous day's value — conservative, documented.
- Position-strategy "trades" (holding spells) use a spread+slippage per-side cost approximation;
  daily returns come from the full engine.
- Pipeline test runtime (~3 min) — consider a `slow` marker.
- SHAP not implemented (permutation importance and PDP are).
- Taxes / wash sales not modelled.
- The vault is a guard rail with an audit trail, not a cryptographic barrier.

## 7. Current data providers

Working live (2026-09-27): Tiingo daily bars + dividends/splits (`TIINGO_API_KEY`; free tier, keep
requests low), Cboe VIX/VIX3M (no key), FRED/ALFRED key verified (`FRED_API_KEY`), EDGAR user agent set
(`EDGAR_USER_AGENT`). Stooq: blocked by a bot challenge. Synthetic: SIMULATED.

Stored (local `data/store`): `bars_daily` 1c54f7fe8a66 (20 ETFs, 132,677 rows, from each ETF's launch to
2026-09-25; 1 warning: 3 zero-volume bars), `corporate_actions` b09f31b09981 (2,421 rows, 10 splits),
`macro_series` VIX 6479851c7c53 (9,247 rows), VIX3M 236d37344f90 (4,281 rows). Splits verified (no
spurious −50% returns); all |daily return| > 15% fall on known crisis dates. Paid options/intraday/
estimates/PIT-membership providers: interface only.

## 8. Next development tasks (in priority order)

1. ~~Get real data~~ — done 2026-09-27: `quantlab init`, `data fetch --provider tiingo`,
   `data fetch-macro`, `research register-program --name v1`, `research evaluate --all-registered`.
   Nothing in v1 reached FROZEN, so no untouched test is pending.
2. **Owner decision pending:** survivorship-free data source (DATA_PROVIDERS.md recommends Sharadar,
   subject to a quote; Norgate Platinum US$630/yr as alternative). Then write the vendor adapter
   (`get_security_master`, bars/actions/membership keyed by security_id) and re-point EDGAR to CIKs.
   PEAD (next item) waits for this; a current-members run could only ever be PRELIMINARY.
3. Built (adapter done): EDGAR 8-K Item 2.02 adapter (earnings announcement timestamps) + PEAD hypothesis via event study.
4. Hypothesis templates for pairs/spreads (using `relative_value.py`) and ML-model hypotheses (using
   `models.py`) so they flow through the same pipeline, multiple-testing and vault.
5. Clustered (by date) bootstrap for event studies; `slow` test marker.
6. Historical options data adapter (ORATS / Cboe DataShop) → earnings straddle hypothesis
   (framed for rejection: implied moves > realised after spreads).
