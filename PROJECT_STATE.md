# PROJECT_STATE.md

_Last updated: 2026-09-27, end of the first build session (Phases 1–7 built; Phase 8 blocked on data)._

This is the hand-off document. A new session should be able to continue from here without the
conversation history. Vocabulary: **IMPLEMENTED** (code exists) · **TESTED** (automated tests cover
it) · **SIMULATED** (only exercised on synthetic data) · **PLANNED** (not built).

## 1. Research findings

**No research findings on real markets exist.** No real market data has been ingested. The build
environment's network policy blocks the free data hosts (query1.finance.yahoo.com, stooq.com,
fred.stlouisfed.org), so every number produced so far comes from SIMULATED synthetic markets. Those
numbers validate the machinery; they are not evidence about markets.

The pre-registered research program v1 (10 hypotheses, `research/program.py`, PLAN.md §12) is
committed to git and has **not** been run on real data.

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
| CLI | `cli.py` | IMPLEMENTED; data-fetch commands untested live |

Tests: `cd quantlab && .venv/bin/python -m pytest` → 178 passed (~4 min; the pipeline tests dominate).

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

Simulated: everything that produced numbers. Real data: nothing yet.

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

## 6. Known limitations / open issues

- Free-provider adapters have never hit live services (network blocked here).
- No earnings-date / EDGAR adapter yet ⇒ earnings strategies cannot be researched yet.
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

Configured and working: synthetic (SIMULATED). Ready but untested live: Stooq (no key), Tiingo
(`TIINGO_API_KEY`), FRED/ALFRED (`FRED_API_KEY`), Cboe indices (no key). Paid options/intraday/
estimates/PIT-membership providers: interface only.

## 8. Next development tasks (in priority order)

1. **Get real data** — run on a machine with internet (or allow the hosts in this environment's
   network settings): `quantlab init`, `data fetch --provider tiingo`, `data fetch-macro`, then
   `research register-program v1` and `research evaluate --all-registered`. Inspect validation
   warnings first. This is Phase 8 and produces the first real findings.
2. EDGAR 8-K Item 2.02 adapter (earnings announcement timestamps) + PEAD hypothesis via event study.
3. Hypothesis templates for pairs/spreads (using `relative_value.py`) and ML-model hypotheses (using
   `models.py`) so they flow through the same pipeline, multiple-testing and vault.
4. Clustered (by date) bootstrap for event studies; `slow` test marker.
5. Historical options data adapter (ORATS / Cboe DataShop) → earnings straddle hypothesis
   (framed for rejection: implied moves > realised after spreads).
