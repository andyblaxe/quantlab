# PROJECT_STATE.md

_Last updated: 2026-09-27 — end of Phase 3 (validation pipeline). Phase 6 (analyst) in progress._

This file is the hand-off document. A new session should be able to continue from here without
the conversation history. Status vocabulary: **IMPLEMENTED** (code exists) · **TESTED** (automated
tests cover it) · **SIMULATED** (runs only on synthetic data so far) · **PLANNED** (not built).

## Research findings

**None.** No real market data has been ingested. The build environment's network policy blocks
the free data hosts (query1.finance.yahoo.com, stooq.com, fred.stlouisfed.org); every number
produced so far comes from the SIMULATED synthetic market and is a machinery check, not a finding.

## What is built

| Component | Module | Status |
|---|---|---|
| Config from env vars, `.env.example` | `config.py` | IMPLEMENTED, TESTED |
| Provenance labels/flags, dataset hashing | `provenance.py` | IMPLEMENTED, TESTED |
| NYSE calendar + bar availability timestamps (early closes) | `calendar.py` | IMPLEMENTED, TESTED |
| Canonical schemas for 11 tables | `data/schemas.py` | IMPLEMENTED (bars/actions/earnings/macro exercised) |
| Provider interface | `data/providers/base.py` | IMPLEMENTED, TESTED |
| Synthetic market (SIMULATED) | `data/providers/synthetic.py` | IMPLEMENTED, TESTED |
| Stooq / Tiingo / FRED-ALFRED / Cboe adapters | `data/providers/free.py` | Parsers TESTED on fixtures; **live fetch never run** |
| Local CSV loader (for purchased data) | `data/providers/local_csv.py` | IMPLEMENTED (light tests pending) |
| Data validation | `data/validation.py` | IMPLEMENTED, TESTED |
| Write-once versioned Parquet store + DuckDB | `data/store.py` | IMPLEMENTED, TESTED |
| PIT adjustment (forward TRI) and research panel | `data/adjust.py` | IMPLEMENTED, TESTED (matches simulator truth to 1e-10) |
| PIT frames / as-of joins for vintaged series | `data/pit.py` | IMPLEMENTED, TESTED |
| PIT universes (membership, static + flags) | `data/universe.py` | IMPLEMENTED, TESTED |
| Append-only, hash-chained research registry | `research/registry.py` | IMPLEMENTED, TESTED |
| Hypothesis pre-registration + lifecycle | `research/hypotheses.py` | IMPLEMENTED, TESTED |
| Acceptance criteria (frozen into spec hash) | `research/acceptance.py` | IMPLEMENTED (evaluation logic: Phase 3) |
| Split plan, embargo, purging, walk-forward folds | `research/splits.py` | IMPLEMENTED, TESTED |
| Test vault (FROZEN-only, one unseal, contamination log) | `research/vault.py` | IMPLEMENTED, TESTED |
| Forward-return labels (isolated module) | `research/labels.py` | IMPLEMENTED, TESTED |
| Feature library (24 families) + automatic truncation test | `features/` | IMPLEMENTED, TESTED |
| Cost model (spread, slippage, sqrt impact, commissions, borrow; profiles) | `backtest/costs.py` | IMPLEMENTED, TESTED |
| Portfolio engine (raw-price share accounting, ex-date actions, next-bar fills, liquidity caps, integer shares, gross vs net runs) | `backtest/engine.py` | IMPLEMENTED, TESTED |
| Event-trade engine (feasibility rules, per-trade gross/net) | `backtest/trades.py` | IMPLEMENTED, TESTED |
| Metrics, benchmarks, event study | `backtest/` | IMPLEMENTED, TESTED (event study: light) |
| Statistics: HAC, block bootstrap, sign-flip, circular-shift, random-entry, effective N, BH/BY/Holm, PSR/DSR, IC, Bayesian shrinkage | `stats/` | IMPLEMENTED, TESTED incl. size/power calibration |
| Strategy templates (threshold/quantile events, state positions, overnight/intraday segments) | `research/strategies.py` | IMPLEMENTED, TESTED |
| Validation pipeline (screening, validation, walk-forward, stress, BY, DSR, regimes, stability, capacity, MC, one-shot test) | `research/pipeline.py` | IMPLEMENTED, TESTED (SIMULATED) |
| Measurement hypotheses (VRP, conditional forward vol/returns) | `research/measurements.py` | IMPLEMENTED, TESTED |
| Signal catalog + enforced status machine | `research/catalog.py` | IMPLEMENTED, TESTED |
| Monte Carlo capital simulation ($100–$100k; fixed costs, minimum size, ruin, stressed variant) | `montecarlo/simulate.py` | IMPLEMENTED, TESTED |
| Research program v1 (10 pre-registered hypotheses) | `research/program.py` | IMPLEMENTED; **not yet run on real data** |
| Budgeted hypothesis generator | `research/generator.py` | IMPLEMENTED, TESTED |
| Pipeline calibration | `research/calibration.py`, `quantlab calibrate` | IMPLEMENTED; results below |
| CLI (`quantlab ...`) | `cli.py` | IMPLEMENTED (data fetch commands untested live) |

Tests: `cd quantlab && .venv/bin/python -m pytest` → 121 passed.

### Machinery calibration (SIMULATED, 2026-09-27)

`quantlab calibrate --n-null 20 --n-power 5`, 1-day reversal hypothesis, retail ETF costs:
null markets → 0/20 reached FROZEN or ACCEPTED (0/20 is consistent with a true rate up to ~14%;
more runs needed for a tight bound; costs make this null relatively easy to reject). Planted
reversal (idiosyncratic AR(1) = −0.45) → 5/5 detected and accepted after the untouched test. A
weaker planted effect (−0.15) is real but smaller than costs and is correctly rejected; −0.3 failed
screening in one run because a stressed regime inside TRAIN hurt it — the pipeline errs toward false
negatives.

### Dry run of program v1 on SIMULATED stand-in data (no planted effects)

All eight trading hypotheses rejected at screening; VRP measurement "SUPPORTED" (true by construction
— the synthetic VIX embeds a premium); VIX term-structure measurement INCONCLUSIVE (no VIX3M series).
This validates plumbing only.

## What is simulated

Everything that produces numbers. The synthetic market is the only data source exercised.

## What uses real data

Nothing yet.

## Planned (not built)

Phase 4: options analytics. Phase 5: risk engine & portfolio construction. Phase 6: analyst,
reports, dashboard, assistant (fact layer started). Phase 7: paper trading. Phase 8: real-data run.
Phase 9: cross-sectional/stat-arb/ML. See PLAN.md §9.

## Current data providers

Configured: synthetic only. Adapters ready (untested live): Stooq, Tiingo (key), FRED/ALFRED (key), Cboe.

## Known bugs / limitations

- Free-provider adapters have never hit the live services.
- `turn_of_month` uses today's exchange calendar; unscheduled historical closures (e.g. 2012
  Hurricane Sandy) were not known in advance — negligible, documented.
- Corporate-action announcement times from Tiingo are unknown; set conservatively to the ex-date open.
- The vault is a guard rail (default path is clean; deviations are logged), not a cryptographic barrier.

## Next development task

Phase 6: finish `analyst/` (research report, strategy report, assistant), then dashboard.
