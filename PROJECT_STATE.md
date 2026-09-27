# PROJECT_STATE.md

_Last updated: 2026-09-27 — end of Phase 1 (foundations)._

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

Tests: `cd quantlab && .venv/bin/python -m pytest` → 73 passed.

## What is simulated

Everything that produces numbers. The synthetic market is the only data source exercised.

## What uses real data

Nothing yet.

## Planned (not built)

Phase 2: cost model, backtest engines, metrics, statistics. Phase 3: validation pipeline, signal
catalog, Monte Carlo. Phase 4: options analytics. Phase 5: risk & portfolio. Phase 6: analyst,
reports, dashboard, assistant. Phase 7: paper trading. See PLAN.md §9.

## Current data providers

Configured: synthetic only. Adapters ready (untested live): Stooq, Tiingo (key), FRED/ALFRED (key), Cboe.

## Known bugs / limitations

- Free-provider adapters have never hit the live services.
- `turn_of_month` uses today's exchange calendar; unscheduled historical closures (e.g. 2012
  Hurricane Sandy) were not known in advance — negligible, documented.
- Corporate-action announcement times from Tiingo are unknown; set conservatively to the ex-date open.
- The vault is a guard rail (default path is clean; deviations are logged), not a cryptographic barrier.

## Next development task

Phase 2: `backtest/costs.py`, `backtest/engine.py` (portfolio), `backtest/trades.py` (event
trades), `backtest/metrics.py`, `stats/*` with calibration tests (null data rejected at ≈α; planted
effect detected).
