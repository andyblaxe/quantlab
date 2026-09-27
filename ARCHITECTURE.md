# ARCHITECTURE.md

The layered design and its rationale are in PLAN.md §1–4. This file is the **current module map**,
the rules the code enforces, and how to extend it.

## Data flow

```
providers (REAL / SIMULATED)          data/providers/*.py      → Dataset(frame, provenance)
   │  validate (errors block, warnings recorded)   data/validation.py
   ▼
write-once versioned store (Parquet + manifest)    data/store.py
   ▼
research panel: forward total-return adjustment    data/adjust.py  (PIT-safe)
   + PIT macro series (asof on available_at)       data/pit.py, research/data.py
   ▼
features (trailing, truncation-tested)             features/
   ▼                       labels (future) live only in research/labels.py
strategy templates → trades / daily returns        research/strategies.py, backtest/{engine,trades,costs}.py
   ▼
pipeline: screening → validation → walk-forward → stress → BY/DSR → regimes/stability/capacity/MC
                                                   research/pipeline.py, stats/, montecarlo/
   ▼   (FROZEN only) vault → untouched test        research/vault.py
append-only registry (hypotheses, experiments, journal, catalog, status, vault log, reports, paper)
                                                   research/registry.py, research/catalog.py
   ▼ read-only
analyst facts → reports / assistant / dashboard    analyst/, dashboard/
   ▲
risk engine (independent) ← proposed trades ← opportunities / paper trading   risk/, paper/, research/opportunities.py
```

## Module map

| Package | Modules | Responsibility |
|---|---|---|
| `quantlab` | `config`, `provenance`, `calendar`, `cli` | env config; REAL/DERIVED/SIMULATED labels + flags + hashes; XNYS sessions and availability; command line |
| `data` | `schemas`, `providers/{base,synthetic,free,local_csv}`, `validation`, `store`, `adjust`, `pit`, `universe` | ingestion, validation, versioning, point-in-time access |
| `features` | `base`, `library` | 24 feature families; registry; truncation (look-ahead) test |
| `backtest` | `costs`, `engine`, `trades`, `metrics`, `benchmarks`, `event_study` | execution realism, gross vs net, performance statistics |
| `stats` | `core`, `multiple`, `signal` | HAC, bootstrap, permutation/random-entry, effective N, FDR, PSR/DSR, IC, Bayesian shrinkage |
| `research` | `registry`, `hypotheses`, `acceptance`, `splits`, `vault`, `labels`, `strategies`, `pipeline`, `measurements`, `catalog`, `program`, `generator`, `calibration`, `regimes`, `artifacts`, `data`, `session`, `opportunities`, `relative_value`, `models` | the scientific process |
| `montecarlo` | `simulate` | capital-path simulation with small-account frictions |
| `options` | `pricing`, `strategies`, `chain`, `execution` | BSM/IV/Greeks/CRR, payoffs, surfaces, conservative option fills |
| `risk` | `metrics`, `sizing`, `gate` | measurement, fractional Kelly, veto |
| `portfolio` | `combine` | signal combination, fractional-Kelly allocation |
| `paper` | `broker`, `forward` | Broker ABC, paper broker, forward ledger, monitoring, live lock |
| `analyst` | `facts`, `doc`, `charts`, `reports`, `assistant` | explanation only (read-only facts) |
| `dashboard` | `app` | local FastAPI views |

## Enforced rules (and the tests that enforce them)

| Rule | Mechanism | Test |
|---|---|---|
| No feature uses future data | truncation test on every family | `test_features.py::test_every_feature_passes_truncation_test`, leaky features caught |
| Features never see labels | AST import check | `test_features_package_never_imports_labels` |
| Fills strictly after information | engine timing assertions | `test_timing_guard_rejects_impossible_availability`, `test_trade_returns_match_labels_and_costs_reduce` |
| Simulation is contagious | `combine_labels` | `test_simulated_is_contagious` |
| Results are never overwritten/deleted | SQLite triggers + hash chain | `test_update_and_delete_are_forbidden`, `test_hash_chain_detects_tampering` |
| Hypotheses registered before evaluation, exactly as registered | spec hash check | `test_registration_is_required_and_exact` |
| Test data opened once, only when FROZEN | vault | `test_vault_requires_frozen_and_logs_contamination`, `test_null_market_is_not_accepted_and_vault_untouched` |
| Status lifecycle | transition table + approvals | `test_catalog_versions_and_lifecycle`, `test_rejected_is_terminal` |
| Analyst cannot write results | no writer methods; only `reports` appends | `test_analyst_cannot_write_results` |
| Risk can veto signals | RiskGate | `test_gate_vetoes_regardless_of_signal` |
| No live trading without flag + signed review | lock | `test_live_trading_lock` |
| Reproducibility | seeds, dataset hashes, code version | `test_experiment_is_reproducible`, `test_backtest_is_deterministic`, `test_monte_carlo_reproducible` |

## Extension points

* **New data vendor:** subclass `DataProvider`, implement only real capabilities, return canonical
  schemas with `available_at` and honest flags; add fixture-based parser tests.
* **New feature:** decorate a factory with `@feature_family(...)` in `features/library.py`; the
  parametrised truncation test picks it up automatically.
* **New strategy template:** add to `SIGNAL_TEMPLATES`/`POSITION_TEMPLATES` (or a new kind) in
  `research/strategies.py`; hypotheses reference it by name.
* **New measurement:** add a function to `research/measurements.py::MEASUREMENTS`.
* **Broker:** implement `paper.broker.Broker`; a live adapter must call
  `assert_live_trading_permitted()` and trade only LIVE_ELIGIBLE signals through the RiskGate.

## Storage layout (`QUANTLAB_DATA_DIR`, default `./data`, git-ignored)

```
data/store/<table>/<sha256>.parquet|.manifest.json, _versions.jsonl   market data
data/registry.sqlite                                                  research record (real)
data/artifacts/<experiment>/*.parquet                                  trades, daily returns
data/reports/R-xxxxxx_*.md|.html                                      saved reports
data/demo/...                                                         SIMULATED demo workspace (separate registry)
```
