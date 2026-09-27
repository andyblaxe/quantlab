# RESEARCH_METHODOLOGY.md

How QuantLab decides whether a trading idea is real. The goal is to **disprove** hypotheses
efficiently and to accept only what survives. A profitable backtest is not evidence by itself.

## 1. Lifecycle

```
register (hash, timestamp)  →  development evaluation  →  FROZEN  →  untouched test (once)  →  catalog
        │                         │  screening (TRAIN)                       │
        │                         │  validation (VALIDATION, walk-forward,   └─ ACCEPTED / REJECTED
        │                         │  stress, multiple testing, DSR)
        │                         └─ REJECTED here never touches the test data
        └─ every variant is a new hypothesis and counts toward the multiple-testing burden
```

Signal statuses: `EXPERIMENTAL → VALIDATING → ACCEPTED → PAPER_TRADING → LIVE_ELIGIBLE`, with
`DEGRADED`, `RETIRED`, `REJECTED`. Transitions are enforced (`research/catalog.py`); REJECTED and
RETIRED are terminal; PAPER_TRADING and LIVE_ELIGIBLE need a named human approval.

Hypothesis lifecycle (separate): `REGISTERED → EVALUATED → FROZEN → TESTED → CONCLUDED`, forward only.

## 2. Data partitions

| Partition | Real-data dates (`config/splits.toml`) | Used for |
|---|---|---|
| TRAIN | start → 2012-12-31 | discovery / screening |
| VALIDATION | 2013-01 (+21-session embargo) → 2018-12-31 | confirmation, walk-forward range ends here |
| UNTOUCHED TEST | 2019-01 (+21-session embargo) → present | one evaluation per frozen hypothesis |

Decided on 2026-09-27 before any real data was examined. A trade belongs to a partition only if
its entry *and* exit are inside it (purging). The embargo (≥ longest holding period) stops a training
trade from realising its outcome in validation.

**Safeguards against test contamination** (`research/vault.py`): development code receives a
panel truncated at the validation end (test data is not in memory); the vault opens only for FROZEN
hypotheses; each opening is logged in the append-only registry; a second opening requires an
explicit acknowledgement and is permanently marked contaminated — a contaminated test cannot
support acceptance. Reports show the total number of vault openings, because each one uses up a
little of the programme-level holdout.

## 3. Timing and look-ahead prevention

* Every datum has `available_at` (see DATA.md). Daily bars: session close + 15 min (early closes
  handled).
* Features use trailing windows only; **every feature family passes an automatic truncation test**
  (compute on history cut at t, compare with full-history value at t). Deliberately leaky features
  (centered windows, `shift(-1)`, full-sample z-scores) are caught by tests.
* Decisions at close t execute at the **next** open or close; same-close execution is not offered.
  Engines assert fill time > decision availability.
* Labels (forward returns) live in `research/labels.py`; a test asserts the feature package never
  imports it.
* Prices are adjusted forward from raw data + corporate actions (no revised vendor history).
* Calendar features use the published exchange calendar (known in advance).

## 4. Evidence computed for every trading hypothesis

Per partition: per-trade gross and net statistics (mean, median, win rate, average win/loss, payoff,
profit factor), daily-return statistics (CAGR, vol, Sharpe, Sortino, max drawdown, VaR/ES),
effective sample size, and:

| Test | Question | Implementation |
|---|---|---|
| HAC t-test on daily net returns | Does it make money after costs, allowing for autocorrelation/overlap? | Newey–West, lags ≥ holding period |
| Random-entry test | Does *timing* add anything beyond drift? (the "random prediction" benchmark) | same symbols, period, horizon, direction mix; 1000+ draws |
| Excess vs buy-and-hold | For position strategies: does it beat simply holding the universe? | HAC test on daily net return difference |
| Stationary block bootstrap | How uncertain is the mean trade? | Politis–Romano, 95% CI |
| Walk-forward | Does *re-selecting parameters on the past* and trading forward work? | expanding folds, 3y min train, 1y test, purge = holding |
| Cost stress | Profitable at 2× costs? | full rerun with scaled cost model |
| Outliers | Profitable without the best 5% of trades? | trimmed mean |
| Concentration | Does one year produce > 40% of P&L? | P&L by entry year |
| Parameter sensitivity | Do pre-declared neighbouring parameters also work? | ≥ 70% net-positive |
| Multiple testing | Is it significant given everything else tried in its family? | Benjamini–Yekutieli (arbitrary dependence); untested hypotheses count as p = 1 |
| Deflated Sharpe | Does the Sharpe beat the luckiest of N null strategies? | Bailey & López de Prado; N = all hypotheses ever registered; dispersion = null sampling s.d. of the Sharpe (the empirical cross-trial dispersion is reported as a diagnostic only, because our trials are heterogeneous — cost-dominated strategies with very negative Sharpes inflated it and made the bar unreachable for genuine effects; found and fixed 2026-09-27) |
| Regime breakdown | Where does it fail? | trend (200-day), volatility (VIX vs 3y median), market drawdown > 20%, year; cells < 10 flagged |
| Stability | Strengthening, stable, weakening, disappeared? | slope of trade returns over time; first vs second half |
| Capacity | Feasible at $100 … $1M? | whole-share affordability, participation vs ADV |
| Monte Carlo | Distribution of outcomes for $100/$1k/$10k/$100k | block bootstrap of trade sequence; fixed costs and minimum position size; stressed (edge-halved) variant |

**Primary p-value** = the *larger* of (HAC p on daily net returns, relevant null test p). The strategy
must beat zero after costs *and* beat its null. Conservative by design.

## 5. Acceptance criteria (`research/acceptance.py`, frozen into each hypothesis hash)

Screening (TRAIN only): effective N ≥ 30 (else INCONCLUSIVE), mean net trade > 0, primary p < 0.05.

Validation (all must pass; any unevaluable criterion ⇒ INCONCLUSIVE, never PASS):
effective N ≥ 100 overall and ≥ 30 in validation · validation mean net trade > 0 · validation Sharpe
> 0 and ≥ 0.5 × train Sharpe · walk-forward ≥ 60% positive folds and pooled OOS Sharpe > 0 ·
BY q < 0.10 · DSR ≥ 0.95 · bootstrap 95% CI lower bound > 0 · positive at 2× costs · positive
without top 5% trades · no year > 40% of P&L · ≥ 70% of parameter neighbours positive.

Untouched test: not contaminated · effective N ≥ 30 · mean net trade > 0 · Sharpe > 0.
Failure ⇒ REJECTED; re-tuning is impossible because the spec hash is frozen.

LIVE_ELIGIBLE (future, paper trading): ≥ 60 paper days and ≥ 30 paper trades, forward mean not below
the 5th percentile of the historical predictive distribution, realised slippage ≤ 1.5× modelled.

## 6. Measurement hypotheses

Some questions are prerequisites, not strategies (e.g. "is the volatility risk premium positive?").
They are pre-registered, tested with HAC inference on TRAIN (p enters the family's BY correction),
checked for sign consistency over the development period, and recorded as SUPPORTED /
NOT_SUPPORTED / INCONCLUSIVE. They never enter the tradable-signal catalog.

## 7. Automated hypothesis generation

`research/generator.py`: a handful of templates, each with a declared mechanism and family; small
declared grids; a hard budget per batch (default 24; a grid larger than the budget is refused rather
than sampled); every spec registered before evaluation, tagged `exploratory`; parameter neighbours
from grid adjacency; the batch is journaled. All generated hypotheses — including every failure —
count toward the family's multiple-testing burden and appear in reports.

## 8. Benchmarks

Every strategy is compared with: zero (after costs), random entries (timing), and buy-and-hold of the
same universe (position strategies). Simple momentum / mean reversion benchmarks are in
`backtest/benchmarks.py`. Linear/logistic model benchmarks and ML models are planned (Phase 9);
complex models must beat the simple ones out of sample after costs.

## 9. Calibration of the machinery (SIMULATED)

`quantlab calibrate` runs the pipeline on synthetic markets with no effect (size) and with a planted
reversal (power) and records the rates as a project event. See PROJECT_STATE.md for the latest numbers.
A known limitation: a planted edge smaller than transaction costs is (correctly) rejected, and a
moderate one that is hurt by a stressed regime inside TRAIN can fail screening — the pipeline is
tuned to prefer false negatives over false positives.

## 10. Known methodological limitations

* BY correction is conservative; power is low for weak effects. Deliberate.
* Effective N via calendar blocks is a heuristic, not an exact dependence correction.
* The event-study bootstrap treats events as independent; earnings events cluster by season
  (clustered bootstrap planned).
* Position-strategy "trades" are holding spells with per-side cost approximated by spread+slippage;
  the daily return series comes from the full engine with complete costs.
* Walk-forward selection uses net Sharpe over each training window among the declared parameter set
  only — it cannot discover parameters outside the pre-declared neighbourhood (by design).
