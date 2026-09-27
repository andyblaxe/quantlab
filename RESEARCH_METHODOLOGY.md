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

### 1a. Research-integrity promotion gate — PERMANENT PROJECT RULE

> **A strategy or signal whose results are materially affected by known survivorship bias may NEVER
> be promoted to ACCEPTED, PAPER_TRADING or LIVE_ELIGIBLE.** More broadly, no signal may be promoted
> to those statuses while its supporting evidence has an unresolved **material** research-integrity
> problem. This rule is an architectural requirement: it must not be removed, weakened or bypassed,
> and changes to it need an explicit decision by the project owner recorded in the registry.

The gate has two layers.

**1. Permanent integrity principles (fixed in code, `research/integrity.py`).** Evidence must be free
of survivorship bias, look-ahead, leakage and test contamination; use valid point-in-time data with
correct timestamps and corporate actions; assume execution that happens only after the decision's
information was available; include appropriate transaction costs and adequate liquidity/capacity
assumptions; carry sufficient statistical evidence; and be corrected appropriately for multiple
testing. These cannot be configured away.

**2. Configurable validation thresholds (`research/acceptance.py`, `AcceptanceCriteria`).** The
numbers that implement some principles — minimum effective sample size, FDR level, the execution
timings a strategy may use, the maximum participation in ADV — depend on the statistical methodology,
the independence structure of the observations, the holding period, the instrument and the execution
model. They are **pre-registered and versioned**: each hypothesis snapshots its criteria (with
`version`) into its spec hash at registration, so they cannot change after results are seen; a
variant is a new hypothesis and counts toward the multiple-testing burden. The class defaults are the
conservative project defaults (effective N ≥ 30, BY q ≤ 0.10, `next_open`/`next_close`, ≤ 10% of ADV)
and apply wherever no strategy-specific methodology exists. A hypothesis may register different
values; any value **looser** than the default is refused at registration unless the criteria carry a
written `justification`. The gate always reads thresholds from the registered hypothesis — never
from values stored in the signal record.

Two kinds of finding, both non-promotable, kept distinct:

* **RESEARCH-INTEGRITY FAILURE** — the evidence is methodologically compromised. Grade `BIASED / …`
  (evidence is wrong) or `PRELIMINARY / …` (incomplete or optimistic).
* **INSUFFICIENT EVIDENCE** — the method may be sound but there is not yet enough evidence: effective
  sample below the registered minimum, or not significant after the registered multiple-testing
  correction. Grade `INSUFFICIENT_EVIDENCE / …`. A small sample does not imply bias.

| Principle / problem | Kind | Detection (threshold source) |
|---|---|---|
| Survivorship bias | integrity | universe flag `SURVIVORSHIP_BIASED_UNIVERSE` (universe kinds `etf` / `stock_pit` / `stock_current`; undeclared ⇒ `stock_current`) |
| Look-ahead bias | integrity | automatic truncation test on every feature the hypothesis uses |
| Data leakage | integrity | pipeline input `leakage`, or a manual finding |
| Test-set contamination | integrity | vault access count at the untouched test |
| Materially incomplete universe | integrity | manual finding |
| Incorrect point-in-time data | integrity | manual finding |
| Execution before information was available | integrity | execution model not known to occur after the information (e.g. `same_close` on daily bars) — principle, not configurable |
| Execution outside the registered methodology | integrity | execution not in registered `allowed_executions` |
| Liquidity / capacity | integrity | cost-model participation > registered `max_participation` |
| Missing transaction costs | integrity | cost model with no frictions |
| Corporate-action / timestamp / price errors | integrity | force-saved datasets with those validation errors; `MODEL_PRICED` data; manual findings |
| Uncorrected multiple testing | integrity | a p-value with no adjusted p-value |
| Any other defect that could invalidate the edge | integrity | manual finding (`raise_finding`) |
| Insufficient sample | insufficient evidence | effective N < registered `min_effective_n_inconclusive` |
| Not significant after correction | insufficient evidence | BY q > registered `fdr_q` |

How it is enforced (in code, not in prompts or reports):

* `catalog.transition()` refuses ACCEPTED / PAPER_TRADING / LIVE_ELIGIBLE when `integrity.assess()`
  finds a MATERIAL problem, **whoever approves**, and journals `promotion_blocked` with the grade and
  every blocking finding before refusing. There is no override parameter.
* The assessment is the union over **every stored version** of the signal record plus open manual
  findings. Editing a record later cannot remove a detected problem; `Evidence_Grade` and
  `Integrity_Findings` are derived on every write and cannot be set by hand.
* Manual findings (`raise_finding`) are closed only by `resolve_finding` with a reason, a resolver and
  an experiment recorded **after** the finding — new evidence, not approval. MINOR findings are
  recorded but do not block.
* The pipeline withholds the one-shot untouched test from non-promotable evidence (the vault stays
  sealed for a clean re-run), and if a problem appears at the test it stops at VALIDATING.
* Paper trading refuses new trades, and the live-eligibility check fails, for any signal with an open
  material finding — including one raised after promotion.
* Blocked research is kept and analysable. Its `Evidence_Grade` names the kind and the problems,
  e.g. `PRELIMINARY / SURVIVORSHIP-BIASED`, `BIASED / LOOK-AHEAD`,
  `INSUFFICIENT_EVIDENCE / INADEQUATE-SAMPLE`, or both parts joined by `; `. The refusal journal
  lists integrity failures and insufficient evidence separately.
* If an override is ever introduced, it must move the signal into a separate, explicitly labelled
  experimental state and must never turn compromised evidence into validated evidence.

Tests: `tests/test_integrity.py` (every category blocks, approval does not override, refusals are
journaled, history cannot be rewritten, resolution needs new evidence, post-promotion findings stop
paper/live, thresholds come from the registered methodology, looser thresholds need a registered
justification, small samples are insufficient evidence rather than bias), `tests/test_pipeline.py` (survivorship and look-ahead end to end),
`tests/test_catalog_mc_regimes.py` (original survivorship gate).

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
`backtest/benchmarks.py`. Model benchmarks (`research/models.py`): walk-forward logistic regression
and gradient boosting predicting P(forward return > 0), evaluated only out of sample against
always-long and random-entry baselines after costs, with calibration tables; MODEL EXPLANATION
(permutation importance, partial dependence, prediction distribution) is reported separately and
never as causal evidence. These are tools today — not yet wired into hypothesis templates, so no
model has been through the vault. Complex models must beat the simple ones out of sample after costs.

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
