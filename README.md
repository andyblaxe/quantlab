# QuantLab — local quantitative research laboratory

A research platform for discovering, **attempting to falsify**, validating and explaining trading
signals on historical market data. It is not a trading bot: real-money execution is not
implemented, and the architecture requires an explicit configuration change plus a safety review
before it ever could be.

Start with **[PROJECT_STATE.md](PROJECT_STATE.md)**: what is built, tested, simulated and planned.
The original architecture proposal and research plan is **[PLAN.md](PLAN.md)**.

| Doc | Contents |
|---|---|
| [PROJECT_STATE.md](PROJECT_STATE.md) | Current status, findings, known bugs, next task |
| [PLAN.md](PLAN.md) | Architecture proposal, data providers, roadmap, first hypotheses, spec flaws |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module map and design rules |
| [DATA.md](DATA.md) | Data model, provenance labels, point-in-time rules, providers |
| [RESEARCH_METHODOLOGY.md](RESEARCH_METHODOLOGY.md) | Lifecycle, statistics, acceptance criteria |
| [RISK_MANAGEMENT.md](RISK_MANAGEMENT.md) | Risk engine, sizing, ruin, live-trading safeguards |

## Setup

```bash
cd quantlab
uv venv .venv --python 3.11          # or: python3.11 -m venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env                 # optional: add FRED/Tiingo keys
.venv/bin/python -m pytest           # run the test suite
```

## Quickstart

```bash
# SIMULATED demo (separate registry under data/demo) — shows the whole lifecycle end to end
.venv/bin/quantlab demo                      # ~4 min: registers program v1 + a generated batch, evaluates all
.venv/bin/quantlab report --demo             # research report (saved to data/demo/reports)
.venv/bin/quantlab strategy-report 11 --demo --mode quant
.venv/bin/quantlab ask --demo "What have you rejected?"
.venv/bin/quantlab dashboard --demo          # http://127.0.0.1:8765
.venv/bin/quantlab calibrate                 # pipeline false-positive rate and power on simulated markets

# REAL research (needs network access to the data hosts, and FRED/Tiingo keys in .env)
.venv/bin/quantlab init                                  # registers the split plan in config/splits.toml
.venv/bin/quantlab data fetch --provider tiingo          # all v1 ETF symbols, raw prices + actions
.venv/bin/quantlab data fetch-macro --series VIX,VIX3M   # Cboe
.venv/bin/quantlab research register-program v1          # pre-register BEFORE evaluating
.venv/bin/quantlab research evaluate --all-registered    # development stage only
.venv/bin/quantlab research test H-000004                # one-time untouched test (FROZEN only)
.venv/bin/quantlab report
```

This project lives in the `quantlab/` subdirectory of a repository that also contains an unrelated
Next.js application; the two share nothing.
