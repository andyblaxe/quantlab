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

This project lives in the `quantlab/` subdirectory of a repository that also contains an unrelated
Next.js application; the two share nothing.
