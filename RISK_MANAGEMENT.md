# RISK_MANAGEMENT.md

Risk management is **independent of signal generation**. The risk modules never import strategy or
signal code; the RiskGate sees only a proposed trade and the portfolio state, and can reject a trade
the model likes.

## 1. Objectives (in order)

Low probability of ruin → controlled drawdowns → positive expected value after costs → long-run
geometric growth. Maximising historical CAGR is never an objective; win rate is never a sizing input.

## 2. Measurement (`risk/metrics.py`)

| Measure | Implementation |
|---|---|
| Portfolio volatility | wᵀΣw with shrunk (constant-correlation target) covariance |
| VaR / expected shortfall | historical (default), Gaussian, Cornish–Fisher (fat-tail cross-check) |
| Leverage | gross and net market value / equity |
| Concentration | largest position / equity |
| Sector exposure | net market value by sector / equity |
| Beta exposure | Σ market value × beta / equity |
| Option Greeks | aggregated position delta, gamma, theta, vega |
| Liquidity risk | fraction of equity in positions > 5% of ADV |
| Overnight / event risk | exposure held overnight; exposure through flagged events (e.g. earnings) |
| Drawdown | equity vs running peak |

## 3. RiskGate (`risk/gate.py`)

Default limits (`RiskLimits`, configuration not code):

| Limit | Default |
|---|---|
| Max position | 20% of equity |
| Max gross / net leverage | 1.0 / 1.0 (no leverage) |
| Max sector | 40% |
| Max beta exposure | 1.2 |
| Drawdown halt (no new risk) | −20% from peak |
| Max event-risk exposure | 10% |
| Max participation | 1% of ADV |
| Max option premium at risk per trade | 5% of equity |
| Max portfolio 1-day VaR95 | 3% |
| Max vega | $0.002 per vol point per $ equity |
| Shorting / naked short options | disabled |
| Cash buffer | 2% |

Decisions: **REJECT** (hard rule broken: drawdown halt, shorting disabled, naked short option,
option without computed max loss, VaR breach, no capacity), **REDUCE** (size cut to the binding limit,
which is named), **APPROVE**. Reducing existing risk is always allowed.

## 4. Sizing (`risk/sizing.py`)

* Edge used for sizing = one-sided 95% **lower confidence bound** of the mean trade return.
* Kelly fraction computed **numerically on the empirical trade distribution** (respects skew/fat
  tails), on the lower-bound-shifted returns.
* Use **¼ Kelly**, then cap at 20% of equity per position. Never full Kelly.
* Fewer than 30 trades, or a non-positive lower bound ⇒ size 0 (NO TRADE).
* `ruin_probability_gaussian` provides an analytic cross-check; it ignores fat tails and is optimistic.

## 5. Probability of ruin and Monte Carlo (`montecarlo/simulate.py`)

Default ruin definition: equity ≤ 20% of starting capital, **or** equity below the minimum position
size (cannot take the next trade). Simulations resample trade sequences in blocks for starting
capitals $100 / $1,000 / $10,000 / $100,000, with a fixed per-trade dollar cost (commission minimum)
and a minimum position size (one share / one option contract). Outputs: ending-capital, CAGR and
max-drawdown distributions, P(loss), P(−10/−25/−50%), P(ruin), P(double), time-to-double; plus a
stressed variant with the average edge halved. Resampling cannot produce crises absent from history;
medians, not best paths, are expectations.

**Small accounts:** a $1 minimum commission on a $200 position is 0.5% per trade each way; one
option contract commonly exceeds a $100–$1,000 account's per-position limit; shorting and most
spreads need a margin account. The Monte Carlo and capacity sections report these frictions
explicitly — many edges that exist at $100,000 do not exist at $1,000.

## 6. Portfolio construction (`portfolio/combine.py`)

Signals are shrunk toward zero, scaled by decay (STABLE 1.0, WEAKENED 0.5, DISAPPEARED 0) and regime
multipliers, and combined with Σ⁻¹μ on a shrunk covariance so correlated signals share weight (the
effective number of independent signals is reported). Allocation: fractional-Kelly mean–variance
(f = ¼) with 20% per-position and 100% gross caps, long-only by default. Inactive signals contribute
nothing; if no adjusted edge is positive the output is NO TRADE.

## 7. Live trading safeguards

Real-money execution is **not implemented**. The `Broker` interface exists so that it could be
added later; any live broker implementation must refuse to start unless **both**
`QUANTLAB_LIVE_TRADING_ENABLED=true` **and** `QUANTLAB_LIVE_SAFETY_REVIEW_FILE` points to a signed
safety-review file, and a strategy may only trade if its catalog status is `LIVE_ELIGIBLE`
(statistical, risk, execution and ≥60-day / ≥30-trade forward-test requirements; see
RESEARCH_METHODOLOGY.md §5). Paper trading uses the same RiskGate as any future live path.
