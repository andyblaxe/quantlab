"""RiskGate: independent approval of every proposed trade.

The gate receives a proposed trade and the current portfolio state — nothing about how the signal
was produced — and returns APPROVE, REDUCE (with an approved size) or REJECT with reasons. A signal
the model loves is rejected just the same if it breaches a limit. Every decision can be logged.

Limits are deliberately conservative defaults for a small account; they are configuration, not code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from quantlab.risk.metrics import PortfolioState, Position


@dataclass(frozen=True)
class RiskLimits:
    max_position_fraction: float = 0.20
    max_gross_leverage: float = 1.0
    max_net_leverage: float = 1.0
    max_sector_fraction: float = 0.40
    max_beta_exposure: float = 1.2
    max_drawdown_halt: float = 0.20  # stop opening new risk beyond this drawdown from peak
    max_event_risk_fraction: float = 0.10  # e.g. positions held through earnings
    max_participation_adv: float = 0.01
    max_option_premium_fraction: float = 0.05  # premium at risk per option trade
    max_portfolio_var95: float = 0.03  # 1-day historical VaR as fraction of equity
    max_abs_vega_per_equity: float = 0.002  # $vega per 1 vol point per $ equity
    allow_short: bool = False
    allow_naked_short_options: bool = False
    min_cash_buffer: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProposedTrade:
    symbol: str
    quantity: float  # + buy / − sell (shares or contracts)
    price: float
    kind: str = "equity"
    sector: str = "UNKNOWN"
    beta: float = 1.0
    multiplier: float = 1.0
    adv_dollar: float = np.nan
    event_risk: bool = False
    option_greeks: dict = field(default_factory=dict)  # position-level greeks after the trade
    max_loss: float | None = None  # dollars; None = unknown/unlimited
    is_short_option: bool = False
    defined_risk: bool = True
    signal_id: str | None = None

    @property
    def notional(self) -> float:
        return self.quantity * self.price * self.multiplier


@dataclass
class GateDecision:
    decision: str  # APPROVE | REDUCE | REJECT
    approved_quantity: float
    reasons: list[str]
    checks: dict

    def to_dict(self) -> dict:
        return asdict(self)


class RiskGate:
    def __init__(self, limits: RiskLimits = RiskLimits()) -> None:
        self.limits = limits

    def evaluate(self, trade: ProposedTrade, state: PortfolioState) -> GateDecision:
        L = self.limits
        eq = state.equity
        reasons: list[str] = []
        checks: dict = {}
        if eq <= 0:
            return GateDecision("REJECT", 0.0, ["no equity"], {})
        exp = state.exposures()
        # --- hard rejections (size cannot fix) ---
        if exp["drawdown"] <= -L.max_drawdown_halt and trade.quantity > 0:
            reasons.append(f"drawdown {exp['drawdown']:.1%} beyond halt level {-L.max_drawdown_halt:.0%}: no new risk")
        existing = sum(p.quantity for p in state.positions if p.symbol == trade.symbol)
        if trade.kind != "option" and existing + trade.quantity < 0 and not L.allow_short:
            reasons.append("would create a short stock position; shorting not permitted")
        if trade.is_short_option and not trade.defined_risk and not L.allow_naked_short_options:
            reasons.append("naked short option (undefined risk) not permitted")
        if trade.kind == "option" and trade.quantity > 0 and trade.max_loss is None:
            reasons.append("option trade without a computed maximum loss")
        if reasons:
            return GateDecision("REJECT", 0.0, reasons, checks)

        # --- size limits: compute the largest compliant quantity ---
        unit = abs(trade.price * trade.multiplier)
        caps = {}
        caps["position"] = (L.max_position_fraction * eq - abs(existing * unit)) / unit
        gross_now = exp["gross_leverage"] * eq
        caps["gross_leverage"] = (L.max_gross_leverage * eq - gross_now) / unit
        sector_now = exp["sector_net"].get(trade.sector, 0.0) * eq
        caps["sector"] = (L.max_sector_fraction * eq - abs(sector_now)) / unit
        beta_now = exp["beta_exposure"] * eq
        caps["beta"] = (L.max_beta_exposure * eq - beta_now) / max(unit * abs(trade.beta), 1e-12)
        spendable = state.cash - L.min_cash_buffer * eq
        if trade.quantity > 0:
            caps["cash"] = spendable / unit
        if np.isfinite(trade.adv_dollar):
            caps["liquidity"] = L.max_participation_adv * trade.adv_dollar / unit
        if trade.event_risk:
            ev_now = exp["event_risk_exposure"] * eq
            caps["event_risk"] = (L.max_event_risk_fraction * eq - ev_now) / unit
        if trade.kind == "option" and trade.max_loss is not None and trade.quantity != 0:
            loss_per_unit = abs(trade.max_loss / trade.quantity)
            caps["option_premium_at_risk"] = L.max_option_premium_fraction * eq / max(loss_per_unit, 1e-12)
        if trade.option_greeks.get("vega") is not None and trade.quantity != 0:
            vega_unit = abs(trade.option_greeks["vega"] / trade.quantity) / 100
            caps["vega"] = L.max_abs_vega_per_equity * eq / max(vega_unit, 1e-12)
        if state.daily_returns is not None and len(state.daily_returns.dropna()) >= 60:
            from quantlab.risk.metrics import historical_var_es
            var = historical_var_es(state.daily_returns)["var"]
            checks["portfolio_var95"] = var
            if var > L.max_portfolio_var95 and trade.quantity > 0:
                return GateDecision("REJECT", 0.0, [f"portfolio 1-day VaR95 {var:.2%} exceeds {L.max_portfolio_var95:.2%}"], checks)
        checks["caps_units"] = {k: float(v) for k, v in caps.items()}
        allowed = max(0.0, min(caps.values())) if trade.quantity > 0 else abs(trade.quantity)  # reducing risk is always allowed
        if trade.kind in ("equity", "etf", "option"):
            allowed = float(np.floor(allowed + 1e-9)) if trade.kind == "option" or trade.multiplier > 1 else allowed
        want = abs(trade.quantity)
        if allowed <= 0:
            binding = min(caps, key=caps.get)
            return GateDecision("REJECT", 0.0, [f"no capacity under limit '{binding}'"], checks)
        if allowed + 1e-9 < want:
            binding = min(caps, key=caps.get)
            return GateDecision("REDUCE", float(np.sign(trade.quantity) * allowed),
                                [f"reduced from {want:g} to {allowed:g} by limit '{binding}'"], checks)
        return GateDecision("APPROVE", trade.quantity, ["within all limits"], checks)
