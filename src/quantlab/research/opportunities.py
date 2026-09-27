"""Current opportunities: evaluate surviving signals on the latest available session.

For every non-rejected tradable signal: is its rule true on the most recent session (using only
data available then)? If so, propose a trade with its predicted return distribution (quantiles of
its historical net trades), a shrunk expected value, downside (5th percentile), the reason (feature
values), a conservative size, and the independent RiskGate's decision. If the rule is false, the
output is NO TRADE — an ordinary, acceptable outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analyst.facts import FactBase
from quantlab.features import get_feature
from quantlab.research.data import ResearchData
from quantlab.research.strategies import OPS, template_kind
from quantlab.risk.gate import ProposedTrade, RiskGate
from quantlab.risk.metrics import PortfolioState
from quantlab.risk.sizing import recommended_fraction
from quantlab.stats.signal import normal_shrinkage_posterior

TRADABLE = ("VALIDATING", "ACCEPTED", "PAPER_TRADING", "LIVE_ELIGIBLE")


def current_opportunities(fb: FactBase, data: ResearchData, equity: float = 10_000.0,
                          gate: RiskGate | None = None) -> list[dict]:
    gate = gate or RiskGate()
    out = []
    last = data.panel["close"].index[-1]
    state = PortfolioState(equity, equity, [], peak_equity=equity)
    for s in fb.signals():
        if s["Status"] not in TRADABLE:
            continue
        hid = s.get("Hypothesis_ID")
        spec = fb.hypothesis(hid)["payload"]["spec"]
        kind = template_kind(spec["strategy"])
        if kind not in ("event", "position") or "conditions" not in spec["params"]:
            continue
        symbols = [x for x in data.universes.get(spec["universe"], []) if x in data.panel["close"].columns]
        vals = {c[0]: get_feature(c[0]).compute(data.panel).loc[last, symbols] for c in spec["params"]["conditions"]}
        fire = pd.Series(True, index=symbols)
        for f, op, v in spec["params"]["conditions"]:
            fire &= OPS[op](vals[f], float(v)).fillna(False)
        ex = fb.development_experiment(hid)
        trades = fb.load_artifact(ex["id"], "trades") if ex else None
        net = trades["net_ret"] if trades is not None and len(trades) else pd.Series(dtype=float)
        post = normal_shrinkage_posterior(float(net.mean()), float(net.std(ddof=1) / np.sqrt(len(net)))) if len(net) > 2 else {}
        dist = {f"p{int(q * 100):02d}": float(net.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if len(net) else {}
        size = recommended_fraction(net)
        for sym in symbols:
            reason = ", ".join(f"{f}={vals[f][sym]:.3g}" for f in vals)
            base = {"signal_id": s["Signal_ID"], "name": s["Name"], "status": s["Status"], "symbol": sym,
                    "session": str(last.date()), "reason": reason, "data_label": data.label.value}
            if not fire[sym]:
                out.append({**base, "action": "NO TRADE", "why": "entry conditions not met"})
                continue
            px = float(data.panel["raw_close"].loc[last, sym])
            qty = np.floor(size["fraction"] * equity / px) if px > 0 else 0
            if qty <= 0:
                out.append({**base, "action": "NO TRADE", "why": f"sizing gives zero: {size['reason']}",
                            "expected_value": post.get("post_mean"), "distribution": dist})
                continue
            adv = float(data.panel["dollar_volume"][sym].iloc[-20:].mean())
            d = gate.evaluate(ProposedTrade(sym, qty, px, kind="etf", adv_dollar=adv, signal_id=s["Signal_ID"]), state)
            out.append({**base, "action": "BUY" if d.decision != "REJECT" else "NO TRADE (risk veto)",
                        "quantity": d.approved_quantity, "price_ref": px, "holding_sessions": spec["holding_period"],
                        "expected_value": post.get("post_mean"), "p_positive": post.get("p_positive"),
                        "distribution": dist, "downside_p05": dist.get("p05"), "risk_decision": d.decision,
                        "risk_reasons": d.reasons, "sizing": size})
    return out
