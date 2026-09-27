"""Research Analyst: reports generated deterministically from stored facts.

The analyst explains; it never decides. Every number printed here is read from the registry or a
hash-verified artifact and printed with the record ID it came from. No text says "winning strategy";
statuses are the catalog's own vocabulary. When nothing survived, the executive summary says so
verbatim: "No sufficiently robust trading edge has been identified."

Two modes: ``plain`` (for an intelligent investor without statistical training) and ``quant``
(test statistics, criteria values, methodology). Both render from the same facts.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.analyst import charts
from quantlab.analyst.doc import Doc
from quantlab.analyst.facts import FactBase, dig, fmt
from quantlab.research.registry import Registry

NO_EDGE = "No sufficiently robust trading edge has been identified."

PLAIN = {
    "p_value": "the chance of seeing a result at least this good if the idea had no real edge",
    "q_value": "the p-value after adjusting for how many ideas we tried (false-discovery control)",
    "dsr": "probability the risk-adjusted return is real after accounting for every strategy tried",
    "effective_n": "number of genuinely independent observations (overlapping or same-day trades count once)",
    "walk_forward": "re-choosing parameters using only past data, then trading the next year, repeatedly",
}

MECHANISM_TEXT = {
    "RISK_PREMIUM": "compensation for bearing a risk others want to shed (e.g. crash risk)",
    "BEHAVIORAL": "systematic investor behaviour such as under-reaction or herding",
    "LIQUIDITY": "payment for supplying liquidity to forced or impatient sellers/buyers",
    "STRUCTURAL": "market structure or trading-hours effects",
    "VOLATILITY": "the dynamics of volatility itself (clustering, mean reversion, regime shifts)",
    "INSTITUTIONAL": "predictable institutional flows (month-end rebalancing, payroll investment)",
    "MICROSTRUCTURE": "order-book and execution mechanics",
}


def _banner(doc: Doc, fb: FactBase) -> None:
    c = fb.counts()
    labels = c["data_labels_used"]
    if not labels:
        doc.banner("No experiments have been run yet.", "warn")
    elif "SIMULATED" in labels:
        doc.banner("SIMULATED DATA: results below come from synthetic markets. They test the research machinery "
                   "and are NOT evidence about real markets.", "bad")
    else:
        doc.banner(f"Data labels used: {', '.join(labels)}.", "ok")


def _failed_criteria(exp: dict) -> list[str]:
    out = []
    for stage in ("screening", "validation_decision"):
        d = exp["payload"].get(stage) or {}
        out += [r["name"] for r in d.get("results", []) if r["passed"] is False]
    return out


# =============================================================================================
# Research report
# =============================================================================================
def build_research_report(fb: FactBase, mode: str = "plain") -> Doc:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    c = fb.counts()
    last = fb.last_report("research")
    changes = fb.changes_since(last["created_at"] if last else None)
    doc = Doc("QuantLab Research Report").h1(f"Research Report — {now[:10]}")
    doc.p(f"Generated {now} from registry records ({mode} mode). Previous report: "
          f"{last['id'] + ' at ' + last['created_at'] if last else 'none'}.")
    _banner(doc, fb)

    ranked = fb.ranked_signals()
    accepted = [s for s in ranked if s["Status"] in ("ACCEPTED", "PAPER_TRADING", "LIVE_ELIGIBLE")]
    validating = [s for s in ranked if s["Status"] == "VALIDATING"]
    rejected = fb.rejected()

    # ---- executive summary ----
    doc.h2("Executive summary")
    lines = [
        f"Research performed: {c['hypotheses_registered']} hypotheses registered "
        f"({c['hypotheses_generated_automatically']} generated automatically), {c['hypotheses_tested']} evaluated, "
        f"{c['experiments_completed']} experiments completed, {c['untouched_tests_run']} one-time untouched tests run "
        f"(vault openings: {c['vault_openings']}, contaminated: {c['contaminated_tests']}).",
    ]
    if accepted:
        lines.append("Signals that survived development validation AND the untouched test: " + "; ".join(
            f"{s['Signal_ID']} {s['Name']} ({s['Status']})" for s in accepted) + ".")
        if fb.uses_only_simulated_data():
            lines.append("Because every experiment used SIMULATED data, none of this is evidence of a real-market edge. "
                         + NO_EDGE)
    else:
        lines.append(NO_EDGE)
    if validating:
        lines.append("Promising but unconfirmed (VALIDATING — passed development criteria, awaiting or failed to "
                     "complete the untouched test): " + "; ".join(f"{s['Signal_ID']} {s['Name']}" for s in validating) + ".")
    reasons = Counter(r for e in fb.reg.find("experiments", kind="development") for r in _failed_criteria(e))
    if rejected:
        lines.append(f"Rejected: {len(rejected)} signals. Most common failed criteria: "
                     + ", ".join(f"{k} ({v})" for k, v in reasons.most_common(4)) + ".")
    queued = [h for h in fb.hypotheses() if h["status"] == "REGISTERED"]
    lines.append(f"Currently queued for evaluation: {len(queued)} registered hypotheses"
                 + (f" (e.g. {', '.join(h['id'] for h in queued[:5])})." if queued else "."))
    blocked = [m for m in fb.measurements() if m["status_detail"] == "DATA_UNAVAILABLE"]
    nxt = []
    if [s for s in validating if fb.reg.current_status("hypothesis", s.get("Hypothesis_ID")) == "FROZEN"]:
        nxt.append("run the one-time untouched test on FROZEN hypotheses")
    if queued:
        nxt.append("evaluate queued hypotheses")
    if blocked:
        nxt.append("acquire missing data for " + ", ".join(m["hypothesis_id"] for m in blocked))
    if fb.uses_only_simulated_data() or not c["data_labels_used"]:
        nxt.append("ingest REAL market data (blocked in the build environment by network policy) and run program v1")
    lines.append("Suggested next steps: " + ("; ".join(nxt) if nxt else "none pending") + ".")
    doc.ul(lines)

    # ---- changes ----
    doc.h2("What changed since the last report")
    if last is None:
        doc.p("This is the first report; everything below is new.")
    else:
        sc = changes["status_changes"]
        doc.ul([f"New hypotheses: {len(changes['new_hypotheses'])}",
                f"New experiments: {len(changes['new_experiments'])}",
                f"Signals promoted: {sum(1 for x in sc if x['to'] in ('VALIDATING', 'ACCEPTED', 'PAPER_TRADING', 'LIVE_ELIGIBLE'))}",
                f"Signals rejected: {sum(1 for x in sc if x['to'] == 'REJECTED')}",
                f"Signals degraded/retired: {sum(1 for x in sc if x['to'] in ('DEGRADED', 'RETIRED'))}",
                f"Vault openings: {len(changes['vault_openings'])}",
                f"Split-plan changes: {len(changes['split_plan_changes'])}"
                + (" — WARNING: the data partition changed after research began" if changes["split_plan_changes"] else "")])
        if sc:
            doc.table(["Event", "Signal", "From", "To", "Reason"],
                      [[x["id"], x["entity"], x["from"] or "", x["to"], (x["reason"] or "")[:120]] for x in sc])

    # ---- counts ----
    doc.h2("Research inventory")
    sb = c["signals_by_status"]
    doc.table(["Measure", "Count"], [
        ["Hypotheses generated automatically", c["hypotheses_generated_automatically"]],
        ["Hypotheses registered (total)", c["hypotheses_registered"]],
        ["Hypotheses tested", c["hypotheses_tested"]],
        ["Experiments completed", c["experiments_completed"]],
        ["Experiments running", c["experiments_running"]],
        ["Strategies rejected", sb.get("REJECTED", 0)],
        ["Strategies under investigation (EXPERIMENTAL)", sb.get("EXPERIMENTAL", 0)],
        ["Strategies in validation (VALIDATING)", sb.get("VALIDATING", 0)],
        ["Strategies accepted (ACCEPTED)", sb.get("ACCEPTED", 0)],
        ["Promoted to paper trading", sb.get("PAPER_TRADING", 0)],
        ["Live-eligible", sb.get("LIVE_ELIGIBLE", 0)],
        ["Degraded", sb.get("DEGRADED", 0)],
        ["Retired", sb.get("RETIRED", 0)],
    ])
    doc.table(["Family", "Hypotheses (multiple-testing burden)"], [[k, v] for k, v in sorted(c["hypotheses_by_family"].items())])

    # ---- important experiments ----
    doc.h2("Important experiments")
    important = [h for h in fb.hypotheses() if not h["exploratory"]]
    important += [h for h in fb.hypotheses() if h["exploratory"] and (fb.signal("SIG-" + h["id"][2:]) or {}).get("Status")
                  not in (None, "REJECTED")]
    if not important:
        doc.p("No experiments have been performed yet.")
    for h in important:
        _experiment_section(doc, fb, h, mode)

    # ---- measurements ----
    ms = fb.measurements()
    if ms:
        doc.h2("Measurements (not trading strategies)")
        doc.table(["Hypothesis", "Experiment", "Outcome", "Estimate", "p", "BY q"],
                  [[f"{m['hypothesis_id']} {m['name']}", m["experiment_id"], m["outcome"], fmt(m["estimate"]),
                    fmt(m["p_value"], "p"), fmt(m["q_value"], "p")] for m in ms])

    # ---- rejected ----
    doc.h2("Rejected ideas (permanent record)")
    doc.table(["Signal", "Hypothesis", "Name", "Reason", "Record"],
              [[r["signal_id"], r["hypothesis_id"], r["name"], (r["reason"] or "")[:140], r["status_event"]] for r in rejected])

    # ---- generated ----
    gen = [h for h in fb.hypotheses() if h["exploratory"]]
    if gen:
        doc.h2("Automatically generated hypotheses")
        st = Counter((fb.signal("SIG-" + h["id"][2:]) or {}).get("Status", h["status"]) for h in gen)
        doc.p(f"{len(gen)} generated hypotheses; outcomes: " + ", ".join(f"{k}: {v}" for k, v in st.items())
              + ". All count toward their family's multiple-testing burden.")

    # ---- calibration ----
    cal = fb.calibration()
    doc.h2("Reliability of the research machinery")
    if cal:
        doc.p(f"Calibration {cal['id']} (SIMULATED markets): with no real effect, "
              f"{fmt(cal['null_accepted_rate'], 'pct')} of {cal['null_runs']} runs produced an ACCEPTED signal; "
              f"with a planted effect, {fmt(cal['power_accepted_rate'], 'pct')} of {cal['power_runs']} runs detected it. "
              "Small run counts give wide uncertainty on these rates.")
    else:
        doc.p("No calibration has been recorded in this registry (run `quantlab calibrate`).")
    return doc


def _experiment_section(doc: Doc, fb: FactBase, h: dict, mode: str) -> None:
    spec = fb.hypothesis(h["id"])["payload"]["spec"]
    ex = fb.development_experiment(h["id"])
    doc.h3(f"{h['id']} — {h['name']}")
    if ex is None:
        doc.p("Not yet evaluated (registered; queued).")
        return
    r = ex["payload"]
    if r.get("kind") == "measurement":
        doc.kv([("HYPOTHESIS", spec["statement"]), ("WHY IT WAS TESTED", spec["rationale"]),
                ("RESULT", f"{r['outcome']} — estimate {fmt(dig(r, 'train.estimate'))} [{ex['id']}]"),
                ("STATISTICAL SIGNIFICANCE", f"p={fmt(dig(r, 'train.p_value'), 'p')}, BY q={fmt(dig(r, 'multiple_testing.q_value'), 'p')}"),
                ("CONCLUSION", fb.reg.find("journal", experiment_id=ex["id"])[-1]["payload"].get("conclusion", ""))])
        return
    test = fb.test_experiment(h["id"])
    tn, vn = r["train"]["trades_net"], r["validation"]["trades_net"]
    tg = r["train"]["trades_gross"]
    j = fb.reg.find("journal", experiment_id=ex["id"])
    concl = j[-1]["payload"].get("conclusion", "") if j else ""
    nxt = j[-1]["payload"].get("next_step", "") if j else ""
    oos = "Not run (hypothesis did not reach FROZEN — the untouched data was preserved)."
    if test:
        tt = test["payload"]["test"]["trades_net"]
        oos = (f"{test['payload']['decision']['outcome']}: mean net trade {fmt(tt.get('mean'), 'pct')} over "
               f"{tt.get('n_trades', 0)} trades [{test['id']}]")
    sig = (f"train p={fmt(r['train']['p_primary'], 'p')} ({PLAIN['p_value']}); BY q={fmt(r['multiple_testing']['q_value'], 'p')} "
           f"over {r['multiple_testing']['family_size']} hypotheses; DSR={fmt(r['deflated_sharpe']['dsr'])}")
    if mode == "plain":
        sig = (f"p={fmt(r['train']['p_primary'], 'p')} — {PLAIN['p_value']}. After adjusting for the "
               f"{r['multiple_testing']['family_size']} ideas tried in this family: q={fmt(r['multiple_testing']['q_value'], 'p')}.")
    rows = [
        ("HYPOTHESIS", spec["statement"]),
        ("WHY IT WAS TESTED", spec["rationale"]),
        ("DATA USED", f"{r['data']['label']} {r['data']['source']} {r['data']['start']}→{r['data']['end']}, universe "
                      f"{spec['universe']}; flags: {', '.join(r['data']['flags'] + r.get('universe_flags', [])) or 'none'}"),
        ("METHOD", f"{spec['strategy']} {spec['params']}; hold {spec['holding_period']} sessions; {spec['execution']} fills; "
                   f"costs '{spec['cost_profile']}'; train {r['bounds']['train'][0]}→{r['bounds']['train'][1]}, "
                   f"validation {r['bounds']['validation'][0]}→{r['bounds']['validation'][1]}"),
        ("RESULT", f"train: {tn.get('n_trades', 0)} trades, mean net {fmt(tn.get('mean'), 'pct')}; validation: "
                   f"{vn.get('n_trades', 0)} trades, mean net {fmt(vn.get('mean'), 'pct')} [{ex['id']}]"),
        ("STATISTICAL SIGNIFICANCE", sig),
        ("ECONOMIC SIGNIFICANCE", f"train net Sharpe {fmt(r['train']['daily_net'].get('sharpe'))}, "
                                  f"max drawdown {fmt(r['train']['daily_net'].get('max_drawdown'), 'pct')}"),
        ("TRANSACTION-COST IMPACT", f"gross {fmt(tg.get('mean'), 'pct')} → net {fmt(tn.get('mean'), 'pct')} per trade (train); "
                                    f"at 2× costs {fmt(r['stress'].get('cost_x2_mean_net_trade'), 'pct')}"),
        ("OUT-OF-SAMPLE RESULT", oos),
        ("CONCLUSION", concl),
        ("NEXT STEP", nxt),
    ]
    if mode == "quant":
        failed = _failed_criteria(ex)
        rows.insert(6, ("CRITERIA FAILED", ", ".join(failed) or "none"))
        rows.insert(7, ("WALK-FORWARD", f"{r['walk_forward'].get('n_folds', 0)} folds, "
                                         f"{fmt(r['walk_forward'].get('positive_fold_frac'), 'pct')} positive, pooled OOS Sharpe "
                                         f"{fmt(dig(r, 'walk_forward.pooled_oos.sharpe'))}"))
    doc.kv(rows)


# =============================================================================================
# Strategy report
# =============================================================================================
def _pseudocode(spec: dict) -> str:
    p = spec["params"]
    lines = [f"# Universe: {spec['universe']}   Decision time: after each session's close (+ publication delay)"]
    if spec["strategy"] in ("threshold_event",):
        conds = " AND ".join(f"{c[0]} {c[1]} {c[2]}" for c in p["conditions"])
        lines += ["for each session t, for each symbol s in universe:",
                  f"    if {conds}:",
                  f"        BUY s at {'next open' if spec['execution'] == 'next_open' else 'next close'} (t+1)",
                  f"        SELL after {spec['holding_period']} sessions at the same time of day",
                  "    else: NO TRADE",
                  "    skip if already holding s, if the entry bar is missing, or if volume is zero"]
    elif spec["strategy"] == "quantile_event":
        lines += [f"every {p.get('rebalance_every', 1)} sessions:",
                  f"    rank universe by {p['feature']}",
                  f"    BUY the {p.get('side', 'top')} {p.get('quantile')} fraction at the next {spec['execution'].split('_')[1]}",
                  f"    hold {spec['holding_period']} sessions",
                  "    otherwise NO TRADE"]
    elif spec["strategy"] == "state_position":
        conds = " AND ".join(f"{c[0]} {c[1]} {c[2]}" for c in p["conditions"])
        lines += ["for each session t:",
                  f"    target = equal weight in symbols where {conds}; cash otherwise",
                  f"    rebalance to target at the {spec['execution'].replace('_', ' ')}"]
    elif spec["strategy"] == "segment_hold":
        lines += [f"every session: hold only the {p['segment']} segment (a full round trip each day)"]
    lines += ["position size: capital / max_concurrent slots (risk engine may reduce or veto)",
              "costs: spread + slippage + sqrt market impact + commission on every fill"]
    return "\n".join(lines)


def _strongest_argument_against(r: dict, test: dict | None) -> str:
    """Pick the evidence dimension with the thinnest margin; SIMULATED data dominates everything."""
    if r["data"]["label"] == "SIMULATED":
        return ("All evidence comes from SIMULATED data generated by our own synthetic model. It demonstrates that the "
                "pipeline can detect an effect that was deliberately planted; it says nothing about real markets.")
    cands = []
    dsr = r["deflated_sharpe"]["dsr"]
    if dsr is not None and np.isfinite(dsr):
        cands.append((dsr - 0.95, f"After accounting for {r['deflated_sharpe']['n_trials']} hypotheses tried, the probability "
                                  f"the Sharpe is real is {fmt(dsr)} — close to the 0.95 bar."))
    ts, vs = r["train"]["daily_net"].get("sharpe"), r["validation"]["daily_net"].get("sharpe")
    if ts and vs:
        cands.append(((vs / ts) - 0.5 if ts > 0 else -1, f"Sharpe fell from {fmt(ts)} in training to {fmt(vs)} in validation."))
    ys = r["stress"].get("max_single_year_share")
    if ys is not None:
        cands.append((0.4 - ys, f"One year produced {fmt(ys, 'pct')} of development P&L."))
    c2, tn = r["stress"].get("cost_x2_mean_net_trade"), r["train"]["trades_net"].get("mean")
    if c2 is not None and tn:
        cands.append((c2 / abs(tn) if tn else -1, f"At double the assumed costs the mean trade is {fmt(c2, 'pct')}; "
                                                  "cost assumptions are estimates, not measured fills."))
    wf = r["walk_forward"].get("positive_fold_frac")
    if wf is not None:
        cands.append((wf - 0.6, f"Only {fmt(wf, 'pct')} of walk-forward years were profitable."))
    st = r["stability"].get("assessment")
    if st in ("WEAKENED", "DISAPPEARED"):
        cands.append((-0.5, f"The edge {st.lower()} over the development period."))
    if test and test["payload"]["degradation"].get("ratio") is not None and test["payload"]["degradation"]["ratio"] < 0.5:
        cands.append((-0.4, f"Untouched-test Sharpe was only {fmt(test['payload']['degradation']['ratio'], 'pct')} of validation."))
    flags = r["data"]["flags"] + r.get("universe_flags", [])
    if "SURVIVORSHIP_RISK" in flags:
        cands.append((-0.3, "The universe excludes delisted securities (survivorship bias)."))
    if not cands:
        return "No quantitative weakness stood out; the main risk is that the future differs from the sample."
    return min(cands, key=lambda x: x[0])[1]


def _grade(value, strong, adequate, higher_better=True) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "not measured"
    v = value if higher_better else -value
    s, a = (strong, adequate) if higher_better else (-strong, -adequate)
    return "strong" if v >= s else "adequate" if v >= a else "weak"


def build_strategy_report(fb: FactBase, ref: str, mode: str = "plain") -> Doc:
    hid, sid = fb.resolve(ref)
    if hid is None:
        return Doc("Not found").h1(f"No hypothesis or strategy matches '{ref}'").p("No analysis has been performed on it.")
    h = fb.hypothesis(hid)
    spec = h["payload"]["spec"]
    ex = fb.development_experiment(hid)
    doc = Doc(f"Strategy report {sid or hid}").h1(f"Strategy Research Report — {sid or hid}: {spec['name']}")
    if ex is None:
        return doc.p("This hypothesis is registered but has not been evaluated. No results exist yet.")
    r = ex["payload"]
    if r.get("kind") == "measurement":
        return doc.p("This is a measurement hypothesis, not a trading strategy; see the research report.")
    rec = fb.signal(sid) or {}
    test = fb.test_experiment(hid)
    doc.p(f"Status: {rec.get('Status', 'n/a')} · hypothesis {hid} · development experiment {ex['id']}"
          + (f" · untouched test {test['id']}" if test else "") + f" · mode: {mode}")
    _banner(doc, fb)
    if rec.get("Status") in ("REJECTED", "EXPERIMENTAL"):
        doc.banner(f"This strategy is {rec.get('Status')}. It did NOT survive validation; this report documents why.", "bad")
    trades = fb.load_artifact(ex["id"], "trades")
    daily = fb.load_artifact(ex["id"], "daily")
    tn, tg, vn = r["train"]["trades_net"], r["train"]["trades_gross"], r["validation"]["trades_net"]

    doc.h2("1. What the strategy does")
    doc.p(spec["question"] + " " + spec["statement"])
    doc.h2("2. Hypothesis")
    doc.p(spec["statement"])
    doc.h2("3. Why it may exist")
    mech = spec["mechanism"]
    doc.ul([
        f"OBSERVED FACT: over training, {tn.get('n_trades', 0)} trades averaged {fmt(tn.get('mean'), 'pct')} net; over "
        f"validation, {vn.get('n_trades', 0)} trades averaged {fmt(vn.get('mean'), 'pct')} net [{ex['id']}].",
        f"STATISTICAL INFERENCE: primary p={fmt(r['train']['p_primary'], 'p')}, family-adjusted q="
        f"{fmt(r['multiple_testing']['q_value'], 'p')}, deflated Sharpe probability {fmt(r['deflated_sharpe']['dsr'])}.",
        ("HYPOTHESIZED EXPLANATION: We observe this statistical relationship but do not currently know why it exists."
         if mech == "UNKNOWN" else f"HYPOTHESIZED EXPLANATION ({mech.lower()}): {MECHANISM_TEXT.get(mech, '')}. {spec['rationale']}"),
        "Correlation is not causation: none of the above establishes why the pattern occurs.",
    ])
    doc.h2("4. Exact trade logic")
    doc.code(_pseudocode(spec))
    doc.p("NO TRADE conditions: the entry rule is false; the entry bar is missing or has zero volume; the symbol is "
          "already held; the order would exceed the participation cap; or the independent risk engine vetoes it.")

    doc.h2("5. Representative historical trade (an example, not proof)")
    if trades is not None and len(trades):
        med = trades["net_ret"].median()
        t = trades.iloc[(trades["net_ret"] - med).abs().argsort().iloc[0]]
        feats = {k[5:]: fmt(v) for k, v in t.items() if str(k).startswith("feat:")}
        regs = {k[7:]: v for k, v in t.items() if str(k).startswith("regime:")}
        doc.kv([("Instrument", t["symbol"]), ("Signal session", str(pd.Timestamp(t["signal_session"]).date())),
                ("Market conditions at signal", ", ".join(f"{k}={v}" for k, v in regs.items()) or "n/a"),
                ("Signal values", ", ".join(f"{k}={v}" for k, v in feats.items()) or "n/a"),
                ("Entry", f"{pd.Timestamp(t['entry_session']).date()} at raw price {fmt(t.get('entry_price_raw'))}"),
                ("Position size", f"${r['capital'] / r['max_concurrent']:,.0f} (capital/slots in the backtest)"),
                ("Expected return at entry (development mean)", fmt(tn.get("mean"), "pct")),
                ("Exit", str(pd.Timestamp(t["exit_session"]).date())),
                ("Actual result", f"gross {fmt(t['gross_ret'], 'pct')}, net {fmt(t['net_ret'], 'pct')}"),
                ("Why this trade", "its net return is the closest to the median trade — typical, not the best")])
    else:
        doc.p("No trades were generated.")

    doc.h2("6. Performance")
    dn_tr, dg_tr = r["train"]["daily_net"], r["train"]["daily_gross"]
    rows = []
    for label, part in (("TRAIN", "train"), ("VALIDATION", "validation")):
        for kind, key_t, key_d in (("GROSS", "trades_gross", "daily_gross"), ("NET", "trades_net", "daily_net")):
            t_, d_ = r[part][key_t], r[part][key_d]
            rows.append([label, kind, fmt(d_.get("cagr"), "pct"), fmt(d_.get("total_return"), "pct"), fmt(t_.get("mean"), "pct"),
                         fmt(t_.get("median"), "pct"), fmt(t_.get("win_rate"), "pct"), fmt(t_.get("avg_win"), "pct"),
                         fmt(t_.get("avg_loss"), "pct"), fmt(t_.get("profit_factor")), fmt(d_.get("sharpe")),
                         fmt(d_.get("sortino")), fmt(d_.get("ann_vol"), "pct"), fmt(d_.get("max_drawdown"), "pct"),
                         t_.get("n_trades", 0)])
    doc.table(["Period", "Basis", "CAGR", "Total", "Mean/trade", "Median", "Win rate", "Avg win", "Avg loss", "PF",
               "Sharpe", "Sortino", "Vol", "Max DD", "Trades"], rows)
    doc.p(f"Turnover (avg daily, position strategies): {fmt(r['strategy_meta'].get('avg_turnover'))}. "
          "Win rate is shown for completeness; it is not a selection criterion.")
    if daily is not None:
        doc.chart(charts.equity_curve(daily), "Growth of $1, gross vs net of estimated costs (development data only)")
        doc.chart(charts.drawdown(daily), "Net drawdown")
        doc.chart(charts.rolling_sharpe(daily), "Rolling one-year Sharpe")
        doc.chart(charts.yearly(daily), "Net return by year")
    if trades is not None and len(trades) >= 10:
        doc.chart(charts.rolling_trade_mean(trades), "Rolling expected return per trade")
        doc.chart(charts.trade_distribution(trades), "Per-trade return distribution")
        doc.chart(charts.win_loss(trades), "Winners vs losers")

    doc.h2("7. Statistical evidence")
    boot = r.get("dev_bootstrap_mean_net_trade") or {}
    ev = [
        f"Sample size: {tn.get('n_trades', 0)} training trades, effective independent observations "
        f"{r['train']['effective_n']} ({PLAIN['effective_n']}).",
        f"95% bootstrap interval for the mean net trade (train+validation): {fmt(boot.get('lo'), 'pct')} to {fmt(boot.get('hi'), 'pct')}. "
        "If it includes zero, the sign of the edge is uncertain.",
        f"Primary p-value: {fmt(r['train']['p_primary'], 'p')} — {PLAIN['p_value']}.",
        f"Adjusted for multiple testing (Benjamini–Yekutieli over {r['multiple_testing']['family_size']} hypotheses in "
        f"family '{r['multiple_testing']['family']}'): q={fmt(r['multiple_testing']['q_value'], 'p')}.",
        f"Deflated Sharpe probability: {fmt(r['deflated_sharpe']['dsr'])} ({PLAIN['dsr']}).",
    ]
    if "random_entry" in r["train"]:
        re_ = r["train"]["random_entry"]
        ev.append(f"Random-entry test: strategy mean gross trade {fmt(re_['statistic'], 'pct')} vs random-entry mean "
                  f"{fmt(re_.get('null_mean'), 'pct')}; p={fmt(re_['p_value'], 'p')} (does timing beat random timing?).")
    if "excess_vs_buy_and_hold" in r["train"]:
        eb = r["train"]["excess_vs_buy_and_hold"]
        ev.append(f"Versus buy-and-hold: annualised excess {fmt(eb.get('ann_excess'), 'pct')}, p={fmt(eb['p_value'], 'p')}; "
                  f"Sharpe {fmt(eb.get('sharpe_strategy'))} vs {fmt(eb.get('sharpe_buy_and_hold'))}.")
    doc.ul(ev)
    if mode == "quant":
        doc.h3("Acceptance criteria (frozen at registration)")
        for stage in ("screening", "validation_decision"):
            d = r.get(stage)
            if d:
                doc.table([f"{d['stage']} → {d['outcome']}", "Passed", "Value", "Threshold", "Meaning"],
                          [[x["name"], x["passed"], fmt(x["value"]) if isinstance(x["value"], (int, float)) else x["value"],
                            x["threshold"], x["explanation"]] for x in d["results"]])
        hac = r["train"]["hac_daily_net"]
        doc.p(f"HAC test on daily net returns (train): t={fmt(hac['statistic'])}, lags={hac.get('lags')}, "
              f"p={fmt(hac['p_value'], 'p')}. Information coefficient (train, first feature): {fmt(r.get('information_coefficient'))}.")

    doc.h2("8. Out-of-sample evidence (shown separately — never pooled)")
    wf = r["walk_forward"]
    oos_rows = [
        ["TRAINING", tn.get("n_trades", 0), fmt(tn.get("mean"), "pct"), fmt(dn_tr.get("sharpe")), ex["id"]],
        ["VALIDATION", vn.get("n_trades", 0), fmt(vn.get("mean"), "pct"), fmt(r["validation"]["daily_net"].get("sharpe")), ex["id"]],
    ]
    if test:
        tt = test["payload"]["test"]
        oos_rows.append(["UNTOUCHED TEST", tt["trades_net"].get("n_trades", 0), fmt(tt["trades_net"].get("mean"), "pct"),
                         fmt(tt["daily_net"].get("sharpe")), test["id"]])
    else:
        oos_rows.append(["UNTOUCHED TEST", "not run", "", "", "vault sealed"])
    oos_rows.append(["WALK-FORWARD (pooled)", sum(f.get("oos_n_trades", 0) for f in wf.get("folds", [])), "",
                     fmt(dig(wf, "pooled_oos.sharpe")), ex["id"]])
    oos_rows.append(["PAPER TRADING", "not started", "", "", ""])
    doc.table(["Stage", "Trades", "Mean net/trade", "Sharpe (net)", "Record"], oos_rows)
    if wf.get("folds"):
        doc.table(["Fold", "Test window", "Selected params", "OOS net return", "Trades"],
                  [[f["fold"], f"{f['test_start']}→{f['test_end']}", str(f["selected_params"])[:60],
                    fmt(f["oos_total_net_return"], "pct"), f["oos_n_trades"]] for f in wf["folds"]])

    doc.h2("9. Regime analysis")
    reg = r.get("regimes", {})
    for dim in ("trend", "volatility", "market_drawdown", "year"):
        cells = reg.get(dim, {})
        if cells:
            doc.table([dim, "Trades", "Mean net", "Win rate", "Note"],
                      [[k, v["n"], fmt(v["mean"], "pct"), fmt(v["win_rate"], "pct"), "small sample" if v["small_sample"] else ""]
                       for k, v in cells.items()])
            fig = charts.by_regime(reg, dim)
            if fig is not None and dim != "year":
                doc.chart(fig, f"Performance by {dim}")
    worst = sorted(((d, k, v["mean"]) for d, cells in reg.items() if d != "year" for k, v in cells.items() if not v["small_sample"]),
                   key=lambda x: x[2])[:2]
    if worst:
        doc.p("Weakest environments: " + "; ".join(f"{d}={k} (mean {fmt(m, 'pct')})" for d, k, m in worst) + ".")
    doc.p("Not available with current data: rising vs falling rates (no rates series loaded), recessions (ex-post NBER "
          "dates not loaded), decades before the data start.")

    doc.h2("10. Why this may not be real")
    flags = r["data"]["flags"] + r.get("universe_flags", [])
    doc.ul([
        f"Data mining / multiple testing: {r['multiple_testing']['family_size']} hypotheses in this family and "
        f"{r['deflated_sharpe']['n_trials']} overall were registered; the adjusted q and DSR above account for them.",
        "Overfitting: parameters were pre-declared; walk-forward re-selection is reported in section 8.",
        f"Survivorship / look-ahead: data flags = {', '.join(flags) or 'none'}; features passed truncation tests; fills at the next bar.",
        f"Transaction costs: modelled, not measured. At 2× costs mean trade = {fmt(r['stress'].get('cost_x2_mean_net_trade'), 'pct')}.",
        f"Outliers: without the best 5% of trades, mean = {fmt(r['stress'].get('trimmed_mean_net_trade'), 'pct')}.",
        f"Concentration: largest single-year share of P&L = {fmt(r['stress'].get('max_single_year_share'), 'pct')}.",
        f"Sample size: effective N = {r.get('dev_effective_n')} over development.",
        "Execution: fills assume the modelled spread/slippage/impact; no queue position or partial fills beyond the ADV cap.",
        f"Signal decay / crowding: stability assessment = {r['stability'].get('assessment')}.",
    ])
    doc.h3("STRONGEST ARGUMENT AGAINST THIS STRATEGY")
    doc.banner(_strongest_argument_against(r, test), "bad")

    doc.h2("11. Edge stability")
    st = r["stability"]
    doc.p(f"Assessment: {st.get('assessment')}. First-half mean {fmt(st.get('first_half_mean'), 'pct')}, second-half mean "
          f"{fmt(st.get('second_half_mean'), 'pct')}, trend {fmt(st.get('slope_per_year'), 'pct')} per year (p={fmt(st.get('slope_p'), 'p')}).")

    doc.h2("12. Monte Carlo")
    mc = r["monte_carlo"]
    if mc.get("status") == "OK":
        rows = []
        for cap, hres in mc["historical"].items():
            q = hres["ending_capital_quantiles"]
            rows.append([f"${int(cap):,}", fmt(q["p05"], "money"), fmt(q["p50"], "money"), fmt(q["p95"], "money"),
                         fmt(hres["max_drawdown_quantiles"]["p50"], "pct"), fmt(hres["p_loss"], "pct"),
                         fmt(hres["p_loss_25"], "pct"), fmt(hres["p_loss_50"], "pct"), fmt(hres["p_ruin"], "pct"),
                         fmt(hres["p_double"], "pct"), fmt(hres["avg_trades_skipped"])])
        doc.table(["Start", "5th pct", "Median", "95th pct", "Median max DD", "P(loss)", "P(−25%)", "P(−50%)", "P(ruin)",
                   "P(double)", "Trades skipped"], rows)
        doc.p(f"{mc['caveat']} Horizon: {mc['historical']['1000']['horizon_trades']} trades "
              f"(~{fmt(mc['historical']['1000']['horizon_years'])} years). The median, not the 95th percentile, is the "
              "central expectation. A stressed variant with the average edge halved is stored in "
              f"{ex['id']} (monte_carlo.stressed).")
        f1 = charts.mc_distribution(mc, "10000")
        if f1:
            doc.chart(f1, "Monte Carlo ending capital (start $10,000)")
        f2 = charts.mc_distribution(mc, "10000", "max_drawdown_sample", "Monte Carlo maximum drawdown (start $10,000)")
        if f2:
            doc.chart(f2, "Monte Carlo max drawdown distribution")
    else:
        doc.p(f"Monte Carlo not run: {mc.get('status')}.")

    doc.h2("13. Capacity")
    cap = r["capacity"]
    if cap.get("by_capital"):
        doc.table(["Account", "Position size", "Trades where 1 share > position", "Median participation of ADV", "Feasible"],
                  [[f"${int(k):,}", fmt(v["position_size"], "money"), fmt(v["frac_trades_unaffordable_whole_share"], "pct"),
                    fmt(v["median_participation"], "pct"), "yes" if v["feasible"] else "NO"] for k, v in cap["by_capital"].items()])
        doc.p(f"Estimated strategy capacity at the participation cap: {fmt(cap.get('strategy_capacity_estimate'), 'money')}; "
              f"~{fmt(cap.get('trades_per_year'))} trades/year. Options versions would need ≥1 contract (×100 multiplier), "
              "which small accounts often cannot afford.")
    else:
        doc.p("Capacity not estimated for this strategy type.")

    doc.h2("14. Evidence assessment (no composite score)")
    ts_, vs_ = dn_tr.get("sharpe"), r["validation"]["daily_net"].get("sharpe")
    doc.table(["Dimension", "Measured", "Reading"], [
        ["Statistical strength (BY q)", fmt(r["multiple_testing"]["q_value"], "p"), _grade(r["multiple_testing"]["q_value"], 0.01, 0.10, False)],
        ["Sample size (effective N)", r.get("dev_effective_n"), _grade(r.get("dev_effective_n"), 300, 100)],
        ["Out-of-sample consistency (val/train Sharpe)", fmt(vs_ / ts_ if ts_ and vs_ is not None else None),
         _grade(vs_ / ts_ if ts_ and vs_ is not None else None, 0.8, 0.5)],
        ["Walk-forward consistency", fmt(wf.get("positive_fold_frac"), "pct"), _grade(wf.get("positive_fold_frac"), 0.8, 0.6)],
        ["Regime robustness (worst non-small regime mean)", fmt(worst[0][2] if worst else None, "pct"),
         _grade(worst[0][2] if worst else None, 0.001, 0.0)],
        ["Cost sensitivity (mean at 2× costs)", fmt(r["stress"].get("cost_x2_mean_net_trade"), "pct"),
         _grade(r["stress"].get("cost_x2_mean_net_trade"), 0.002, 0.0)],
        ["Parameter sensitivity (neighbours positive)", fmt(r["stress"].get("param_neighbors_positive_frac"), "pct"),
         _grade(r["stress"].get("param_neighbors_positive_frac"), 1.0, 0.7)],
        ["Signal stability", r["stability"].get("assessment"), ""],
        ["Economic rationale", mech, "none stated" if mech == "UNKNOWN" else "hypothesized, unproven"],
    ])
    doc.p("If readings disagree, the evidence is mixed — and this report says so rather than averaging it away.")
    if mode == "quant":
        doc.h2("Appendix — model specification")
        doc.code(f"strategy={spec['strategy']}\nparams={spec['params']}\nneighbors={spec['param_neighbors']}\n"
                 f"holding={spec['holding_period']} execution={spec['execution']} cost_profile={spec['cost_profile']}\n"
                 f"cost_model={r['cost_model']}\nsplit={r['split_plan']}\nseed={r['seed']} code={r['code_version']}\n"
                 f"dataset_hashes={r['data']['dataset_hashes']}\nspec_hash={r['spec_hash']}")
    return doc


# =============================================================================================
# Persisting
# =============================================================================================
def save_report(reg: Registry, doc: Doc, kind: str, subject: str, out_dir: Path) -> dict:
    """Append the report to the registry (verbatim markdown) and write .md/.html files. Never overwrites."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md, page = doc.to_markdown(), doc.to_html()
    digest = hashlib.sha256(md.encode()).hexdigest()
    rid = reg.append("reports", {"kind": kind, "subject": subject}, {"title": doc.title, "markdown": md, "sha256": digest})
    stem = out_dir / f"{rid}_{kind}_{subject.replace('/', '_')}"
    stem.with_suffix(".md").write_text(md)
    stem.with_suffix(".html").write_text(page)
    _ensure_plotly(out_dir)
    return {"report_id": rid, "markdown": str(stem.with_suffix(".md")), "html": str(stem.with_suffix(".html"))}


def _ensure_plotly(out_dir: Path) -> None:
    target = out_dir / "plotly.min.js"
    if not target.exists():
        import plotly
        src = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if src.exists():
            target.write_bytes(src.read_bytes())
