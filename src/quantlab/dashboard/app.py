"""Local research dashboard (FastAPI, server-rendered, bound to 127.0.0.1).

Every page is a *view* over the registry, artifacts and data store through the analyst fact layer.
The only write the dashboard can perform is appending a generated report (the analyst's one
permitted write). It cannot run experiments, change statuses or trade.

Run: ``quantlab dashboard`` (real workspace) or ``quantlab dashboard --demo`` (SIMULATED demo).
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from quantlab.analyst import charts
from quantlab.analyst.assistant import ResearchAssistant
from quantlab.analyst.doc import REPORT_CSS, Doc
from quantlab.analyst.facts import FactBase, dig, fmt
from quantlab.analyst.reports import build_research_report, build_strategy_report, save_report
from quantlab.montecarlo.simulate import DEFAULT_CAPITALS
from quantlab.options.chain import atm_term_structure, enrich_chain, simulated_chain, skew_25d
from quantlab.options.pricing import bs_greeks, bs_price, expected_move_from_iv
from quantlab.portfolio.combine import SignalInput, combine_signals
from quantlab.research.session import Workspace, demo_data, demo_market, open_workspace, real_data
from quantlab.risk.gate import RiskLimits
from quantlab.risk.metrics import historical_var_es

NAV = [("/", "Overview"), ("/portfolio", "Portfolio"), ("/strategies", "Strategies"), ("/signals", "Signals"),
       ("/opportunities", "Opportunities"), ("/research", "Research"), ("/options", "Options"), ("/risk", "Risk"),
       ("/montecarlo", "Monte Carlo"), ("/reports", "Reports"), ("/assistant", "Assistant")]

EXTRA_CSS = """
nav{display:flex;flex-wrap:wrap;gap:4px 14px;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
nav a{color:var(--muted);text-decoration:none;font-size:.92rem}nav a.on{color:var(--fg);font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.8rem;border:1px solid var(--line)}
form.inline{display:flex;flex-wrap:wrap;gap:8px;align-items:end;margin:8px 0}
input,select,textarea,button{font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:6px;padding:5px 8px}
button{cursor:pointer;font-weight:600}label{font-size:.85rem;color:var(--muted);display:flex;flex-direction:column;gap:2px}
.answer{border-left:3px solid var(--line);padding:4px 14px;margin:10px 0}
"""


def md_to_html(md: str) -> str:
    """Minimal Markdown renderer for assistant answers (escapes everything first)."""
    out, in_list, table = [], False, []

    def flush_table():
        nonlocal table
        if table:
            rows = [r for r in table if not re.fullmatch(r"\|?(-+\|)+-*\|?", r.replace(" ", ""))]
            cells = [[html.escape(c.strip()) for c in r.strip("|").split("|")] for r in rows]
            head = "".join(f"<th>{c}</th>" for c in cells[0])
            body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells[1:])
            out.append(f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
            table = []

    for line in md.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{html.escape(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        m = re.match(r"^(#{1,3}) (.*)", line)
        if m:
            n = len(m.group(1)) + 1
            out.append(f"<h{n}>{html.escape(m.group(2))}</h{n}>")
        elif line.startswith("> "):
            out.append(f'<div class="banner">{html.escape(line[2:].strip("*"))}</div>')
        elif line.strip() == "```":
            continue
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    flush_table()
    if in_list:
        out.append("</ul>")
    return "".join(out)


def create_app(demo: bool = False, workspace: Workspace | None = None) -> FastAPI:
    ws = workspace or open_workspace(demo=demo)
    fb = FactBase(ws.registry, ws.artifacts)
    app = FastAPI(title="QuantLab dashboard", docs_url=None, redoc_url=None)
    plotly_js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    assistant = ResearchAssistant(fb)

    @lru_cache(maxsize=1)
    def data():
        return demo_data(demo_market()) if ws.simulated else real_data()

    def page(active: str, doc: Doc) -> HTMLResponse:
        body, _ = doc.body_html()
        nav = "".join(f'<a href="{u}" class="{"on" if u == active else ""}">{t}</a>' for u, t in NAV)
        label = ('<div class="banner bad">SIMULATED DEMO WORKSPACE — machinery demonstration, not market evidence.</div>'
                 if ws.simulated else "")
        return HTMLResponse(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>QuantLab — {html.escape(doc.title)}</title>
<script src="/static/plotly.min.js"></script><style>{REPORT_CSS}{EXTRA_CSS}</style></head>
<body><nav>{nav}</nav><main>{label}{body}</main></body></html>""")

    @app.get("/static/plotly.min.js")
    def _plotly():
        return FileResponse(plotly_js, media_type="application/javascript")

    # ---------------------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def overview():
        doc = build_research_report(fb, "plain")
        doc.title = "Overview"
        return page("/", doc)

    def _live_signal_returns() -> tuple[pd.DataFrame, list]:
        series, inputs = {}, []
        for s in fb.ranked_signals():
            eid = (s.get("Experiment_IDs") or [None])[0]
            df = fb.load_artifact(eid, "daily") if eid else None
            if df is None or s["Status"] not in ("ACCEPTED", "PAPER_TRADING", "LIVE_ELIGIBLE"):
                continue
            series[s["Signal_ID"]] = df["net"]
            n = df["net"].replace(0, np.nan).dropna()
            inputs.append(SignalInput(s["Signal_ID"], float(df["net"].mean()), float(df["net"].std() / np.sqrt(len(df))),
                                      s.get("Signal_Decay")))
        return pd.DataFrame(series), inputs

    @app.get("/portfolio", response_class=HTMLResponse)
    def portfolio():
        doc = Doc("Portfolio").h1("Portfolio (simulated combination of ACCEPTED signals)")
        R, inputs = _live_signal_returns()
        if R.empty:
            doc.banner("No ACCEPTED signals — the portfolio holds cash. NO TRADE is the correct position.", "warn")
            return page("/portfolio", doc)
        comb = combine_signals(inputs, R)
        if comb["status"] != "OK":
            doc.banner(f"{comb['status']}: {comb.get('reason', '')}", "warn")
            return page("/portfolio", doc)
        w = pd.Series(comb["weights"])
        daily = pd.DataFrame({"net": (R[w.index] * w).sum(axis=1)})
        doc.p("Development-period returns of the accepted signals, combined with correlation-aware weights "
              "(Σ⁻¹μ on shrunk covariance). This is a historical simulation, not live performance.")
        doc.kv([("Weights", ", ".join(f"{k}: {v:.0%}" for k, v in comb["weights"].items())),
                ("Effective independent signals", fmt(comb["effective_independent_signals"])),
                ("Annualised volatility", fmt(daily["net"].std() * np.sqrt(252), "pct")),
                ("Max drawdown", fmt(float(((1 + daily['net']).cumprod() / (1 + daily['net']).cumprod().cummax() - 1).min()), "pct"))])
        doc.chart(charts.equity_curve(daily, "Simulated equity curve (net)"), "")
        doc.chart(charts.drawdown(daily), "")
        pnl = go.Figure(go.Bar(x=daily.index[-250:], y=daily["net"].iloc[-250:], marker_color=charts.BLUE,
                               hovertemplate="%{y:.2%}<extra></extra>"))
        doc.chart(charts._style(pnl, "Daily P&L (last 250 sessions, % of equity)", "", ".1%"), "")
        return page("/portfolio", doc)

    @app.get("/strategies", response_class=HTMLResponse)
    def strategies():
        doc = Doc("Strategies").h1("Strategies — research results")
        rows = []
        for s in sorted(fb.signals(), key=lambda x: x["Signal_ID"]):
            rows.append([s["Signal_ID"], s.get("Name", "")[:60], s["Status"], fmt(s.get("Sharpe")), fmt(s.get("Expected_Return"), "pct"),
                         fmt(s.get("Max_Drawdown"), "pct"), s.get("Sample_Size"), s.get("Effective_Sample_Size"),
                         fmt(s.get("Estimated_Transaction_Cost"), "pct"), fmt(s.get("Adjusted_P_Value"), "p"),
                         s.get("Data_Label")])
        doc.table(["Signal", "Name", "Status", "Sharpe (train, net)", "Mean/trade (net)", "Max DD", "Trades", "Eff. N",
                   "Cost/trade", "Adj. p", "Data"], rows)
        links = " · ".join(f'<a href="/strategy/{html.escape(s["Signal_ID"])}">{html.escape(s["Signal_ID"])}</a>'
                           for s in sorted(fb.signals(), key=lambda x: x["Signal_ID"]))
        doc.h3("Strategy reports").raw_html(f"<p>{links}</p>")
        return page("/strategies", doc)

    @app.get("/strategy/{ref}", response_class=HTMLResponse)
    def strategy(ref: str, mode: str = "plain"):
        if mode not in ("plain", "quant"):
            raise HTTPException(400, "mode must be plain or quant")
        doc = build_strategy_report(fb, ref, mode)
        other = "quant" if mode == "plain" else "plain"
        doc.blocks.insert(1, Doc("x").raw_html(
            f'<p><a href="/strategy/{html.escape(ref)}?mode={other}">Switch to {other.upper()} MODE</a></p>').blocks[0])
        return page("/strategies", doc)

    @app.get("/signals", response_class=HTMLResponse)
    def signals():
        doc = Doc("Signals").h1("Signals — strength, uncertainty, reliability")
        rows = []
        for s in fb.ranked_signals():
            hid = s.get("Hypothesis_ID")
            ex = fb.development_experiment(hid)
            boot = dig(ex["payload"], "dev_bootstrap_mean_net_trade") if ex else None
            rows.append([s["Signal_ID"], s.get("Name", "")[:50], s["Status"], fmt(s.get("Expected_Return"), "pct"),
                         f"{fmt((boot or {}).get('lo'), 'pct')} … {fmt((boot or {}).get('hi'), 'pct')}",
                         fmt(s.get("Information_Coefficient")), fmt(s.get("Adjusted_P_Value"), "p"),
                         fmt(dig(ex["payload"], "walk_forward.positive_fold_frac") if ex else None, "pct"), s.get("Signal_Decay")])
        if not rows:
            doc.banner("No non-rejected signals.", "warn")
        doc.table(["Signal", "Name", "Status", "Expected value/trade", "95% CI", "IC", "Adj. p", "WF positive", "Stability"], rows)
        return page("/signals", doc)

    @app.get("/opportunities", response_class=HTMLResponse)
    def opportunities(equity: float = 10_000.0):
        from quantlab.research.opportunities import current_opportunities
        doc = Doc("Opportunities").h1(f"Opportunities on the latest session (account ${equity:,.0f})")
        try:
            ops = current_opportunities(fb, data(), equity)
        except (KeyError, FileNotFoundError) as e:
            doc.banner(f"No data available to evaluate current signals: {e}", "warn")
            return page("/opportunities", doc)
        live = [o for o in ops if o["action"].startswith("BUY")]
        if not live:
            doc.banner("NO TRADE: no surviving signal's entry conditions hold on the latest session.", "ok")
        doc.table(["Signal", "Status", "Symbol", "Action", "Qty", "Expected value", "Predicted 5%…95%", "Downside (5%)",
                   "Reason", "Risk"],
                  [[o["signal_id"], o["status"], o["symbol"], o["action"], fmt(o.get("quantity")),
                    fmt(o.get("expected_value"), "pct"),
                    f"{fmt(o.get('distribution', {}).get('p05'), 'pct')} … {fmt(o.get('distribution', {}).get('p95'), 'pct')}",
                    fmt(o.get("downside_p05"), "pct"), o.get("reason", ""), o.get("risk_decision", o.get("why", ""))] for o in ops])
        return page("/opportunities", doc)

    @app.get("/research", response_class=HTMLResponse)
    def research():
        doc = Doc("Research").h1("Research — experiments, hypotheses, journal")
        hyps = fb.hypotheses()
        doc.h2("Hypotheses")
        doc.table(["ID", "Name", "Family", "Status", "Signal status", "Generated by"],
                  [[h["id"], h["name"][:70], h["family"], h["status"], (fb.signal("SIG-" + h["id"][2:]) or {}).get("Status", ""),
                    h["generated_by"]] for h in hyps])
        doc.h2("Experiments")
        rows = []
        for e in fb.reg.find("experiments"):
            if e["kind"] == "artifacts":
                continue
            p = e["payload"]
            rows.append([e["id"], e["hypothesis_id"], e["kind"], e["status"],
                         fmt(dig(p, "train.p_primary") if p.get("kind") != "measurement" else dig(p, "train.p_value"), "p"),
                         fmt(dig(p, "multiple_testing.q_value"), "p"), dig(p, "data.label"), e["created_at"][:19]])
        doc.table(["Experiment", "Hypothesis", "Stage", "Outcome", "p", "BY q", "Data", "When"], rows)
        doc.h2("Research journal (latest 40)")
        doc.table(["Entry", "When", "Action", "Hypothesis", "Conclusion / reason"],
                  [[j["id"], j["created_at"][:19], j["action"], j["hypothesis_id"] or "",
                    (j["payload"].get("conclusion") or j["payload"].get("reason") or "")[:140]] for j in fb.journal(40)])
        chain = fb.reg.verify_chain()
        broken = {k: v for k, v in chain.items() if v}
        doc.banner("Registry hash chain intact." if not broken else f"REGISTRY TAMPERING DETECTED: {broken}",
                   "ok" if not broken else "bad")
        return page("/research", doc)

    @app.get("/options", response_class=HTMLResponse)
    def options(S: float = 100.0, K: float = 100.0, days: float = 30.0, r: float = 0.04, q: float = 0.0, iv: float = 0.20):
        doc = Doc("Options").h1("Options analytics")
        T = days / 365
        form = f"""<form class="inline" method="get">
<label>Spot<input name="S" value="{S}" size="6"></label><label>Strike<input name="K" value="{K}" size="6"></label>
<label>Days<input name="days" value="{days}" size="4"></label><label>Rate<input name="r" value="{r}" size="4"></label>
<label>Div yield<input name="q" value="{q}" size="4"></label><label>IV<input name="iv" value="{iv}" size="4"></label>
<button>Price</button></form>"""
        doc.raw_html(form)
        rows = []
        for right in ("C", "P"):
            g = bs_greeks(S, K, T, r, q, iv, right)
            rows.append([right, fmt(float(bs_price(S, K, T, r, q, iv, right))), fmt(float(g["delta"])), fmt(float(g["gamma"])),
                         fmt(float(g["vega"]) / 100), fmt(float(g["theta"]) / 365)])
        doc.table(["Right", "BS price", "Delta", "Gamma", "Vega / vol pt", "Theta / day"], rows)
        em = expected_move_from_iv(S, iv, T)
        doc.p(f"Expected move to expiry: ±{fmt(em['one_sd_move'])} (1 s.d., {fmt(em['one_sd_pct'], 'pct')}); expected absolute "
              f"move {fmt(em['expected_abs_move'])}. Black–Scholes is a baseline model, not a forecast.")
        doc.h2("Example chain analytics (SIMULATED, model-priced)")
        doc.banner("No historical option quotes are loaded (paid data required). The chain below is generated by a model "
                   "with skew, term structure and an earnings bump; it demonstrates the analytics only.", "warn")
        ch = enrich_chain(simulated_chain(S, "2024-03-01", r=r, base_iv=iv, skew=-0.15, event_move=0.05,
                                          event_before="2024-03-10"), r=r)
        ts = atm_term_structure(ch)
        f1 = go.Figure(go.Scatter(x=ts["T"] * 365, y=ts["atm_iv"], mode="lines+markers", line={"color": charts.BLUE, "width": 2},
                                  marker={"size": 8}, hovertemplate="%{x:.0f}d: %{y:.1%}<extra></extra>"))
        doc.chart(charts._style(f1, "ATM implied volatility term structure", "IV", ".0%"), "Earnings variance lifts the expiry after the event")
        front = ch[(ch["expiration"] == ch["expiration"].unique()[2]) & ch["iv_reliable"]].sort_values("strike")
        f2 = go.Figure(go.Scatter(x=front["strike"], y=front["iv_mid"], mode="markers", marker={"color": charts.BLUE, "size": 8},
                                  hovertemplate="K=%{x}: %{y:.1%}<extra></extra>"))
        doc.chart(charts._style(f2, "Volatility smile (30-day expiry, reliable OTM quotes)", "IV", ".0%"), "")
        sk = skew_25d(ch)
        doc.table(["Expiry (days)", "25Δ put IV", "25Δ call IV", "Risk reversal"],
                  [[f"{r_['T'] * 365:.0f}", fmt(r_["iv_put25"], "pct"), fmt(r_["iv_call25"], "pct"),
                    fmt(r_["risk_reversal_25d"], "pct")] for _, r_ in sk.iterrows()])
        return page("/options", doc)

    @app.get("/risk", response_class=HTMLResponse)
    def risk():
        doc = Doc("Risk").h1("Risk")
        R, inputs = _live_signal_returns()
        if not R.empty:
            port = R.mean(axis=1)
            h = historical_var_es(port)
            eq = (1 + port).cumprod()
            doc.kv([("1-day VaR95 (historical, equal-weight accepted signals)", fmt(h["var"], "pct")),
                    ("1-day expected shortfall 95%", fmt(h["es"], "pct")),
                    ("Max drawdown (development period)", fmt(float((eq / eq.cummax() - 1).min()), "pct"))])
        else:
            doc.banner("No accepted signals: no portfolio risk to measure (all cash).", "warn")
        doc.h2("RiskGate limits (independent of signals)")
        doc.table(["Limit", "Value"], [[k, v] for k, v in RiskLimits().to_dict().items()])
        doc.h2("Probability of ruin by strategy (Monte Carlo, development trades)")
        rows = []
        for s in fb.ranked_signals():
            ex = fb.development_experiment(s.get("Hypothesis_ID"))
            mc = dig(ex["payload"], "monte_carlo") if ex else None
            if mc and mc.get("status") == "OK":
                rows.append([s["Signal_ID"], s["Status"]] + [fmt(mc["historical"][str(int(c))]["p_ruin"], "pct") for c in DEFAULT_CAPITALS])
        doc.table(["Signal", "Status"] + [f"P(ruin) ${int(c):,}" for c in DEFAULT_CAPITALS], rows)
        return page("/risk", doc)

    @app.get("/montecarlo", response_class=HTMLResponse)
    def montecarlo(signal: str | None = None, capital: str = "10000"):
        doc = Doc("Monte Carlo").h1("Monte Carlo")
        cands = [s for s in fb.signals() if s["Status"] != "REJECTED"] or fb.signals()
        if not cands:
            doc.p("No evaluated signals.")
            return page("/montecarlo", doc)
        sid = signal or cands[0]["Signal_ID"]
        opts = "".join(f'<option value="{html.escape(s["Signal_ID"])}" {"selected" if s["Signal_ID"] == sid else ""}>'
                       f'{html.escape(s["Signal_ID"])} ({html.escape(s["Status"])})</option>' for s in fb.signals())
        caps = "".join(f'<option {"selected" if str(int(c)) == capital else ""}>{int(c)}</option>' for c in DEFAULT_CAPITALS)
        doc.raw_html(f'<form class="inline" method="get"><label>Signal<select name="signal">{opts}</select></label>'
                     f'<label>Start capital<select name="capital">{caps}</select></label><button>Show</button></form>')
        s = fb.signal(sid)
        ex = fb.development_experiment(s["Hypothesis_ID"])
        mc = dig(ex["payload"], "monte_carlo") or {}
        if mc.get("status") != "OK":
            doc.p(f"Monte Carlo not available: {mc.get('status')}")
            return page("/montecarlo", doc)
        doc.p(mc["caveat"])
        for variant in ("historical", "stressed"):
            h = mc[variant][capital]
            q = h["ending_capital_quantiles"]
            doc.h3(f"{variant.title()} — start ${int(capital):,}, {h['horizon_trades']} trades (~{h['horizon_years']:.1f} years)")
            doc.kv([("Ending capital 5% / 25% / median / 75% / 95%", " / ".join(fmt(q[k], "money") for k in ("p05", "p25", "p50", "p75", "p95"))),
                    ("CAGR median (5%–95%)", f"{fmt(h['cagr_quantiles']['p50'], 'pct')} ({fmt(h['cagr_quantiles']['p05'], 'pct')} – {fmt(h['cagr_quantiles']['p95'], 'pct')})"),
                    ("Max drawdown median (worst 5%)", f"{fmt(h['max_drawdown_quantiles']['p50'], 'pct')} ({fmt(h['max_drawdown_quantiles']['p05'], 'pct')})"),
                    ("P(loss) / P(−10%) / P(−25%) / P(−50%)", " / ".join(fmt(h[k], "pct") for k in ("p_loss", "p_loss_10", "p_loss_25", "p_loss_50"))),
                    ("P(ruin) / P(double)", f"{fmt(h['p_ruin'], 'pct')} / {fmt(h['p_double'], 'pct')}"),
                    ("Time to double (years, 25/50/75%)", " / ".join(fmt(v) for v in (h["time_to_double_years_quantiles"] or {}).values()) or "never within horizon"),
                    ("Trades skipped (below minimum size)", fmt(h["avg_trades_skipped"]))])
        f = charts.mc_distribution(mc, capital)
        if f:
            doc.chart(f, "")
        f2 = charts.mc_distribution(mc, capital, "max_drawdown_sample", "Maximum drawdown distribution")
        if f2:
            doc.chart(f2, "")
        return page("/montecarlo", doc)

    @app.get("/reports", response_class=HTMLResponse)
    def reports():
        doc = Doc("Reports").h1("Reports")
        doc.raw_html('<form method="post" action="/reports/generate" class="inline"><label>Mode<select name="mode">'
                     '<option>plain</option><option>quant</option></select></label><button>Generate research report now</button></form>')
        rows = [[f'{r["id"]}', r["kind"], r["subject"], r["created_at"][:19]] for r in fb.reg.find("reports", order="DESC")]
        doc.table(["Report", "Kind", "Subject", "Created"], rows)
        links = " · ".join(f'<a href="/reports/{html.escape(r[0])}">{html.escape(r[0])}</a>' for r in rows)
        doc.raw_html(f"<p>{links}</p>")
        return page("/reports", doc)

    @app.post("/reports/generate")
    def generate(mode: str = Form("plain")):
        if mode not in ("plain", "quant"):
            raise HTTPException(400, "bad mode")
        out = save_report(ws.registry, build_research_report(fb, mode), "research", mode, ws.reports_dir)
        return RedirectResponse(f"/reports/{out['report_id']}", status_code=303)

    @app.get("/reports/{rid}", response_class=HTMLResponse)
    def report(rid: str):
        try:
            r = fb.reg.get("reports", rid)
        except KeyError:
            raise HTTPException(404, "no such report")
        doc = Doc(r["payload"]["title"]).raw_html(md_to_html(r["payload"]["markdown"]))
        return page("/reports", doc)

    @app.get("/assistant", response_class=HTMLResponse)
    def assistant_get():
        doc = Doc("Assistant").h1("Research Assistant")
        doc.p("Answers come only from stored research records and cite their IDs. It cannot speculate.")
        doc.raw_html(_ask_form(""))
        return page("/assistant", doc)

    @app.post("/assistant", response_class=HTMLResponse)
    def assistant_post(question: str = Form(...)):
        doc = Doc("Assistant").h1("Research Assistant")
        doc.raw_html(_ask_form(question))
        ans = assistant.ask(question)
        doc.raw_html(f'<div class="answer"><p><strong>Q:</strong> {html.escape(question)}</p>{md_to_html(ans.text)}</div>')
        return page("/assistant", doc)

    def _ask_form(q: str) -> str:
        return (f'<form method="post" action="/assistant" class="inline"><label style="flex:1">Question'
                f'<input name="question" value="{html.escape(q)}" style="width:100%"></label><button>Ask</button></form>')

    return app
