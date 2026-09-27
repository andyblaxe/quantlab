"""Natural-language Research Assistant over the platform's OWN research.

Deliberately not a language model. Questions are routed by keyword rules to fact queries; answers
are assembled from :class:`~quantlab.analyst.facts.FactBase` results with record IDs. If no rule
matches or no record exists, the assistant says so and lists what it *can* answer. An optional LLM
front-end could later rephrase these answers, subject to a validator that rejects any number or ID
not present in the fact bundle (see PLAN.md §16) — not implemented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from quantlab.analyst.facts import FactBase, dig, fmt
from quantlab.analyst.reports import NO_EDGE, _strongest_argument_against, build_research_report, build_strategy_report

HELP = ("I answer only from stored research records. Try: 'What have you been working on?', 'What have you discovered?', "
        "'Give me my research report', 'What changed since my last report?', 'What are the most promising strategies?', "
        "'What have you rejected?', 'Explain strategy 4', 'Why does strategy 4 appear to work?', 'How statistically "
        "convincing is it?', 'Show me the evidence', 'How would the trade actually work?', 'What could cause it to stop "
        "working?', 'What are you researching next?', 'What has the system learned about Bollinger Bands?', 'Which "
        "signals survived out-of-sample testing?', 'Which strategies are least correlated?', 'Now give me the "
        "mathematical explanation'.")

TOPIC_WORDS = {"bollinger": ["bollinger"], "moving average": ["sma", "moving", "ma_cross", "200"],
               "momentum": ["mom(", "momentum"], "reversal": ["reversal", "revert", "zret"], "rsi": ["rsi"],
               "earnings": ["earnings"], "volatility": ["vix", "volatility", "vrp"], "overnight": ["overnight"],
               "turn of the month": ["turn_of_month", "turn-of-month", "turn of the month"],
               "breakout": ["breakout"], "sector": ["sector"], "option": ["option", "straddle"]}


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    intent: str = ""


class ResearchAssistant:
    def __init__(self, fb: FactBase) -> None:
        self.fb = fb
        self.context_ref: str | None = None  # last strategy discussed (for follow-ups)

    # ------------------------------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        q = question.lower().strip()
        ref = self._extract_ref(q)
        if ref:
            self.context_ref = ref
        routes = [
            (r"(research )?report|full report", self._report),
            (r"changed since|since (my|the) last|what's new|what is new", self._changes),
            (r"working on|been doing|recent(ly)? activity", self._working_on),
            (r"researching next|next steps?\b|what.*\bnext\b|\bplans?\b", self._next),
            (r"\breject|\bfail(ed|ures?)?\b", self._rejected),
            (r"survived|out[- ]of[- ]sample|untouched", self._survived),
            (r"least correlated|correlat", self._correlated),
            (r"learn(ed|t)? about|know about|findings? on|tell me about", self._topic),
            (r"mathemat|quant mode|technical", self._math),
            (r"simply|plain english|simple terms", self._simple),
            (r"why does .* work|why .*appear|why would", self._why),
            (r"convincing|significan|how sure|confiden", self._convincing),
            (r"evidence", self._evidence),
            (r"trade actually work|how would .*trade|how does .* trade|trade logic", self._trade_logic),
            (r"stop working|could go wrong|risks?\b|break", self._stop_working),
            (r"what happened to", self._what_happened),
            (r"explain", self._explain),
            (r"promising|best|top|discover|found", self._promising),
        ]
        for pattern, fn in routes:
            if re.search(pattern, q):
                return fn(q)
        return Answer("I could not match that question to a stored analysis. " + HELP, intent="unknown")

    # ------------------------------------------------------------------------------------------
    def _extract_ref(self, q: str) -> str | None:
        m = re.search(r"(?:strategy|signal|sig-|hypothesis|h-)\s*#?\s*(\d+)", q)
        return m.group(1) if m else None

    def _need_ref(self) -> Answer | None:
        if self.context_ref is None:
            return Answer("Which strategy? Name one, e.g. 'strategy 4'. " + self._list_brief(), intent="clarify")
        hid, _ = self.fb.resolve(self.context_ref)
        if hid is None:
            return Answer(f"There is no strategy or hypothesis {self.context_ref} in the registry. No analysis has been "
                          "performed on it.", intent="not_found")
        return None

    def _list_brief(self) -> str:
        sigs = self.fb.ranked_signals()[:5]
        return ("Current candidates: " + ", ".join(f"{s['Signal_ID']} ({s['Status']})" for s in sigs)) if sigs else ""

    def _sim_note(self) -> str:
        return (" NOTE: all results so far come from SIMULATED data and are not evidence about real markets."
                if self.fb.uses_only_simulated_data() else "")

    # ------------------------------------------------------------------------------------------
    def _report(self, q):
        mode = "quant" if re.search(r"quant|math|technical", q) else "plain"
        return Answer(build_research_report(self.fb, mode).to_markdown(), ["registry"], "research_report")

    def _changes(self, q):
        last = self.fb.last_report("research")
        ch = self.fb.changes_since(last["created_at"] if last else None)
        head = f"Since report {last['id']} ({last['created_at']}):" if last else "No previous report exists; everything is new:"
        lines = [head, f"- new hypotheses: {len(ch['new_hypotheses'])}", f"- new experiments: {len(ch['new_experiments'])}"]
        lines += [f"- {x['entity']}: {x['from']} → {x['to']} ({x['reason']}) [{x['id']}]" for x in ch["status_changes"]]
        if ch["vault_openings"]:
            lines.append(f"- untouched-test openings: {', '.join(ch['vault_openings'])}")
        return Answer("\n".join(lines) + self._sim_note(), [x["id"] for x in ch["status_changes"]], "changes")

    def _working_on(self, q):
        j = self.fb.journal(15)
        if not j:
            return Answer("Nothing has been recorded in the research journal yet.", intent="working_on")
        lines = ["Most recent research actions (journal):"]
        for e in j:
            p = e["payload"]
            what = p.get("conclusion") or p.get("reason") or ""
            lines.append(f"- {e['created_at'][:19]} {e['action']} {e['hypothesis_id'] or ''} — {what[:140]} [{e['id']}]")
        return Answer("\n".join(lines) + self._sim_note(), [e["id"] for e in j], "working_on")

    def _next(self, q):
        queued = [h for h in self.fb.hypotheses() if h["status"] == "REGISTERED"]
        frozen = [h for h in self.fb.hypotheses() if h["status"] == "FROZEN"]
        lines = []
        if frozen:
            lines.append("Awaiting their one-time untouched test: " + ", ".join(f"{h['id']} {h['name']}" for h in frozen))
        if queued:
            lines.append("Registered and queued for evaluation: " + ", ".join(f"{h['id']} {h['name']}" for h in queued[:10]))
        blocked = [m for m in self.fb.measurements() if m["status_detail"] == "DATA_UNAVAILABLE"]
        if blocked:
            lines.append("Blocked on missing data: " + ", ".join(f"{m['hypothesis_id']} {m['name']}" for m in blocked))
        if self.fb.uses_only_simulated_data() or not self.fb.counts()["data_labels_used"]:
            lines.append("Real-data research (program v1) is pending data ingestion.")
        return Answer("\n".join(lines) or "Nothing is queued.", intent="next")

    def _rejected(self, q):
        rej = self.fb.rejected()
        if self.context_ref and re.search(r"why", q):
            hid, sid = self.fb.resolve(self.context_ref)
            r = next((x for x in rej if x["signal_id"] == sid), None)
            if r:
                return Answer(f"{sid} ({r['name']}) was rejected: {r['reason']} [{r['status_event']}]", [r["status_event"]], "why_rejected")
            return Answer(f"{sid or self.context_ref} has not been rejected (status: {(self.fb.signal(sid) or {}).get('Status', 'unknown')}).",
                          intent="why_rejected")
        if not rej:
            return Answer("No strategies have been rejected yet.", intent="rejected")
        lines = [f"{len(rej)} rejected (kept permanently on record):"]
        lines += [f"- {r['signal_id']} {r['name']}: {r['reason']} [{r['status_event']}]" for r in rej[:30]]
        meas = [m for m in self.fb.measurements() if m["outcome"] == "NOT_SUPPORTED"]
        lines += [f"- measurement {m['hypothesis_id']} {m['name']}: NOT_SUPPORTED [{m['experiment_id']}]" for m in meas]
        return Answer("\n".join(lines) + self._sim_note(), [r["status_event"] for r in rej], "rejected")

    def _survived(self, q):
        out = []
        for h in self.fb.hypotheses():
            t = self.fb.test_experiment(h["id"])
            if t:
                out.append(f"- {h['id']} {h['name']}: untouched test {t['payload']['decision']['outcome']} [{t['id']}]")
        if not out:
            return Answer("No hypothesis has reached the untouched out-of-sample test yet (none passed development "
                          "validation), so none has survived out-of-sample testing." + self._sim_note(), intent="survived")
        return Answer("Untouched-test results:\n" + "\n".join(out) + self._sim_note(), intent="survived")

    def _correlated(self, q):
        pairs = self.fb.least_correlated_pairs()
        if not pairs:
            return Answer("Fewer than two non-rejected strategies have stored return series, so correlations cannot be "
                          "computed yet.", intent="correlation")
        return Answer("Least correlated pairs (development daily net returns):\n" +
                      "\n".join(f"- {a} vs {b}: {c:+.2f}" for a, b, c in pairs) + self._sim_note(), intent="correlation")

    def _topic(self, q):
        terms = []
        for topic, words in TOPIC_WORDS.items():
            if topic in q or any(w in q for w in words if len(w) > 3):
                terms += words
        if not terms:
            m = re.search(r"about (.+?)\??$", q)
            terms = [m.group(1).strip()] if m else []
        hits = self.fb.search(terms)
        if not hits:
            return Answer(f"No analysis has been performed on '{' '.join(terms) or q}'. Nothing in the registry matches.",
                          intent="topic")
        lines = []
        for h in hits:
            sid = "SIG-" + h["id"][2:]
            s = self.fb.signal(sid)
            ex = self.fb.development_experiment(h["id"])
            if ex is None:
                lines.append(f"- {h['id']} {h['name']}: registered, not yet evaluated.")
            elif ex["payload"].get("kind") == "measurement":
                lines.append(f"- {h['id']} {h['name']}: measurement {ex['status']} [{ex['id']}]")
            else:
                tn = ex["payload"]["train"]["trades_net"]
                lines.append(f"- {h['id']} {h['name']}: {s['Status'] if s else ex['status']}; train mean net "
                             f"{fmt(tn.get('mean'), 'pct')} over {tn.get('n_trades', 0)} trades, q="
                             f"{fmt(ex['payload']['multiple_testing']['q_value'], 'p')} [{ex['id']}]")
        return Answer("What the registry records on this topic:\n" + "\n".join(lines) + self._sim_note(), intent="topic")

    def _promising(self, q):
        ranked = self.fb.ranked_signals()
        accepted = [s for s in ranked if s["Status"] in ("ACCEPTED", "PAPER_TRADING", "LIVE_ELIGIBLE")]
        head = "" if accepted and not self.fb.uses_only_simulated_data() else NO_EDGE + " "
        if not ranked:
            return Answer(head + "No strategy is currently under validation either." + self._sim_note(), intent="promising")
        lines = [head + "Non-rejected strategies, strongest evidence first:"]
        for s in ranked[:10]:
            lines.append(f"- {s['Signal_ID']} {s['Name']} — {s['Status']}; train mean net {fmt(s.get('Expected_Return'), 'pct')}, "
                         f"adjusted p {fmt(s.get('Adjusted_P_Value'), 'p')}, effective N {s.get('Effective_Sample_Size')}")
        return Answer("\n".join(lines) + self._sim_note(), [s["Signal_ID"] for s in ranked[:10]], "promising")

    def _what_happened(self, q):
        if self.context_ref is None:
            ranked = self.fb.ranked_signals()
            if not ranked:
                return Answer("There is no non-rejected strategy to follow up on." + self._sim_note(), intent="what_happened")
            self.context_ref = ranked[0]["Signal_ID"][4:]
        _, sid = self.fb.resolve(self.context_ref)
        hist = self.fb.signal_history(sid) if sid else []
        if not hist:
            return Answer("No status history recorded for that strategy.", intent="what_happened")
        return Answer(f"History of {sid}:\n" + "\n".join(
            f"- {e['created_at'][:19]}: → {e['to_status']} ({e['payload'].get('reason')}) [{e['id']}]" for e in hist),
            [e["id"] for e in hist], "what_happened")

    # --- strategy-specific ------------------------------------------------------------------------
    def _section(self, titles: list[str], mode: str = "plain", intent: str = "") -> Answer:
        miss = self._need_ref()
        if miss:
            return miss
        md = build_strategy_report(self.fb, self.context_ref, mode).to_markdown()
        parts = re.split(r"(?m)^## ", md)
        keep = [p for p in parts if any(p.startswith(t) for t in titles)]
        head = parts[0].strip().splitlines()[0] if parts else ""
        return Answer(head + "\n\n" + "\n\n".join("## " + k for k in keep), [self.context_ref], intent)

    def _explain(self, q):
        if self._extract_ref(q) is None and (self.context_ref is None or re.search(r"research|our|about", q)):
            return self._topic(q)
        return self._section(["1.", "2.", "3.", "4.", "6.", "14."], intent="explain")

    def _simple(self, q):
        return self._section(["1.", "3.", "5.", "10.", "14."], "plain", "plain")

    def _math(self, q):
        return self._section(["4.", "7.", "8.", "Appendix"], "quant", "quant")

    def _why(self, q):
        return self._section(["3."], intent="why")

    def _convincing(self, q):
        return self._section(["7.", "8.", "14."], intent="convincing")

    def _evidence(self, q):
        return self._section(["6.", "7.", "8.", "9."], intent="evidence")

    def _trade_logic(self, q):
        return self._section(["4.", "5."], intent="trade_logic")

    def _stop_working(self, q):
        miss = self._need_ref()
        if miss:
            return miss
        return self._section(["10.", "11."], intent="stop_working")
