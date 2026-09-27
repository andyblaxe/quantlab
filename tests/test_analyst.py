"""Research Analyst: reports and assistant answer only from stored records."""

import re

import pytest

from quantlab.analyst.assistant import ResearchAssistant
from quantlab.analyst.facts import FactBase
from quantlab.analyst.reports import NO_EDGE, build_research_report, build_strategy_report, save_report
from quantlab.data.providers.synthetic import SyntheticConfig, SyntheticMarket
from quantlab.research.artifacts import ArtifactStore
from quantlab.research.data import ResearchData
from quantlab.research.hypotheses import HypothesisSpec, Mechanism, register_hypothesis
from quantlab.research.pipeline import ResearchPipeline
from quantlab.research.registry import Registry
from quantlab.research.splits import SplitPlan, register_split_plan


def test_counts_reflect_registry(world):
    c = world["fb"].counts()
    assert c["hypotheses_registered"] == 3 and c["hypotheses_tested"] == 2
    assert c["untouched_tests_run"] == 1 and c["vault_openings"] == 1
    assert c["data_labels_used"] == ["SIMULATED"]


def test_research_report_is_honest_about_simulated_data(world):
    md = build_research_report(world["fb"], "plain").to_markdown()
    assert "SIMULATED DATA" in md
    assert NO_EDGE in md  # an accepted signal on simulated data is not a real-market edge
    assert world["good"] in md and "E-" in md  # record IDs are cited
    assert "winning strategy" not in md.lower()
    assert "Rejected ideas" in md and "queued" in md.lower()


def test_strategy_report_has_all_sections(world):
    doc = build_strategy_report(world["fb"], world["good"], "quant")
    md = doc.to_markdown()
    for i in range(1, 15):
        assert f"## {i}." in md, f"missing section {i}"
    assert "STRONGEST ARGUMENT AGAINST THIS STRATEGY" in md
    assert "SIMULATED" in md and "an example, not proof" in md
    assert "UNTOUCHED TEST" in md and "PAPER TRADING" in md
    html = doc.to_html()
    assert "<figure>" in html and "plotly" in html


def test_unknown_mechanism_is_stated_plainly(world):
    md = build_strategy_report(world["fb"], world["bad"]).to_markdown()
    assert "do not currently know why it exists" in md


def test_reports_are_saved_verbatim_and_never_overwritten(world, tmp_path):
    fb = world["fb"]
    a = save_report(world["reg"], build_research_report(fb), "research", "all", tmp_path)
    b = save_report(world["reg"], build_research_report(fb), "research", "all", tmp_path)
    assert a["report_id"] != b["report_id"]
    stored = world["reg"].get("reports", a["report_id"])["payload"]["markdown"]
    assert open(a["markdown"]).read() == stored
    # the second report knows about the first
    assert a["report_id"] in world["reg"].get("reports", b["report_id"])["payload"]["markdown"]


def test_assistant_answers_from_records(world):
    ra = ResearchAssistant(world["fb"])
    n = int(world["good"][2:])
    a = ra.ask(f"Explain strategy {n}")
    assert "## 1." in a.text and "## 4." in a.text
    assert "## 3." in ra.ask("Why does it appear to work?").text  # follow-up uses context
    assert "## 7." in ra.ask("How statistically convincing is it?").text
    assert "Appendix" in ra.ask("Now give me the mathematical explanation.").text
    assert "## 10." in ra.ask("What could cause it to stop working?").text
    assert "untouched test" in ra.ask("Which signals survived out-of-sample testing?").text.lower()
    assert re.search(r"S-\d{6}", ra.ask("What have you rejected?").text)
    assert NO_EDGE in ra.ask("What are the most promising strategies?").text
    assert world["queued"] in ra.ask("What are you researching next?").text


def test_assistant_admits_ignorance(world):
    ra = ResearchAssistant(world["fb"])
    assert "No analysis has been performed" in ra.ask("What has the system learned about earnings?").text
    assert "No analysis has been performed" in ra.ask("Explain strategy 999").text or "no strategy" in ra.ask("Explain strategy 999").text.lower()
    assert ra.ask("What is the weather tomorrow?").intent == "unknown"
    t = ra.ask("What has the system learned about Bollinger Bands?").text
    assert world["bad"] in t


def test_analyst_cannot_write_results(world):
    import quantlab.analyst.facts as facts
    writers = [n for n in dir(facts.FactBase) if any(w in n for w in ("append", "write", "record", "transition", "upsert"))]
    assert writers == []
