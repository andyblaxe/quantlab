"""QuantLab command line. Every research action goes through here (and is journaled)."""

from __future__ import annotations

import json
from typing import Optional

import typer

from quantlab.config import get_settings

app = typer.Typer(help="QuantLab: local quantitative research laboratory (research only; no live trading).",
                  no_args_is_help=True)
data_app = typer.Typer(help="Fetch, validate and store market data.", no_args_is_help=True)
research_app = typer.Typer(help="Register and evaluate hypotheses.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(research_app, name="research")


def _ws(demo: bool):
    from quantlab.research.session import open_workspace
    return open_workspace(demo=demo)


# ---------------------------------------------------------------------------------------------
@app.command()
def init() -> None:
    """Create the data directories and register the project split plan (config/splits.toml)."""
    from quantlab.research.session import ensure_split_plan, load_split_plan
    ws = _ws(False)
    plan = ensure_split_plan(ws, load_split_plan(), "initial project split plan from config/splits.toml")
    get_settings().store_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"workspace: {ws.root.resolve()}\nsplit plan: {plan.to_dict()}")


# ---------------------------------------------------------------------------------------------
@data_app.command("fetch")
def data_fetch(provider: str = typer.Option("tiingo", help="tiingo | stooq"),
               symbols: Optional[str] = typer.Option(None, help="comma list; default = all v1 universe symbols"),
               start: str = "1993-01-01", end: str = typer.Option(None)) -> None:
    """Download daily bars (and corporate actions where the provider has them) into the store."""
    import pandas as pd

    from quantlab.data.providers.free import StooqProvider, TiingoProvider
    from quantlab.data.store import DataStore
    from quantlab.research.program import UNIVERSES_V1
    syms = symbols.split(",") if symbols else sorted({s for u in UNIVERSES_V1.values() for s in u})
    end = end or str(pd.Timestamp.today().date())
    p = {"tiingo": TiingoProvider, "stooq": StooqProvider}[provider]()
    st = DataStore(get_settings().store_dir)
    bars = p.get_daily_bars(syms, start, end)
    actions = p.get_corporate_actions(syms, start, end) if "get_corporate_actions" in p.capabilities() else None
    if actions is not None:
        typer.echo(f"corporate_actions: {st.save('corporate_actions', actions)}")
    h = st.save("bars_daily", bars, actions=actions.frame if actions is not None else None)
    typer.echo(f"bars_daily: {h} ({len(bars.frame)} rows, label={bars.label}, flags={sorted(bars.provenance.flags)})")


@data_app.command("fetch-macro")
def data_fetch_macro(series: str = typer.Option("VIX,VIX3M"), provider: str = "cboe",
                     start: str = "1990-01-01", end: str = typer.Option(None)) -> None:
    """Download index/macro series (Cboe VIX family by default; FRED with provider=fred)."""
    import pandas as pd

    from quantlab.data.providers.free import CboeIndexProvider, FredProvider
    from quantlab.data.store import DataStore
    end = end or str(pd.Timestamp.today().date())
    p = {"cboe": CboeIndexProvider, "fred": FredProvider}[provider]()
    st = DataStore(get_settings().store_dir)
    for sid in series.split(","):
        ds = p.get_macro_series(sid, start, end)
        typer.echo(f"{sid}: {st.save('macro_series', ds, name=sid)} ({len(ds.frame)} rows)")


@data_app.command("list")
def data_list() -> None:
    """List stored dataset versions."""
    from quantlab.data.store import DataStore
    st = DataStore(get_settings().store_dir)
    for table in sorted(p.name for p in st.root.iterdir() if p.is_dir()):
        for v in st.versions(table):
            m = st.manifest(table, v["content_hash"])
            typer.echo(f"{table:18s} {v['name']:12s} {v['content_hash'][:12]} rows={m['rows']:>8} "
                       f"label={m['provenance']['label']} warnings={len([i for i in m['validation']['issues'] if i['severity'] == 'WARNING'])}")


# ---------------------------------------------------------------------------------------------
@research_app.command("register-program")
def register_program(name: str = "v1", demo: bool = False) -> None:
    """Pre-register a research program's hypotheses (before evaluating any of them)."""
    from quantlab.research.hypotheses import DuplicateHypothesis, register_hypothesis
    from quantlab.research.program import PROGRAMS
    ws = _ws(demo)
    for spec in PROGRAMS[name]():
        try:
            typer.echo(f"{register_hypothesis(ws.registry, spec, reason=f'program {name}')}  {spec.name}")
        except DuplicateHypothesis as e:
            typer.echo(f"{e.existing_id}  (already registered) {spec.name}")


@research_app.command("generate")
def generate(template: str, universe: str, budget: int = 24, demo: bool = False) -> None:
    """Generate a budgeted batch from a template and register every hypothesis before evaluation."""
    from quantlab.research.generator import generate as gen, register_batch
    ws = _ws(demo)
    ids = register_batch(ws.registry, gen(template, universe, budget), f"generator {template} on {universe}")
    typer.echo(f"registered {len(ids)}: {', '.join(ids)}")


def _pipeline(demo: bool):
    from quantlab.research.session import (
        DEMO_PLAN, demo_data, demo_market, ensure_split_plan, load_split_plan, make_pipeline, real_data,
    )
    ws = _ws(demo)
    if demo:
        plan = ensure_split_plan(ws, DEMO_PLAN, "demo split plan (SIMULATED data)")
        return ws, make_pipeline(ws, demo_data(demo_market()), plan)
    plan = ensure_split_plan(ws, load_split_plan(), "initial project split plan")
    return ws, make_pipeline(ws, real_data(), plan)


@research_app.command("evaluate")
def evaluate(hypothesis_id: Optional[str] = typer.Argument(None), all_registered: bool = False,
             demo: bool = False) -> None:
    """Run development-stage evaluation (never touches the untouched test partition)."""
    from quantlab.research.hypotheses import HypothesisStatus, hypothesis_status
    ws, pipe = _pipeline(demo)
    ids = [hypothesis_id] if hypothesis_id else [
        h["id"] for h in ws.registry.find("hypotheses")
        if hypothesis_status(ws.registry, h["id"]) == HypothesisStatus.REGISTERED] if all_registered else []
    if not ids:
        raise typer.BadParameter("give a hypothesis id or --all-registered")
    for hid in ids:
        out = pipe.evaluate(hid)
        typer.echo(f"{hid} → {out['signal_status']}: {out['conclusion']}\n    next: {out['next_step']}")


@research_app.command("test")
def untouched_test(hypothesis_id: str, demo: bool = False) -> None:
    """Open the vault ONCE for a FROZEN hypothesis and evaluate it on untouched data."""
    ws, pipe = _pipeline(demo)
    out = pipe.run_untouched_test(hypothesis_id)
    typer.echo(f"{hypothesis_id} → {out['signal_status']}: {out['conclusion']}\n    next: {out['next_step']}")


@research_app.command("status")
def status(demo: bool = False) -> None:
    """Counts of hypotheses/experiments/signals by status."""
    from quantlab.analyst.facts import FactBase
    fb = FactBase(_ws(demo).registry)
    typer.echo(json.dumps(fb.counts(), indent=2, default=str))


# ---------------------------------------------------------------------------------------------
@app.command()
def demo(evaluate_generated: bool = True) -> None:
    """End-to-end run on SIMULATED data in a separate demo registry (machinery demonstration only)."""
    from quantlab.research.generator import generate as gen, register_batch
    from quantlab.research.hypotheses import DuplicateHypothesis, HypothesisStatus, hypothesis_status, register_hypothesis
    from quantlab.research.program import program_v1
    ws, pipe = _pipeline(True)
    typer.secho("DEMO — all data SIMULATED; results are machinery checks, not findings.", fg="red", bold=True)
    for spec in program_v1():
        try:
            register_hypothesis(ws.registry, spec, reason="demo program v1")
        except DuplicateHypothesis:
            pass
    if evaluate_generated:
        register_batch(ws.registry, gen("extreme_move_reversal", "synthetic_all", 24), "demo generator batch")
    for h in ws.registry.find("hypotheses"):
        if hypothesis_status(ws.registry, h["id"]) == HypothesisStatus.REGISTERED:
            out = pipe.evaluate(h["id"])
            typer.echo(f"{h['id']} {out['signal_status']:13s} {h['name'][:60]}")
            if out["signal_status"] == "VALIDATING" and "run_untouched_test" in out["next_step"]:
                t = pipe.run_untouched_test(h["id"])
                typer.echo(f"         untouched test → {t['signal_status']}")
    typer.echo(f"demo registry: {ws.root / 'registry.sqlite'}")


def _fb(demo: bool):
    from quantlab.analyst.facts import FactBase
    ws = _ws(demo)
    return ws, FactBase(ws.registry, ws.artifacts)


@app.command()
def report(mode: str = typer.Option("plain", help="plain | quant"), demo: bool = False) -> None:
    """Generate and save the Research Report (includes what changed since the last one)."""
    from quantlab.analyst.reports import build_research_report, save_report
    ws, fb = _fb(demo)
    out = save_report(ws.registry, build_research_report(fb, mode), "research", mode, ws.reports_dir)
    typer.echo(json.dumps(out, indent=2))


@app.command("strategy-report")
def strategy_report(ref: str, mode: str = typer.Option("plain", help="plain | quant"), demo: bool = False) -> None:
    """Generate and save the Strategy Research Report for a strategy/hypothesis (e.g. 4, SIG-000004)."""
    from quantlab.analyst.reports import build_strategy_report, save_report
    ws, fb = _fb(demo)
    out = save_report(ws.registry, build_strategy_report(fb, ref, mode), "strategy", f"{ref}_{mode}", ws.reports_dir)
    typer.echo(json.dumps(out, indent=2))


@app.command()
def ask(question: str, demo: bool = False) -> None:
    """Ask the Research Assistant about the platform's own research (answers cite record IDs)."""
    from quantlab.analyst.assistant import ResearchAssistant
    _, fb = _fb(demo)
    typer.echo(ResearchAssistant(fb).ask(question).text)


@app.command()
def dashboard(demo: bool = False, port: int = 8765) -> None:
    """Serve the local dashboard on 127.0.0.1 (read-only views over the research record)."""
    import uvicorn

    from quantlab.dashboard.app import create_app
    uvicorn.run(create_app(demo=demo), host="127.0.0.1", port=port, log_level="warning")


@app.command()
def calibrate(n_null: int = 10, n_power: int = 3, demo_dir: bool = True) -> None:
    """Measure the pipeline's false-positive rate (null markets) and power (planted effect)."""
    from quantlab.research.calibration import run_calibration
    ws = _ws(True)
    res = run_calibration(ws, n_null=n_null, n_power=n_power)
    typer.echo(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    app()
