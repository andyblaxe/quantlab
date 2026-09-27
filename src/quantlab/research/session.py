"""Wiring: build a registry + data + pipeline for the real research store or the SIMULATED demo.

The demo lives under ``<data_dir>/demo`` with its own registry, so simulated results can never be
mixed into the real research record.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from quantlab.config import Settings, get_settings
from quantlab.data.providers.synthetic import SyntheticConfig, SyntheticMarket
from quantlab.data.store import DataStore
from quantlab.research.artifacts import ArtifactStore
from quantlab.research.data import ResearchData, universe_flags_from_kinds
from quantlab.research.pipeline import ResearchPipeline
from quantlab.research.program import MARKET_SYMBOL_V1, UNIVERSE_KINDS_V1, UNIVERSES_V1
from quantlab.research.registry import Registry
from quantlab.research.splits import SplitPlan, active_split_plan, register_split_plan

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def load_split_plan(path: Path | None = None) -> SplitPlan:
    with open(path or CONFIG_DIR / "splits.toml", "rb") as f:
        d = tomllib.load(f)["split"]
    return SplitPlan(train_end=d["train_end"], validation_end=d["validation_end"],
                     embargo_sessions=int(d.get("embargo_sessions", 21)))


@dataclass
class Workspace:
    root: Path
    registry: Registry
    artifacts: ArtifactStore
    simulated: bool

    @property
    def reports_dir(self) -> Path:
        d = self.root / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d


def open_workspace(demo: bool = False, settings: Settings | None = None) -> Workspace:
    s = settings or get_settings()
    root = Path(s.data_dir) / ("demo" if demo else "")
    root.mkdir(parents=True, exist_ok=True)
    return Workspace(root, Registry(root / "registry.sqlite"), ArtifactStore(root / "artifacts"), demo)


def ensure_split_plan(ws: Workspace, plan: SplitPlan, reason: str) -> SplitPlan:
    cur = active_split_plan(ws.registry)
    if cur is None:
        register_split_plan(ws.registry, plan, reason)
        return plan
    return cur


def demo_market(seed: int = 42) -> SyntheticMarket:
    return SyntheticMarket(SyntheticConfig(n_symbols=21, start="2000-01-01", end="2025-12-31", seed=seed,
                                           idio_ar1=-0.35))


def demo_data(market: SyntheticMarket) -> ResearchData:
    """SIMULATED stand-in for the v1 universes (a data swap only; research code is unchanged)."""
    S = market.symbols
    unis = {"spy": ["SYNMKT"], "us_index_etfs": ["SYNMKT"] + S[:3], "sector_spdrs": S[3:12],
            "multi_asset_etfs": ["SYNMKT"] + S[12:19], "synthetic_all": S}
    return ResearchData.from_datasets(market.bars(), market.corporate_actions(), unis, "SYNMKT", {"VIX": market.vix()})


DEMO_PLAN = SplitPlan("2012-12-31", "2018-12-31", embargo_sessions=21)


def real_data(settings: Settings | None = None) -> ResearchData:
    """Assemble ResearchData from the local store (latest versions). Raises if nothing is stored."""
    s = settings or get_settings()
    st = DataStore(s.store_dir)
    bars = st.load("bars_daily")
    try:
        actions = st.load("corporate_actions")
    except KeyError:
        actions = None
    macro = {}
    for key in ("VIX", "VIX3M"):
        try:
            macro[key] = st.load("macro_series", name=key)
        except KeyError:
            pass
    data = ResearchData.from_datasets(bars, actions, UNIVERSES_V1, MARKET_SYMBOL_V1, macro,
                                      universe_flags=universe_flags_from_kinds(UNIVERSES_V1, UNIVERSE_KINDS_V1))
    data.quality_issues = forced_validation_errors(st, [("bars_daily", bars), ("corporate_actions", actions),
                                                        *[("macro_series", m) for m in macro.values()]])
    return data


def forced_validation_errors(st: DataStore, datasets: list) -> list[dict]:
    """ERROR-level validation issues of datasets that were saved with force=True."""
    out = []
    for table, ds in datasets:
        if ds is None:
            continue
        man = st.manifest(table, ds.content_hash)
        if man.get("forced"):
            out += [{"table": table, "check": i["check"], "count": i.get("count")}
                    for i in man["validation"]["issues"] if i["severity"] == "ERROR"]
    return out


def make_pipeline(ws: Workspace, data: ResearchData, plan: SplitPlan, **kw) -> ResearchPipeline:
    return ResearchPipeline(ws.registry, data, plan, ws.artifacts, **kw)
