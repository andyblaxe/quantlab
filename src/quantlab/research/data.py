"""ResearchData: the panel a pipeline runs on, bound to its provenance and dataset versions."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantlab.data.adjust import build_panel
from quantlab.data.pit import asof_series
from quantlab.provenance import DataFlag, DataLabel, Dataset, Provenance


@dataclass
class ResearchData:
    panel: dict[str, pd.DataFrame]
    provenance: Provenance
    dataset_hashes: dict[str, str]
    universes: dict[str, list[str]]
    market_symbol: str
    # macro / index series aligned to sessions, point-in-time: the value known when each session's
    # bar became available (asof join on available_at). Keys e.g. "VIX", "VIX3M", "DGS10".
    series: dict[str, pd.Series] = field(default_factory=dict)
    universe_flags: dict[str, list[str]] = field(default_factory=dict)
    vix_key: str = "VIX"
    # validation ERRORs of datasets that were force-saved ({table, check, count}); each one is a
    # research-integrity finding on every result that uses this data (research/integrity.py)
    quality_issues: list[dict] = field(default_factory=list)

    @property
    def label(self) -> DataLabel:
        return self.provenance.label

    @property
    def vix(self) -> pd.Series | None:
        return self.series.get(self.vix_key)

    def describe(self) -> dict:
        idx = self.panel["close"].index
        return {"label": self.label.value, "flags": sorted(f.value for f in self.provenance.flags),
                "source": self.provenance.source, "dataset_hashes": self.dataset_hashes,
                "start": str(idx.min().date()), "end": str(idx.max().date()), "n_sessions": int(len(idx)),
                "symbols": list(self.panel["close"].columns), "market_symbol": self.market_symbol,
                "series": sorted(self.series)}

    @staticmethod
    def from_datasets(bars: Dataset, actions: Dataset | None, universes: dict[str, list[str]], market_symbol: str,
                      macro: dict[str, Dataset] | None = None, universe_flags: dict[str, list[str]] | None = None,
                      vix_key: str = "VIX") -> "ResearchData":
        macro = macro or {}
        panel = build_panel(bars.frame, actions.frame if actions is not None else None)
        inputs = [bars.provenance] + ([actions.provenance] if actions is not None else []) + \
                 [m.provenance for m in macro.values()]
        hashes = {"bars_daily": bars.content_hash}
        if actions is not None:
            hashes["corporate_actions"] = actions.content_hash
        prov = Provenance.derive("research_panel", inputs, parents=list(hashes.values()))
        decision_times = pd.DatetimeIndex(panel["available_at"][market_symbol])
        series = {}
        for key, ds in macro.items():
            hashes[f"macro:{key}"] = ds.content_hash
            v = asof_series(ds.frame, decision_times)
            series[key] = pd.Series(v["value"].to_numpy(), index=panel["close"].index, name=key)
        return ResearchData(panel, prov, hashes, universes, market_symbol, series, universe_flags or {}, vix_key)

    @staticmethod
    def from_synthetic(market, universes: dict[str, list[str]] | None = None) -> "ResearchData":
        syms = market.symbols
        universes = universes or {"synthetic_all": syms, "synthetic_market": [market.cfg.market_symbol]}
        return ResearchData.from_datasets(market.bars(), market.corporate_actions(), universes,
                                          market.cfg.market_symbol, {"SYN_VIX": market.vix()}, vix_key="SYN_VIX")


def universe_flags_for(symbols_are_single_stocks: bool, has_pit_membership: bool) -> list[str]:
    flags = []
    if symbols_are_single_stocks and not has_pit_membership:
        flags += [DataFlag.SURVIVORSHIP_RISK.value, DataFlag.NO_PIT_MEMBERSHIP.value,
                  DataFlag.SURVIVORSHIP_BIASED_UNIVERSE.value]
    return flags


def universe_flags_from_kinds(universes: dict[str, list[str]], kinds: dict[str, str]) -> dict[str, list[str]]:
    """Flags per universe from its declared kind. Fail-safe: an undeclared universe is treated as a
    current-members stock universe (survivorship-biased) until someone declares otherwise.

    Kinds: ``"etf"`` (fixed ETF set that existed throughout), ``"stock_pit"`` (point-in-time
    membership with delisted securities), ``"stock_current"`` (today's members only).
    """
    out = {}
    for name in universes:
        kind = kinds.get(name, "stock_current")
        if kind not in ("etf", "stock_pit", "stock_current"):
            raise ValueError(f"unknown universe kind {kind!r} for {name}")
        out[name] = universe_flags_for(kind != "etf", kind == "stock_pit")
    return out
