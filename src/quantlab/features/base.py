"""Feature registry and the automatic look-ahead (truncation) test.

A feature maps a research panel (dict of wide session × symbol frames, see
:func:`quantlab.data.adjust.build_panel`) to a wide frame of values. The value in row *t* must use
only bars up to and including session *t*; it therefore becomes available at that bar's
``available_at`` (session close + publication delay). Trading on it can happen no earlier than the
next execution opportunity — the backtest engines enforce that separately.

The truncation test: for sampled dates t, compute the feature on the panel truncated at t and
compare with the full-history computation at t. Any difference means the feature used data after t
(e.g. centered windows, full-sample normalisation, back-adjusted levels, ``shift(-1)``).

Features are candidate *inputs*. Being in this library says nothing about predictive value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.data.pit import LookAheadError

Panel = dict[str, pd.DataFrame]


@dataclass(frozen=True)
class FeatureDef:
    name: str
    family: str
    params: dict[str, Any]
    fn: Callable[[Panel], pd.DataFrame]
    lookback: int
    category: str
    description: str
    inputs: tuple[str, ...] = ()

    def compute(self, panel: Panel) -> pd.DataFrame:
        out = self.fn(panel)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"feature {self.name} must return a DataFrame")
        return out.replace([np.inf, -np.inf], np.nan)


@dataclass
class FeatureFamily:
    family: str
    factory: Callable[..., Callable[[Panel], pd.DataFrame]]
    defaults: dict[str, Any]
    lookback: Callable[[dict[str, Any]], int]
    category: str
    description: str
    inputs: tuple[str, ...] = ()
    verified: dict[str, bool] = field(default_factory=dict)


FAMILIES: dict[str, FeatureFamily] = {}


def feature_family(family: str, *, defaults: dict[str, Any], lookback: Callable[[dict[str, Any]], int],
                   category: str, description: str, inputs: tuple[str, ...] = ("close",)):
    """Decorator registering a parameterised feature factory."""
    def deco(factory):
        if family in FAMILIES:
            raise ValueError(f"feature family {family} already registered")
        FAMILIES[family] = FeatureFamily(family, factory, defaults, lookback, category, description, inputs)
        return factory
    return deco


_SPEC = re.compile(r"^(?P<fam>[a-z_][a-z0-9_]*)(\((?P<args>.*)\))?$")


def _parse_value(v: str) -> Any:
    v = v.strip()
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v.strip("'\"")


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """``"sma_ratio(n=50)"`` → ``("sma_ratio", {"n": 50})``."""
    m = _SPEC.match(spec.replace(" ", ""))
    if not m:
        raise ValueError(f"bad feature spec {spec!r}")
    args = {}
    if m.group("args"):
        for part in m.group("args").split(","):
            k, _, v = part.partition("=")
            args[k] = _parse_value(v)
    return m.group("fam"), args


def canonical_name(family: str, params: dict[str, Any]) -> str:
    if not params:
        return family
    return f"{family}(" + ",".join(f"{k}={params[k]}" for k in sorted(params)) + ")"


def get_feature(spec: str) -> FeatureDef:
    fam_name, args = parse_spec(spec)
    if fam_name not in FAMILIES:
        raise KeyError(f"unknown feature family {fam_name!r}; known: {sorted(FAMILIES)}")
    fam = FAMILIES[fam_name]
    unknown = set(args) - set(fam.defaults)
    if unknown:
        raise ValueError(f"{fam_name}: unknown parameters {unknown}")
    params = {**fam.defaults, **args}
    return FeatureDef(canonical_name(fam_name, params), fam_name, params, fam.factory(**params),
                      fam.lookback(params), fam.category, fam.description, fam.inputs)


def compute_features(panel: Panel, specs: list[str]) -> dict[str, pd.DataFrame]:
    return {get_feature(s).name: get_feature(s).compute(panel) for s in specs}


def check_no_lookahead(fdef: FeatureDef, panel: Panel, n_checks: int = 8, seed: int = 0,
                       rtol: float = 1e-9, atol: float = 1e-12) -> list[pd.Timestamp]:
    """Return the dates at which truncated and full computations disagree (empty = passes)."""
    full = fdef.compute(panel)
    idx = full.index
    lb = max(fdef.lookback, 1)
    if len(idx) <= lb + 2:
        raise ValueError(f"panel too short to test {fdef.name} (lookback {lb})")
    rng = np.random.default_rng(seed)
    cand = np.arange(lb + 1, len(idx) - 1)
    picks = sorted(set(rng.choice(cand, size=min(n_checks, len(cand)), replace=False).tolist()) | {cand[-1]})
    failures = []
    for i in picks:
        t = idx[i]
        trunc = {k: v.loc[v.index <= t] for k, v in panel.items()}
        a = fdef.compute(trunc).loc[t]
        b = full.loc[t].reindex(a.index)
        both_nan = a.isna() & b.isna()
        close = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=rtol, atol=atol)
        if not (both_nan.to_numpy() | close).all():
            failures.append(t)
    return failures


def assert_no_lookahead(fdef: FeatureDef, panel: Panel, **kw) -> None:
    bad = check_no_lookahead(fdef, panel, **kw)
    if bad:
        raise LookAheadError(f"feature {fdef.name} uses future data: truncated values differ at {bad[:3]}")
