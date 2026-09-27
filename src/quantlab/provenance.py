"""Provenance labels: every dataset is REAL, DERIVED or SIMULATED — never ambiguous.

Rules
-----
* Data fetched from an external source as-is is ``REAL``.
* Anything computed from inputs is ``DERIVED`` — unless any input is ``SIMULATED``, in which case
  the output is ``SIMULATED``. Simulation is contagious; it can never be laundered into
  "derived" or "real" by further processing.
* Quality/bias flags (e.g. survivorship risk) propagate as a union to everything downstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

import pandas as pd


class DataLabel(StrEnum):
    REAL = "REAL"
    DERIVED = "DERIVED"
    SIMULATED = "SIMULATED"


class DataFlag(StrEnum):
    """Known biases/limitations that must follow the data into every result that uses it."""

    SURVIVORSHIP_RISK = "SURVIVORSHIP_RISK"  # universe excludes delisted securities
    NO_PIT_MEMBERSHIP = "NO_PIT_MEMBERSHIP"  # index membership is not point-in-time
    VENDOR_ADJUSTED = "VENDOR_ADJUSTED"  # prices adjusted by vendor (possibly revised history)
    PARTIAL_VOLUME = "PARTIAL_VOLUME"  # e.g. IEX-only volume
    MODEL_PRICED = "MODEL_PRICED"  # prices from a model, not observed quotes
    REVISABLE_NO_VINTAGE = "REVISABLE_NO_VINTAGE"  # revised series without vintage history
    UNVERIFIED_SOURCE = "UNVERIFIED_SOURCE"  # unofficial API / scraped
    # universe built from securities that exist today (e.g. current index members): delisted and
    # removed members are missing. Results are PRELIMINARY and cannot be promoted (catalog gate).
    SURVIVORSHIP_BIASED_UNIVERSE = "SURVIVORSHIP_BIASED_UNIVERSE"


def combine_labels(labels: Iterable[DataLabel]) -> DataLabel:
    """Label of data derived from inputs with the given labels."""
    labels = list(labels)
    if not labels:
        raise ValueError("derived data must have at least one input")
    if any(lbl == DataLabel.SIMULATED for lbl in labels):
        return DataLabel.SIMULATED
    return DataLabel.DERIVED


@dataclass(frozen=True)
class Provenance:
    label: DataLabel
    source: str
    flags: frozenset[DataFlag] = field(default_factory=frozenset)
    parents: tuple[str, ...] = ()  # content hashes of input datasets
    notes: str = ""

    @staticmethod
    def derive(source: str, inputs: Iterable["Provenance"], parents: Iterable[str] = (), notes: str = "") -> "Provenance":
        inputs = list(inputs)
        flags: frozenset[DataFlag] = frozenset().union(*(p.flags for p in inputs)) if inputs else frozenset()
        return Provenance(
            label=combine_labels(p.label for p in inputs),
            source=source,
            flags=flags,
            parents=tuple(parents),
            notes=notes,
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "source": self.source,
            "flags": sorted(f.value for f in self.flags),
            "parents": list(self.parents),
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "Provenance":
        return Provenance(
            label=DataLabel(d["label"]),
            source=d["source"],
            flags=frozenset(DataFlag(f) for f in d.get("flags", [])),
            parents=tuple(d.get("parents", [])),
            notes=d.get("notes", ""),
        )


def frame_hash(df: pd.DataFrame, extra: dict | None = None) -> str:
    """Deterministic content hash of a DataFrame (values, index, column names, dtypes).

    Used as the dataset version: identical content ⇒ identical hash, any change ⇒ new hash.
    """
    h = hashlib.sha256()
    h.update(json.dumps([str(c) for c in df.columns]).encode())
    h.update(json.dumps([str(t) for t in df.dtypes]).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    if extra:
        h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()


@dataclass
class Dataset:
    """A DataFrame that cannot be separated from its provenance."""

    name: str
    frame: pd.DataFrame
    provenance: Provenance

    @property
    def label(self) -> DataLabel:
        return self.provenance.label

    @property
    def content_hash(self) -> str:
        return frame_hash(self.frame, {"name": self.name, "provenance": self.provenance.to_dict()})
