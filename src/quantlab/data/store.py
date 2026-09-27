"""Versioned, write-once market-data store (Parquet files + JSON manifests, queried with DuckDB).

Layout::

    <store_dir>/<table>/<content_hash>.parquet
    <store_dir>/<table>/<content_hash>.manifest.json   provenance, validation report, row count
    <store_dir>/<table>/_versions.jsonl                append-only log of versions (name → hash)

* A dataset version is identified by its content hash; saving identical content is a no-op, and an
  existing version file is **never overwritten**.
* Experiments record the hash they used, so any result can be tied to the exact bytes it saw.
* Validation runs on every save. Datasets with validation errors are refused unless ``force=True``,
  in which case the manifest records that the save was forced.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from quantlab.data.validation import ValidationReport, validate_actions, validate_bars, validate_generic
from quantlab.provenance import Dataset, Provenance


class DataValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        super().__init__(report.summary())
        self.report = report


class DataStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------------------------
    def _validate(self, table: str, ds: Dataset, actions: pd.DataFrame | None) -> ValidationReport:
        if table == "bars_daily":
            return validate_bars(ds.frame, actions=actions)
        if table == "corporate_actions":
            return validate_actions(ds.frame)
        return validate_generic(ds.frame, table)

    def save(self, table: str, ds: Dataset, name: str | None = None, force: bool = False,
             actions: pd.DataFrame | None = None) -> str:
        """Validate and persist a dataset version. Returns its content hash."""
        report = self._validate(table, ds, actions)
        if not report.ok and not force:
            raise DataValidationError(report)
        h = ds.content_hash
        tdir = self.root / table
        tdir.mkdir(parents=True, exist_ok=True)
        pq = tdir / f"{h}.parquet"
        manifest = tdir / f"{h}.manifest.json"
        if not pq.exists():
            tmp = pq.with_suffix(".tmp")
            ds.frame.to_parquet(tmp, index=False)
            tmp.rename(pq)
            manifest.write_text(json.dumps({
                "table": table,
                "name": name or ds.name,
                "content_hash": h,
                "rows": int(len(ds.frame)),
                "provenance": ds.provenance.to_dict(),
                "validation": report.to_dict(),
                "forced": bool(force and not report.ok),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        with open(tdir / "_versions.jsonl", "a") as f:
            f.write(json.dumps({"name": name or ds.name, "content_hash": h,
                                "at": datetime.now(timezone.utc).isoformat()}) + "\n")
        return h

    def versions(self, table: str) -> list[dict]:
        log = self.root / table / "_versions.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    def latest_hash(self, table: str, name: str | None = None) -> str:
        vs = [v for v in self.versions(table) if name is None or v["name"] == name]
        if not vs:
            raise KeyError(f"no versions of {table}{'/' + name if name else ''} in store")
        return vs[-1]["content_hash"]

    def manifest(self, table: str, content_hash: str) -> dict:
        return json.loads((self.root / table / f"{content_hash}.manifest.json").read_text())

    def load(self, table: str, content_hash: str | None = None, name: str | None = None) -> Dataset:
        h = content_hash or self.latest_hash(table, name)
        man = self.manifest(table, h)
        df = pd.read_parquet(self.root / table / f"{h}.parquet")
        ds = Dataset(man["name"], df, Provenance.from_dict(man["provenance"]))
        if ds.content_hash != h:
            raise IOError(f"stored dataset {table}/{h} fails its integrity check (content changed on disk)")
        return ds

    def query(self, sql: str, **tables: str) -> pd.DataFrame:
        """Run DuckDB SQL; keyword args map view names to ``table`` or ``table@hash``."""
        con = duckdb.connect()
        try:
            for view, spec in tables.items():
                table, _, h = spec.partition("@")
                h = h or self.latest_hash(table)
                path = (self.root / table / f"{h}.parquet").as_posix()
                con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{path}')")
            return con.execute(sql).df()
        finally:
            con.close()
