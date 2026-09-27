"""The research registry: a permanent, append-only, tamper-evident scientific record.

Stored in SQLite because it supports triggers: every table has ``BEFORE UPDATE`` and
``BEFORE DELETE`` triggers that abort, so results can never be overwritten and failed
experiments can never be deleted through the database. Additionally each row stores
``prev_hash``/``row_hash`` forming a per-table hash chain; :meth:`Registry.verify_chain` detects
any modification made by bypassing the triggers (e.g. editing the file by hand).

"State" (a hypothesis's status, a signal's latest record) is never stored by mutation: it is
derived from the latest appended event.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

GENESIS = "0" * 64

# table -> (id prefix, indexed columns)
TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "project_events": ("P", ("kind",)),
    "hypotheses": ("H", ("family", "name", "spec_hash", "generated_by")),
    "experiments": ("E", ("hypothesis_id", "kind", "status")),
    "journal": ("J", ("action", "experiment_id", "hypothesis_id")),
    "signal_records": ("SR", ("signal_id", "version", "status")),
    "status_events": ("S", ("entity_type", "entity_id", "to_status")),
    "vault_access": ("V", ("hypothesis_id", "experiment_id", "contaminated")),
    "reports": ("R", ("kind", "subject")),
    "paper_trades": ("PT", ("signal_id", "symbol")),
}


class AppendOnlyViolation(RuntimeError):
    pass


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if np.isnan(f) else f
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    raise TypeError(f"not JSON serializable: {type(o)}")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_json_default, allow_nan=False, separators=(",", ":"))


def _sanitize(obj: Any) -> Any:
    """Replace NaN/inf with None recursively so payloads are strict JSON."""
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def code_version(path: str | Path | None = None) -> dict:
    """Git commit and dirty state of the code that produced a result."""
    cwd = Path(path or Path(__file__).resolve().parent)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=cwd, capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}
    return {"commit": commit, "dirty": dirty}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Registry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path), isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL" if str(path) != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self._init_schema()

    # ---------------------------------------------------------------------------------------------
    def _init_schema(self) -> None:
        for table, (_, cols) in TABLES.items():
            extra = "".join(f", {c} TEXT" for c in cols)
            self._con.execute(
                f"CREATE TABLE IF NOT EXISTS {table} (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,"
                f" created_at TEXT NOT NULL{extra}, payload TEXT NOT NULL, prev_hash TEXT NOT NULL,"
                f" row_hash TEXT NOT NULL)")
            for c in cols:
                self._con.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_{c} ON {table}({c})")
            self._con.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table} "
                f"BEGIN SELECT RAISE(ABORT, 'registry is append-only: UPDATE forbidden on {table}'); END")
            self._con.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table} "
                f"BEGIN SELECT RAISE(ABORT, 'registry is append-only: DELETE forbidden on {table}'); END")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._con.execute("BEGIN IMMEDIATE")
        try:
            yield self._con
            self._con.execute("COMMIT")
        except BaseException:
            self._con.execute("ROLLBACK")
            raise

    @staticmethod
    def _row_hash(table: str, prev_hash: str, rid: str, created_at: str, cols: dict, payload_json: str) -> str:
        h = hashlib.sha256()
        h.update(canonical_json([table, prev_hash, rid, created_at, cols, payload_json]).encode())
        return h.hexdigest()

    def append(self, table: str, cols: dict[str, Any], payload: dict[str, Any]) -> str:
        """Append one row; returns its ID (e.g. ``H-000012``)."""
        prefix, allowed = TABLES[table]
        unknown = set(cols) - set(allowed)
        if unknown:
            raise ValueError(f"unknown indexed columns for {table}: {unknown}")
        cols = {c: (None if cols.get(c) is None else str(cols[c])) for c in allowed}
        payload_json = canonical_json(_sanitize(payload))
        with self._tx() as con:
            last = con.execute(f"SELECT seq, row_hash FROM {table} ORDER BY seq DESC LIMIT 1").fetchone()
            prev_hash = last["row_hash"] if last else GENESIS
            next_seq = (last["seq"] + 1) if last else 1
            rid = f"{prefix}-{next_seq:06d}"
            created_at = now_utc()
            row_hash = self._row_hash(table, prev_hash, rid, created_at, cols, payload_json)
            names = ["id", "created_at", *allowed, "payload", "prev_hash", "row_hash"]
            values = [rid, created_at, *[cols[c] for c in allowed], payload_json, prev_hash, row_hash]
            con.execute(f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})", values)
        return rid

    # ---------------------------------------------------------------------------------------------
    def _decode(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["payload"] = json.loads(d["payload"])
        return d

    def get(self, table: str, rid: str) -> dict:
        row = self._con.execute(f"SELECT * FROM {table} WHERE id = ?", (rid,)).fetchone()
        if row is None:
            raise KeyError(f"{table}: no record {rid}")
        return self._decode(row)

    def find(self, table: str, order: str = "ASC", limit: int | None = None, **where: Any) -> list[dict]:
        _, allowed = TABLES[table]
        clauses, params = [], []
        for k, v in where.items():
            if k not in allowed and k != "id":
                raise ValueError(f"cannot filter {table} on {k}")
            clauses.append(f"{k} = ?")
            params.append(str(v))
        sql = f"SELECT * FROM {table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY seq {'DESC' if order.upper() == 'DESC' else 'ASC'}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._decode(r) for r in self._con.execute(sql, params).fetchall()]

    def count(self, table: str, **where: Any) -> int:
        return len(self.find(table, **where))

    def verify_chain(self) -> dict[str, list[str]]:
        """Recompute every hash chain. Returns {table: [ids of broken rows]} (empty lists = intact)."""
        broken: dict[str, list[str]] = {}
        for table, (_, allowed) in TABLES.items():
            prev = GENESIS
            bad = []
            for row in self._con.execute(f"SELECT * FROM {table} ORDER BY seq").fetchall():
                cols = {c: row[c] for c in allowed}
                expect = self._row_hash(table, prev, row["id"], row["created_at"], cols, row["payload"])
                if row["prev_hash"] != prev or row["row_hash"] != expect:
                    bad.append(row["id"])
                prev = row["row_hash"]
            broken[table] = bad
        return broken

    # --- convenience writers -------------------------------------------------------------------
    def journal(self, action: str, question: str = "", hypothesis: str = "", reason: str = "", *,
                experiment_id: str | None = None, hypothesis_id: str | None = None, **details: Any) -> str:
        """Record a research action. Extra keyword details (dataset, features, model, parameters,
        results, conclusion, status, next_step, ...) are stored in the payload."""
        payload = {"question": question, "hypothesis": hypothesis, "reason": reason,
                   "code_version": code_version(), **details}
        return self.append("journal", {"action": action, "experiment_id": experiment_id,
                                       "hypothesis_id": hypothesis_id}, payload)

    def status_event(self, entity_type: str, entity_id: str, to_status: str, reason: str,
                     from_status: str | None = None, **details: Any) -> str:
        return self.append("status_events", {"entity_type": entity_type, "entity_id": entity_id,
                                             "to_status": to_status},
                           {"from_status": from_status, "reason": reason, **details})

    def current_status(self, entity_type: str, entity_id: str) -> str | None:
        rows = self.find("status_events", order="DESC", limit=1, entity_type=entity_type, entity_id=entity_id)
        return rows[0]["to_status"] if rows else None

    def close(self) -> None:
        self._con.close()
