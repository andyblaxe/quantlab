"""Write-once storage for experiment artifacts (trade lists, daily return series).

The registry stores summary results; bulky series live here as Parquet, referenced from the
registry by path and content hash, so charts and reports are rebuilt from exactly the data that
produced the recorded numbers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, experiment_id: str, name: str, df: pd.DataFrame | pd.Series) -> dict:
        if isinstance(df, pd.Series):
            df = df.to_frame(name=df.name or "value")
        d = self.root / experiment_id
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.parquet"
        if path.exists():
            raise FileExistsError(f"artifact {path} already exists (artifacts are write-once)")
        df.to_parquet(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"path": str(path.relative_to(self.root)), "sha256": digest, "rows": int(len(df))}

    def load(self, ref: dict) -> pd.DataFrame:
        path = self.root / ref["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise IOError(f"artifact {path} fails its integrity check")
        return pd.read_parquet(path)
