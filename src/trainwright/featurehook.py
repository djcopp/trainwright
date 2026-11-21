"""Project feature engineering, done inside the framework.

Put a ``features.py`` next to ``trainwright.yaml``::

    import pandas as pd

    def fit(train: pd.DataFrame) -> dict:
        # learn statistics from the TRAIN split only; return JSON-serialisable state
        means = train.groupby("store_id")["basket_value"].mean()
        return {"store_mean": {str(k): float(v) for k, v in means.items()}}

    def transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:
        # must also work on a single row at inference time
        df = df.copy()
        df["store_mean"] = df["store_id"].astype(str).map(state["store_mean"])
        df["ratio"] = df["a"] / (df["b"] + 1e-9)
        return df

trainwright fits it on train only, applies it to train/val/test and to the baselines, records
its hash in every run, and ships the source + fitted state inside exported bundles so serving
applies exactly the same transform. (Tabular projects; ``fit`` is optional.)
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path
from typing import Any

FILENAME = "features.py"


class FeatureHook:
    def __init__(self, source: str, state: dict[str, Any] | None = None, fitted: bool = False):
        self.source = source
        self.state = state or {}
        self.fitted = fitted
        self._mod: types.ModuleType | None = None

    @classmethod
    def from_project(cls, root: Path) -> FeatureHook | None:
        path = root / FILENAME
        return cls(path.read_text()) if path.exists() else None

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:16]

    def _module(self) -> types.ModuleType:
        if self._mod is None:
            mod = types.ModuleType("tw_features")
            exec(compile(self.source, FILENAME, "exec"), mod.__dict__)
            if not hasattr(mod, "transform"):
                raise ValueError(f"{FILENAME} must define transform(df, state) -> df")
            self._mod = mod
        return self._mod

    def fit(self, train: Any) -> FeatureHook:
        mod = self._module()
        state = mod.fit(train.copy()) if hasattr(mod, "fit") else {}
        # round-trip through JSON so the state is guaranteed to survive export
        from ._utils import to_json

        self.state = json.loads(to_json(state or {}, indent=None))
        self.fitted = True
        return self

    def transform(self, df: Any) -> Any:
        out = self._module().transform(df.copy(), self.state)
        if out is None:
            raise ValueError(f"{FILENAME}: transform() returned None; return the DataFrame")
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"source": self.source, "state": self.state, "sha": self.sha}

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> FeatureHook:
        return cls(s["source"], s.get("state"), fitted=True)
