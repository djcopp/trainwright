"""Standalone inference runtime for exported bundles.

Depends only on numpy + onnxruntime (+ Pillow for images). This file and
``preprocessing.py`` are copied into every bundle as the ``tw_runtime`` package, so a
deployed model needs neither torch nor trainwright::

    from tw_runtime.runtime import Predictor
    p = Predictor("artifacts/007-mlp")
    p.predict([{"age": 41, "plan": "pro"}])
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import (
    ImagePreprocessor,
    TabularPreprocessor,
    TargetScaler,
    TextTokenizer,
)


class Predictor:
    def __init__(self, bundle_dir: str | Path, threads: int | None = None):
        self.dir = Path(bundle_dir)
        self.meta: dict[str, Any] = json.loads((self.dir / "bundle.json").read_text())
        self.modality: str = self.meta["modality"]
        self.task: str = self.meta["task"]
        self.classes: list[str] | None = self.meta.get("classes")
        self.kind: str = self.meta.get("kind", "onnx")
        pre = self.meta["preprocessor"]
        feats = pre["features"]
        self.target = TargetScaler.from_state(pre["target"]) if pre.get("target") else None
        if self.kind == "sklearn":
            import joblib

            self.model = joblib.load(self.dir / "model.joblib")["model"]
            return
        if self.modality == "tabular":
            self.pre: Any = TabularPreprocessor.from_state(feats)
        elif self.modality == "text":
            self.pre = TextTokenizer.from_state(feats)
        else:
            self.pre = ImagePreprocessor.from_state(feats)
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    # ------------------------------------------------------------------ inputs
    def featurise(self, instances: list[Any]) -> dict[str, np.ndarray]:
        if self.modality == "tabular":
            cols = self.pre.numeric_columns + self.pre.categorical_columns
            table = {
                c: [row.get(c) if isinstance(row, dict) else None for row in instances]
                for c in cols
            }
            feats = self.pre.transform(table)
        elif self.modality == "text":
            feats = {"tokens": self.pre.encode([str(t) for t in instances])}
        else:
            feats = self.pre.transform([_decode_image_input(x) for x in instances])
        return {k: v for k, v in feats.items() if k in self.input_names}

    def predict_raw(self, instances: list[Any]) -> np.ndarray:
        if self.kind == "sklearn":
            return self._sklearn_raw(instances)
        return self.session.run(None, self.featurise(instances))[0]

    def predict(self, instances: list[Any], top_k: int = 3) -> list[dict[str, Any]]:
        out = self.predict_raw(instances)
        if self.task == "classification":
            assert self.classes is not None
            results = []
            for p in out:
                order = np.argsort(-p)[:top_k]
                results.append(
                    {
                        "label": self.classes[int(order[0])],
                        "confidence": float(p[order[0]]),
                        "top": [{"label": self.classes[int(i)], "p": float(p[i])} for i in order],
                    }
                )
            return results
        vals = out.reshape(-1)
        if self.target is not None and self.kind != "sklearn":
            vals = self.target.decode(vals)
        return [{"prediction": float(v)} for v in vals]

    def _sklearn_raw(self, instances: list[Any]) -> np.ndarray:
        if self.modality == "tabular":
            import pandas as pd

            num, cat = self.meta["columns"]
            df = pd.DataFrame([{c: row.get(c) for c in num + cat} for row in instances])
            for c in cat:
                df[c] = df[c].astype(object).where(df[c].notna(), "<NA>").astype(str)
            X: Any = df
        elif self.modality == "text":
            X = [str(t) for t in instances]
        else:
            raise NotImplementedError("sklearn bundles are not supported for images")
        if self.meta.get("baseline") == "dummy":
            X = np.zeros((len(instances), 1))
        if self.task == "classification":
            p = self.model.predict_proba(X)
            full = np.zeros((len(instances), len(self.classes or [])))
            full[:, np.asarray(self.model.classes_, dtype=int)] = p
            return full
        return np.asarray(self.model.predict(X))


def _decode_image_input(x: Any) -> Any:
    """Accept a path, raw bytes, a PIL image, or a base64 string (optionally a data URL)."""
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        if x.startswith("data:"):
            x = x.split(",", 1)[1]
        p = Path(x)
        if len(x) < 1024 and p.exists():
            return p
        return base64.b64decode(x)
    return x
