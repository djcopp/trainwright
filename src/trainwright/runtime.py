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
    text_preprocessor_from_state,
)


class Predictor:
    def __init__(self, bundle_dir: str | Path, threads: int | None = None):
        self.dir = Path(bundle_dir)
        self.meta: dict[str, Any] = json.loads((self.dir / "bundle.json").read_text())
        self.modality: str = self.meta["modality"]
        self.task: str = self.meta["task"]
        self.classes: list[str] | None = self.meta.get("classes")
        self.kind: str = self.meta.get("kind", "onnx")
        if self.kind == "ensemble":
            self.members = [Predictor(self.dir / m, threads) for m in self.meta["members"]]
            self.weights = np.asarray(self.meta["weights"], dtype=np.float64)
            self.weights = self.weights / self.weights.sum()
            self.target = None
            return
        pre = self.meta["preprocessor"]
        feats = pre["features"]
        self.target = TargetScaler.from_state(pre["target"]) if pre.get("target") else None
        self.hook: Any = None
        if pre.get("feature_hook"):
            from .featurehook import FeatureHook

            self.hook = FeatureHook.from_state(pre["feature_hook"])
        if self.kind == "sklearn":
            import joblib

            self.model = joblib.load(self.dir / "model.joblib")["model"]
            return
        if self.modality == "tabular":
            self.pre: Any = TabularPreprocessor.from_state(feats)
        elif self.modality == "text":
            self.pre = text_preprocessor_from_state(feats)
        else:
            self.pre = ImagePreprocessor.from_state(feats)
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.dir / "model.onnx"), opts, providers=_providers(ort)
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    # ------------------------------------------------------------------ inputs
    def _apply_hook(self, instances: list[Any]) -> list[dict[str, Any]]:
        rows = [row if isinstance(row, dict) else {} for row in instances]
        texts = self.meta.get("text_features") or {}
        if texts and rows:
            from .preprocessing import expand_text_features

            cols = {c: [r.get(c) for r in rows] for c in texts}
            ext = {k: v for k, v in expand_text_features(cols, texts).items() if k not in cols}
            rows = [{**r, **{k: float(v[i]) for k, v in ext.items()}} for i, r in enumerate(rows)]
        dates = self.meta.get("date_columns") or {}
        if dates and rows:
            from .preprocessing import expand_dates

            cols = {c: [r.get(c) for r in rows] for c in dates}
            ext = {k: v for k, v in expand_dates(cols, dates).items() if k not in cols}
            rows = [{**r, **{k: float(v[i]) for k, v in ext.items()}} for i, r in enumerate(rows)]
        if self.hook is None:
            return rows
        import pandas as pd

        df = self.hook.transform(pd.DataFrame(rows))
        return df.astype(object).where(df.notna(), None).to_dict(orient="records")

    def featurise(self, instances: list[Any]) -> dict[str, np.ndarray]:
        if self.modality == "tabular":
            cols = self.pre.numeric_columns + self.pre.categorical_columns
            rows = self._apply_hook(instances)
            table = {c: [row.get(c) for row in rows] for c in cols}
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

    def predict_values(self, instances: list[Any]) -> np.ndarray:
        """Probabilities [N, K] (classification) or target-scale predictions [N] (regression)."""
        if self.kind == "ensemble":
            return sum(w * m.predict_values(instances) for w, m in zip(self.weights, self.members))
        out = self.predict_raw(instances)
        if self.task == "classification":
            return np.asarray(out, dtype=np.float64)
        vals = np.asarray(out, dtype=np.float64).reshape(-1)
        if self.target is not None and self.kind != "sklearn":
            vals = self.target.decode(vals)
        return vals

    def predict_masks(self, instances: list[Any]) -> list[np.ndarray]:
        """Segmentation: one uint8 class-index mask per image, at the image's own size."""
        from PIL import Image

        logits = np.asarray(self.predict_raw(instances))
        masks = []
        for x, lg in zip(instances, logits):
            src = _decode_image_input(x)
            size = (src if isinstance(src, Image.Image) else Image.open(src)).size
            masks.append(
                _resize_logits(np.asarray(lg, dtype=np.float32), size).argmax(0).astype(np.uint8)
            )
        return masks

    def predict(self, instances: list[Any], top_k: int = 3) -> list[dict[str, Any]]:
        if self.task == "segmentation":
            out_seg = []
            for m in self.predict_masks(instances):
                vals, counts = np.unique(m, return_counts=True)
                share = {
                    (self.classes[int(v)] if self.classes else str(int(v))): round(c / m.size, 4)
                    for v, c in zip(vals, counts)
                }
                out_seg.append({"mask_shape": list(m.shape), "pixel_share": share})
            return out_seg
        out = self.predict_values(instances)
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
        return [{"prediction": float(v)} for v in out.reshape(-1)]

    def _sklearn_raw(self, instances: list[Any]) -> np.ndarray:
        if self.modality == "tabular":
            import pandas as pd

            num, cat = self.meta["columns"]
            df = pd.DataFrame(
                [{c: row.get(c) for c in num + cat} for row in self._apply_hook(instances)]
            )
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


def _resize_logits(lg: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of [K, h, w] class scores to (width, height) before the argmax, so masks
    from a coarse model keep smooth edges at the image's own size (numpy only)."""
    _, h, w = lg.shape
    W, H = size
    if (h, w) == (H, W):
        return lg

    def axis(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        lo = np.clip(np.floor(pos).astype(np.int64), 0, n_in - 1)
        hi = np.clip(lo + 1, 0, n_in - 1)
        return lo, hi, np.clip(pos - lo, 0.0, 1.0).astype(np.float32)

    y0, y1, fy = axis(h, H)
    x0, x1, fx = axis(w, W)
    rows = lg[:, y0] * (1 - fy)[None, :, None] + lg[:, y1] * fy[None, :, None]
    return rows[:, :, x0] * (1 - fx)[None, None, :] + rows[:, :, x1] * fx[None, None, :]


def _providers(ort: Any) -> list[str]:
    """GPU execution providers when onnxruntime-gpu offers one, CPU otherwise.

    Override with TW_ORT_PROVIDERS, e.g. "CPUExecutionProvider" or
    "CUDAExecutionProvider,CPUExecutionProvider".
    """
    import os

    env = os.environ.get("TW_ORT_PROVIDERS")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    avail = set(ort.get_available_providers())
    gpu = [p for p in ("CUDAExecutionProvider", "ROCMExecutionProvider") if p in avail]
    return [*gpu, "CPUExecutionProvider"]


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
