"""In-project predictions with torch on the training device (GPU / Apple Silicon).

Loads run checkpoints directly and reuses the runtime's preprocessing, so outputs match the
exported ONNX bundle without exporting. Deployment still uses the bundle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .preprocessing import (
    ImagePreprocessor,
    TabularPreprocessor,
    TargetScaler,
    text_preprocessor_from_state,
)
from .project import Project
from .runtime import Predictor
from .tracking import Run


class TorchPredictor(Predictor):
    """Same interface as the bundle ``Predictor``; built from a run instead of a bundle."""

    def __init__(self, project: Project, run: Run, device: str = "auto"):
        import torch

        from ._utils import resolve_device

        cfg = project.config
        self.dir = run.dir
        self.device = resolve_device(device)
        self.modality = cfg.modality
        self.task = cfg.task
        self.meta = {
            "modality": cfg.modality,
            "task": cfg.task,
            "date_columns": dict(cfg.data.date_columns),
            "text_features": _text_states(project),
        }
        meta = run.meta
        members = meta.get("members")
        if members:
            self.kind = "ensemble"
            self.members = [
                TorchPredictor(project, Run(project.runs_dir / m), device) for m in members
            ]
            self.weights = np.asarray(meta.get("weights") or [1.0] * len(members), dtype=np.float64)
            self.weights = self.weights / self.weights.sum()
            self.classes = self.members[0].classes
            self.target = None
            return
        if (run.dir / "model.joblib").exists():
            import joblib

            blob = joblib.load(run.dir / "model.joblib")
            self.kind = "sklearn"
            self.model = blob["model"]
            self.classes = _classes(project)
            self.meta.update(
                columns=blob.get("columns"), baseline=meta.get("model", "").split(":")[-1]
            )
            self.target = None
            self.hook = _hook(blob.get("feature_hook"))
            return
        from .models import build_model
        from .pipeline import load_checkpoint

        ckpt = load_checkpoint(run)
        pre = ckpt["preprocessor"]
        feats = pre["features"]
        self.kind = "torch"
        self.classes = _classes(project)
        self.target = TargetScaler.from_state(pre["target"]) if pre.get("target") else None
        self.hook = _hook(pre.get("feature_hook"))
        if self.modality == "tabular":
            self.pre = TabularPreprocessor.from_state(feats)
        elif self.modality == "text":
            self.pre = text_preprocessor_from_state(feats)
        else:
            self.pre = ImagePreprocessor.from_state(feats)
        project.load_user_plugins()
        model = build_model(ckpt["model_name"], ckpt["spec"], ckpt["model_params"])
        model.load_state_dict(ckpt["model_state"])
        for m in model.modules():
            if hasattr(m, "cache_features"):
                m.cache_features = False  # type: ignore[assignment]
        self.torch_model = model.to(self.device).eval()
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.input_names = list(ckpt["spec"].get("input_names") or _input_names(self.modality))
        self._torch = torch

    def predict_raw(self, instances: list[Any]) -> np.ndarray:
        if self.kind == "sklearn":
            return self._sklearn_raw(instances)
        torch = self._torch
        feats = self.featurise(instances)
        with torch.inference_mode():
            kwargs = {k: torch.as_tensor(v).to(self.device) for k, v in feats.items()}
            out = self.torch_model(**kwargs).float()
            if self.task == "classification":
                out = torch.softmax(out / self.temperature, dim=-1)
        return out.cpu().numpy()


def _classes(project: Project) -> list[str] | None:
    from .data.split import load_split

    return (
        (load_split(project) or {}).get("classes")
        if project.config.task == "classification"
        else None
    )


def _input_names(modality: str) -> list[str]:
    return {"tabular": ["x_num", "x_cat"], "text": ["tokens"], "image": ["image"]}[modality]


def _hook(state: dict[str, Any] | None) -> Any:
    if not state:
        return None
    from .featurehook import FeatureHook

    return FeatureHook.from_state(state)


def _text_states(project: Project) -> dict[str, Any]:
    from .export import _text_feature_states

    return _text_feature_states(project)


def accelerator_available() -> bool:
    from ._utils import resolve_device

    return resolve_device("auto") != "cpu"
