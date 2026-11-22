"""Tasks: a model + its loss + optimisation + metrics (the LightningModule analogue).

Subclass ``Task`` (or ``ClassificationTask``/``RegressionTask``) for custom behaviour; the
``Trainer`` only calls the methods defined here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import OptimConfig
from .metrics import compute_metrics
from .preprocessing import TargetScaler

Batch = dict[str, torch.Tensor]


def model_inputs(batch: Batch) -> dict[str, torch.Tensor]:
    return {k: v for k, v in batch.items() if k != "y"}


class Task(nn.Module):
    def __init__(self, model: nn.Module, optim: OptimConfig | None = None):
        super().__init__()
        self.model = model
        self.optim_cfg = optim or OptimConfig()

    # -- override these ------------------------------------------------------
    def loss(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_metrics(self, outputs: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {}

    def postprocess(self, outputs: np.ndarray) -> np.ndarray:
        """Raw model outputs -> probabilities (classification) or target-scale predictions."""
        return outputs

    # -- default implementations ---------------------------------------------
    def forward(self, batch: Batch) -> torch.Tensor:
        return self.model(**model_inputs(batch))

    def training_step(self, batch: Batch) -> dict[str, torch.Tensor]:
        out = self(batch)
        if out.ndim == 3 and hasattr(self.model, "aggregate"):  # [B, k, out]: k ensemble members
            loss = torch.stack(
                [self.loss(out[:, j], batch["y"]) for j in range(out.shape[1])]
            ).mean()
            agg = self.model.aggregate(out)  # type: ignore[operator]
            return {"loss": loss, "outputs": agg.detach()}
        return {"loss": self.loss(out, batch["y"]), "outputs": out.detach()}

    def validation_step(self, batch: Batch) -> dict[str, torch.Tensor]:
        out = self(batch)
        return {"loss": self.loss(out, batch["y"]), "outputs": out}

    def param_groups(self) -> list[dict[str, Any]]:
        """Weight decay on weights only (not biases/norms); lower LR for pretrained backbones."""
        cfg = self.optim_cfg
        backbone = (
            getattr(self.model, "backbone", None)
            if getattr(self.model, "pretrained_backbone", False)
            else None
        )
        backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for p in self.model.parameters():
            key = (id(p) in backbone_ids, p.ndim <= 1)
            groups.setdefault(key, []).append(p)
        out = []
        for (is_backbone, no_decay), params in groups.items():
            out.append(
                {
                    "params": params,
                    "lr": cfg.lr * (cfg.backbone_lr_mult if is_backbone else 1.0),
                    "weight_decay": 0.0 if no_decay else cfg.weight_decay,
                    "name": f"{'backbone' if is_backbone else 'model'}{'_no_decay' if no_decay else ''}",
                }
            )
        return out

    def configure_optimizers(
        self, steps_per_epoch: int, max_epochs: int
    ) -> tuple[torch.optim.Optimizer, Any, str]:
        cfg = self.optim_cfg
        groups = self.param_groups()
        opt: torch.optim.Optimizer
        if cfg.name == "adamw":
            opt = torch.optim.AdamW(groups, lr=cfg.lr)
        elif cfg.name == "adam":
            opt = torch.optim.Adam(groups, lr=cfg.lr)
        else:
            opt = torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
        total = max(1, steps_per_epoch * max_epochs)
        warm = int(cfg.warmup_epochs * steps_per_epoch)
        if cfg.scheduler == "cosine":
            horizon = {"total": total}  # mutable: the trainer shrinks it to fit a time budget

            def f(step: int) -> float:
                if step < warm:
                    return 0.01 + 0.99 * step / max(1, warm)
                prog = (step - warm) / max(1, horizon["total"] - warm)
                return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

            cos = torch.optim.lr_scheduler.LambdaLR(opt, f)
            cos.tw_horizon = horizon  # type: ignore[attr-defined]
            return opt, cos, "step"
        if cfg.scheduler == "onecycle":
            max_lrs = [g["lr"] for g in opt.param_groups]
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=max_lrs,
                total_steps=total,
                pct_start=max(0.05, min(0.5, warm / total or 0.25)),
            )
            return opt, sched, "step"
        if cfg.scheduler == "plateau":
            return (
                opt,
                torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=3),
                "plateau",
            )
        if warm > 0:
            return (
                opt,
                torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, 0.01 + 0.99 * s / warm)),
                "step",
            )
        return opt, None, "none"


class ClassificationTask(Task):
    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        optim: OptimConfig | None = None,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__(model, optim)
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        self.temperature = 1.0  # fitted on val after training (temperature scaling)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def loss(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            outputs.float(),
            y,
            weight=self.class_weights,  # type: ignore[arg-type]
            label_smoothing=self.label_smoothing,
        )

    def postprocess(self, outputs: np.ndarray) -> np.ndarray:
        outputs = outputs / self.temperature
        z = outputs - outputs.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def compute_metrics(self, outputs: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return compute_metrics(
            "classification", y, self.postprocess(outputs), self.n_classes, outputs_are_probs=True
        )


class RegressionTask(Task):
    def __init__(
        self,
        model: nn.Module,
        optim: OptimConfig | None = None,
        target_scaler: TargetScaler | None = None,
        loss_name: str = "mse",
    ):
        super().__init__(model, optim)
        self.target_scaler = target_scaler or TargetScaler("none")
        self.loss_name = loss_name

    def loss(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = outputs.float().reshape(-1)
        if self.loss_name == "huber":
            return F.smooth_l1_loss(out, y.float())
        return F.mse_loss(out, y.float())

    def postprocess(self, outputs: np.ndarray) -> np.ndarray:
        return self.target_scaler.decode(np.asarray(outputs).reshape(-1))

    def compute_metrics(self, outputs: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return compute_metrics(
            "regression", self.target_scaler.decode(y), self.postprocess(outputs)
        )
