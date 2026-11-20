"""The training loop. One loop for every model, so runs differ only by their config.

Handles devices, mixed precision, gradient accumulation and clipping, LR schedules, early
stopping, best-checkpointing, wall-clock budgets, non-finite-loss detection and per-epoch
logging to a ``Run``. Behaviour is extended through ``Callback``s.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._utils import fmt_seconds, resolve_device
from .task import Task
from .tracking import Run


class Callback:
    def on_fit_start(self, trainer: Trainer, task: Task) -> None: ...
    def on_epoch_start(self, trainer: Trainer, task: Task, epoch: int) -> None: ...
    def on_epoch_end(self, trainer: Trainer, task: Task, row: dict[str, Any]) -> None: ...
    def on_fit_end(self, trainer: Trainer, task: Task) -> None: ...


class EarlyStopping(Callback):
    def __init__(self, patience: int):
        self.patience = patience

    def on_epoch_end(self, trainer: Trainer, task: Task, row: dict[str, Any]) -> None:
        if trainer.epochs_since_best >= self.patience:
            trainer.stop(
                f"early stopping: no {trainer.monitor} improvement for {self.patience} epochs"
            )


class FreezeBackbone(Callback):
    """Train only the head for the first N epochs (linear probe), then unfreeze."""

    def __init__(self, epochs: int):
        self.epochs = epochs

    def _set(self, task: Task, trainable: bool) -> None:
        bb = getattr(task.model, "backbone", None)
        if bb is not None:
            for p in bb.parameters():
                p.requires_grad_(trainable)

    def on_fit_start(self, trainer: Trainer, task: Task) -> None:
        if self.epochs > 0:
            self._set(task, False)

    def on_epoch_start(self, trainer: Trainer, task: Task, epoch: int) -> None:
        if epoch == self.epochs + 1 and self.epochs > 0:
            self._set(task, True)
            trainer.say(f"unfroze backbone at epoch {epoch}")


@dataclass
class FitResult:
    best_epoch: int
    best_score: float
    best_row: dict[str, Any]
    epochs_run: int
    stop_reason: str
    train_time_s: float
    history: list[dict[str, Any]] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        max_epochs: int = 10,
        device: str = "auto",
        precision: str = "auto",
        grad_clip: float | None = 1.0,
        accumulate: int = 1,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int | None = None,
        time_budget_s: float | None = None,
        callbacks: list[Callback] | None = None,
        run: Run | None = None,
        checkpoint_path: Path | None = None,
        checkpoint_extra: dict[str, Any] | None = None,
        eval_train_metrics: bool = True,
        progress: Callable[[str], None] | None = print,
        restore_best: bool = True,
    ):
        self.max_epochs = max_epochs
        self.device = resolve_device(device)
        self.precision = precision
        self.grad_clip = grad_clip
        self.accumulate = accumulate
        self.monitor = monitor
        self.mode = mode
        self.time_budget_s = time_budget_s
        self.callbacks = list(callbacks or [])
        if patience:
            self.callbacks.append(EarlyStopping(patience))
        self.run = run
        self.checkpoint_path = checkpoint_path or (run.best_checkpoint if run else None)
        self.checkpoint_extra = checkpoint_extra or {}
        self.eval_train_metrics = eval_train_metrics
        self.progress = progress
        self.restore_best = restore_best
        # state
        self.epoch = 0
        self.best_score = math.inf if mode == "min" else -math.inf
        self.best_epoch = 0
        self.best_row: dict[str, Any] = {}
        self.epochs_since_best = 0
        self.history: list[dict[str, Any]] = []
        self._stop_reason: str | None = None
        self._best_state: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------ utils
    def say(self, msg: str) -> None:
        if self.progress:
            self.progress(msg)
        if self.run:
            self.run.log(msg)

    def stop(self, reason: str) -> None:
        if self._stop_reason is None:
            self._stop_reason = reason

    def _autocast(self) -> Any:
        p = self.precision
        dev = self.device.split(":")[0]
        if p == "auto":
            if dev == "cuda":
                p = "bf16" if torch.cuda.is_bf16_supported() else "16"
            else:
                p = "32"
        if p == "32":
            return nullcontext(), None
        dtype = torch.bfloat16 if p == "bf16" else torch.float16
        scaler = torch.amp.GradScaler(dev) if (p == "16" and dev == "cuda") else None
        return torch.autocast(device_type=dev, dtype=dtype), scaler

    def _improved(self, score: float) -> bool:
        if not math.isfinite(score):
            return False
        return score < self.best_score if self.mode == "min" else score > self.best_score

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        task: Task,
        train_loader: Iterable[Any] | None = None,
        val_loader: Iterable[Any] | None = None,
        datamodule: Any = None,
    ) -> FitResult:
        if datamodule is not None:
            train_loader = train_loader or datamodule.train_dataloader()
            val_loader = val_loader or datamodule.val_dataloader()
        assert train_loader is not None and val_loader is not None
        task.to(self.device)
        steps_per_epoch = max(1, math.ceil(len(train_loader) / self.accumulate))  # type: ignore[arg-type]
        optimizer, scheduler, interval = task.configure_optimizers(steps_per_epoch, self.max_epochs)
        amp_ctx, scaler = self._autocast()
        t0 = time.time()
        for cb in self.callbacks:
            cb.on_fit_start(self, task)

        for epoch in range(1, self.max_epochs + 1):
            self.epoch = epoch
            for cb in self.callbacks:
                cb.on_epoch_start(self, task, epoch)
            te = time.time()
            task.train()
            losses, outs, ys = [], [], []
            optimizer.zero_grad(set_to_none=True)
            n_batches = len(train_loader)  # type: ignore[arg-type]
            for i, batch in enumerate(train_loader):
                with amp_ctx:
                    step = task.training_step(batch)
                loss = step["loss"]
                if not torch.isfinite(loss):
                    self.stop(f"diverged: non-finite training loss at epoch {epoch} batch {i}")
                    break
                scaled = loss / self.accumulate
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                if (i + 1) % self.accumulate == 0 or i + 1 == n_batches:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(task.parameters(), self.grad_clip)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None and interval == "step":
                        scheduler.step()
                losses.append(loss.detach())
                if self.eval_train_metrics:
                    outs.append(step["outputs"].float().cpu())
                    ys.append(batch["y"].cpu())
                if self.time_budget_s and time.time() - t0 > self.time_budget_s:
                    self.stop(f"time budget of {fmt_seconds(self.time_budget_s)} reached")
                    break
            if not losses:
                break
            train_loss = float(torch.stack(losses).mean().item())
            row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
            if self.eval_train_metrics and outs:
                tm = task.compute_metrics(torch.cat(outs).numpy(), torch.cat(ys).numpy())
                row.update({f"train_{k}": v for k, v in tm.items()})
            val_out, val_y, val_loss = self.predict_raw(task, val_loader, amp_ctx)
            row["val_loss"] = val_loss
            row.update({f"val_{k}": v for k, v in task.compute_metrics(val_out, val_y).items()})
            row["lr"] = optimizer.param_groups[-1]["lr"]
            row["epoch_time_s"] = round(time.time() - te, 3)
            row["elapsed_s"] = round(time.time() - t0, 3)
            if not math.isfinite(train_loss):
                self.stop("diverged: non-finite training loss")
            if scheduler is not None and interval == "plateau":
                scheduler.step(row.get(self.monitor, val_loss))
            score = row.get(self.monitor, float("nan"))
            improved = self._improved(score)
            if improved:
                self.best_score, self.best_epoch, self.best_row = score, epoch, dict(row)
                self.epochs_since_best = 0
                self._save_best(task, row)
            else:
                self.epochs_since_best += 1
            row["is_best"] = improved
            self.history.append(row)
            if self.run:
                self.run.log_metrics(row)
            self._print_row(row, improved)
            for cb in self.callbacks:
                cb.on_epoch_end(self, task, row)
            if self._stop_reason:
                break

        for cb in self.callbacks:
            cb.on_fit_end(self, task)
        if self.restore_best and self._best_state is not None:
            task.model.load_state_dict(self._best_state)
        reason = self._stop_reason or f"completed max_epochs={self.max_epochs}"
        res = FitResult(
            self.best_epoch,
            self.best_score,
            self.best_row,
            self.epoch,
            reason,
            round(time.time() - t0, 2),
            self.history,
        )
        self.say(
            f"done: {reason}; best {self.monitor}={self.best_score:.4g} at epoch {self.best_epoch} "
            f"({fmt_seconds(res.train_time_s)})"
        )
        return res

    def _save_best(self, task: Task, row: dict[str, Any]) -> None:
        self._best_state = {
            k: v.detach().to("cpu", copy=True) for k, v in task.model.state_dict().items()
        }
        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": self._best_state,
                    "epoch": row["epoch"],
                    "metrics": {k: v for k, v in row.items() if isinstance(v, (int, float))},
                    **copy.deepcopy(self.checkpoint_extra),
                },
                self.checkpoint_path,
            )

    def _print_row(self, row: dict[str, Any], improved: bool) -> None:
        if not self.progress:
            return
        mon = self.monitor
        parts = [
            f"epoch {row['epoch']:>3}/{self.max_epochs}",
            f"train_loss {row['train_loss']:.4f}",
            f"val_loss {row['val_loss']:.4f}",
        ]
        if mon != "val_loss" and mon in row:
            parts.append(f"{mon} {row[mon]:.4f}{'*' if improved else ''}")
        elif improved:
            parts[-1] += "*"
        parts += [f"lr {row['lr']:.2e}", fmt_seconds(row["epoch_time_s"])]
        self.progress("  ".join(parts))

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict_raw(
        self, task: Task, loader: Iterable[Any], amp_ctx: Any = None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """-> (raw outputs [N, K], targets [N], mean loss)"""
        task.eval()
        task.to(self.device)
        ctx = amp_ctx if amp_ctx is not None else nullcontext()
        outs, ys, losses, ns = [], [], [], []
        for batch in loader:
            with ctx:
                step = task.validation_step(batch)
            outs.append(step["outputs"].float().cpu())
            ys.append(batch["y"].cpu())
            losses.append(float(step["loss"].item()))
            ns.append(len(batch["y"]))
        if not outs:
            return np.zeros((0,)), np.zeros((0,)), float("nan")
        loss = float(np.average(losses, weights=ns))
        return torch.cat(outs).numpy(), torch.cat(ys).numpy(), loss
