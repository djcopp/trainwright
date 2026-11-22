"""High-level lifecycle operations shared by the CLI, MCP server and Python API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._utils import parse_duration, resolve_device, seed_everything
from .config import ExperimentConfig, ProjectConfig, dump_yaml
from .data.modules import ProjectDataModule
from .data.records import Records, load_records
from .data.split import load_split, make_split, save_split
from .metrics import metric_mode
from .models import build_model, count_parameters
from .project import Project
from .task import ClassificationTask, RegressionTask, Task
from .tracking import Run
from .trainer import Callback, FreezeBackbone, Trainer


class LifecycleError(RuntimeError):
    """Raised when a step is attempted out of order (e.g. training before a split exists)."""


# --------------------------------------------------------------------------- data + split


def get_records(project: Project) -> Records:
    return load_records(project.root, project.config)


def ensure_split(
    project: Project, records: Records | None = None, force: bool = False
) -> tuple[dict[str, Any], bool]:
    """Return the frozen split, creating it if missing. -> (split, created)"""
    existing = load_split(project)
    records = records or get_records(project)
    if existing is not None and not force:
        if existing["data_fingerprint"] != records.fingerprint:
            raise LifecycleError(
                "The data changed since the split was frozen (fingerprint mismatch). Results from new "
                "runs would not be comparable with old ones. If the change is intended run "
                "`trainwright split --force` and re-run baselines."
            )
        return existing, False
    split = make_split(records, project.config, *_dup_groups(project, records))
    save_split(project, split)
    return split, True


def _dup_groups(project: Project, records: Records) -> tuple[Any, Any]:
    cfg = project.config
    if not cfg.split.dedup or cfg.data.group_column or cfg.data.time_column:
        return None, None
    from .data.dedup import duplicate_groups

    groups, stats = duplicate_groups(
        records, cfg.split.near_duplicate_threshold, project.state_dir / "cache"
    )
    return (groups if stats["items_with_duplicates"] else None), stats


def require_split(project: Project, records: Records) -> dict[str, Any]:
    split = load_split(project)
    if split is None:
        raise LifecycleError("No frozen split yet. Run `trainwright split` first.")
    if split["data_fingerprint"] != records.fingerprint:
        raise LifecycleError(
            "Data fingerprint differs from the frozen split. Run `trainwright split --force` "
            "if the data change is intended (this invalidates earlier comparisons)."
        )
    return split


# --------------------------------------------------------------------------- tasks


def build_task(cfg: ProjectConfig, exp: ExperimentConfig, dm: ProjectDataModule) -> Task:
    model = build_model(exp.model.name, dm.spec, exp.model.params)
    if cfg.task == "classification":
        return ClassificationTask(
            model, len(dm.classes or []), exp.optim, dm.class_weights(), exp.train.label_smoothing
        )
    return RegressionTask(model, exp.optim, dm.target_scaler)


def monitor_key(cfg: ProjectConfig) -> tuple[str, str]:
    mode = cfg.metric_mode or metric_mode(cfg.metric)
    return f"val_{cfg.metric}", mode


# --------------------------------------------------------------------------- training


def train(
    project: Project,
    exp: ExperimentConfig,
    name: str | None = None,
    hypothesis: str | None = None,
    parent: str | None = None,
    tags: list[str] | None = None,
    progress: Callable[[str], None] | None = print,
    split_override: dict[str, Any] | None = None,
    kind: str = "neural",
    extra_meta: dict[str, Any] | None = None,
) -> Run:
    cfg = project.config
    project.ensure_dirs()
    project.load_user_plugins()
    records = get_records(project)
    split, created = ensure_split(project, records)
    if split_override is not None:
        split = split_override
    run = Run.create(
        project.runs_dir,
        name or exp.model.name,
        kind=kind,
        hypothesis=hypothesis,
        tags=tags,
        parent=parent,
        project_root=project.root,
        extra_meta={
            "corrections": records.corrections,
            "model": exp.model.name,
            "split_hash": split["split_hash"],
            "data_fingerprint": records.fingerprint,
            **(extra_meta or {}),
        },
    )
    (run.config_path).write_text(dump_yaml(exp))

    def say(msg: str) -> None:
        if progress:
            progress(msg)
        run.log(msg)

    if created:
        say(f"no split existed: froze a new {split['strategy']} split {split['counts']}")
    with run.lifecycle():
        seed_everything(exp.train.seed, exp.train.deterministic)
        device = resolve_device(exp.train.device)
        from .featurehook import FeatureHook

        hook = FeatureHook.from_project(project.root) if cfg.modality == "tabular" else None
        if hook is not None:
            run.update_meta(features_sha=hook.sha)
        dm = ProjectDataModule(
            cfg,
            exp,
            records,
            split,
            device,
            cache_dir=project.state_dir / "cache",
            log=say,
            feature_hook=hook,
        )
        task = build_task(cfg, exp, dm)
        n_params = count_parameters(task.model)
        monitor, mode = monitor_key(cfg)
        say(
            f"run {run.id}: {exp.model.name} ({n_params:,} params) on {device}; "
            f"train={dm.n('train')} val={dm.n('val')}; selecting on {monitor} ({mode})"
        )
        callbacks: list[Callback] = (
            [FreezeBackbone(exp.train.freeze_backbone_epochs)]
            if exp.train.freeze_backbone_epochs
            else []
        )
        trainer = Trainer(
            max_epochs=exp.train.max_epochs,
            device=device,
            precision=exp.train.precision,
            grad_clip=exp.train.grad_clip,
            accumulate=exp.train.accumulate,
            monitor=monitor,
            mode=mode,
            patience=exp.train.patience,
            min_steps=exp.train.min_steps,
            time_budget_s=parse_duration(exp.train.time_budget),
            callbacks=callbacks,
            run=run,
            checkpoint_extra={
                "model_name": exp.model.name,
                "model_params": exp.model.params,
                "spec": dm.spec,
                "preprocessor": dm.preprocessor_state(),
                "experiment": exp.model_dump(),
                "project": cfg.model_dump(),
            },
            eval_train_metrics=exp.train.eval_train_metrics,
            progress=progress,
            save_last=kind == "final",
        )
        result = trainer.fit(task, datamodule=dm)
        best = result.best_row
        val_se = float("nan")
        calibration = None
        if result.best_row:  # best weights are restored: measure how noisy the val metric is
            from .metrics import bootstrap_se

            raw, yv, _ = trainer.predict_raw(task, dm.dataloader("val", train=False))
            yv_eval = dm.target_scaler.decode(yv) if dm.target_scaler is not None else yv
            val_se = bootstrap_se(
                cfg.task, cfg.metric, yv_eval, task.postprocess(raw), len(dm.classes or []) or None
            )
            if cfg.task == "classification":
                from .metrics import expected_calibration_error, fit_temperature

                t = fit_temperature(raw, yv)
                ece_before = expected_calibration_error(task.postprocess(raw), yv)
                task.temperature = t  # type: ignore[assignment]
                ece_after = expected_calibration_error(task.postprocess(raw), yv)
                calibration = {
                    "temperature": round(t, 4),
                    "ece_before": round(ece_before, 4),
                    "ece_after": round(ece_after, 4),
                }
                if run.best_checkpoint.exists():
                    ck = torch.load(run.best_checkpoint, map_location="cpu", weights_only=False)
                    ck["temperature"] = t
                    torch.save(ck, run.best_checkpoint)
        summary = {
            "kind": kind,
            "model": exp.model.name,
            "n_params": n_params,
            "device": device,
            "primary_metric": cfg.metric,
            "metric_mode": mode,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "max_epochs": trainer.max_epochs,
            "stop_reason": result.stop_reason,
            "train_time_s": result.train_time_s,
            "val": {k[4:]: v for k, v in best.items() if k.startswith("val_")},
            "train": {k[6:]: v for k, v in best.items() if k.startswith("train_")},
            "split_hash": split["split_hash"],
            "n_train": dm.n("train"),
            "n_val": dm.n("val"),
            "val_se": {cfg.metric: val_se},
            "calibration": calibration,
        }
        if not result.best_row:
            summary["error"] = "no epoch produced a finite validation score"
        run.write_summary(summary)
        try:
            plot_curves(run, cfg.metric)
        except Exception as e:  # pragma: no cover - plotting must never fail a run
            say(f"could not plot curves: {e}")
        from .diagnose import diagnose_run

        diag = diagnose_run(project, run)
        from ._utils import write_json

        write_json(run.diagnosis_path, diag)
        if not result.best_row:
            raise LifecycleError(f"Training failed: {result.stop_reason}")
    return run


# --------------------------------------------------------------------------- loading


def load_checkpoint(run: Run) -> dict[str, Any]:
    if not run.best_checkpoint.exists():
        raise LifecycleError(f"Run {run.id} has no checkpoint (status: {run.meta.get('status')}).")
    return torch.load(run.best_checkpoint, map_location="cpu", weights_only=False)


def load_task_and_data(
    project: Project,
    run: Run,
    splits: tuple[str, ...] = ("train", "val"),
    device: str = "auto",
    ids: list[str] | None = None,
) -> tuple[Task, ProjectDataModule, dict[str, Any]]:
    """Rebuild the exact model + preprocessing of a run from its checkpoint."""
    project.load_user_plugins()
    ckpt = load_checkpoint(run)
    exp = ExperimentConfig.model_validate(ckpt["experiment"])
    records = get_records(project)
    split = require_split(project, records)
    if run.meta.get("split_hash") and run.meta["split_hash"] != split["split_hash"]:
        raise LifecycleError(
            f"Run {run.id} was trained on a different split; it cannot be evaluated on this one."
        )
    if ids is not None:  # predict arbitrary rows (e.g. a bag fold's out-of-fold rows) as "val"
        split = {**split, "val": list(ids)}
    dev = resolve_device(device)
    dm = ProjectDataModule(
        project.config,
        exp,
        records,
        split,
        dev,
        cache_dir=project.state_dir / "cache",
        splits=splits,
        preprocessor_state=ckpt["preprocessor"],
    )
    task = build_task(project.config, exp, dm)
    task.model.load_state_dict(ckpt["model_state"])
    if hasattr(task, "temperature"):
        task.temperature = float(ckpt.get("temperature", 1.0))  # type: ignore[assignment]
    task.to(dev).eval()
    return task, dm, ckpt


def predict_split(
    project: Project, run: Run, split: str, ids: list[str] | None = None
) -> dict[str, Any]:
    task, dm, ckpt = load_task_and_data(project, run, splits=("train", split), ids=ids)
    trainer = Trainer(device=dm.device, progress=None)
    raw, y, loss = trainer.predict_raw(task, dm.dataloader(split, train=False))
    return {
        "raw": raw,
        "y": y,
        "loss": loss,
        "task": task,
        "dm": dm,
        "ckpt": ckpt,
        "outputs": task.postprocess(raw),
    }


# --------------------------------------------------------------------------- plots


def plot_curves(run: Run, metric: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = run.history
    if not h:
        raise ValueError("no history")
    ep = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(ep, [r["train_loss"] for r in h], label="train")
    ax.plot(ep, [r["val_loss"] for r in h], label="val")
    ax.set_title("loss")
    ax.set_xlabel("epoch")
    ax.legend()
    ax = axes[1]
    if f"train_{metric}" in h[0]:
        ax.plot(ep, [r.get(f"train_{metric}") for r in h], label="train (train mode)")
    ax.plot(ep, [r.get(f"val_{metric}") for r in h], label="val")
    best = [r for r in h if r.get("is_best")]
    if best:
        b = best[-1]
        ax.axvline(b["epoch"], color="grey", ls="--", lw=0.8)
        ax.scatter(
            [b["epoch"]],
            [b.get(f"val_{metric}")],
            color="red",
            zorder=5,
            label=f"best (ep {b['epoch']})",
        )
    ax.set_title(metric)
    ax.set_xlabel("epoch")
    ax.legend()
    ax = axes[2]
    ax.plot(ep, [r["lr"] for r in h])
    ax.set_title("learning rate")
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    meta = run.meta
    fig.suptitle(f"{run.id}  {meta.get('hypothesis') or ''}"[:140], fontsize=10)
    fig.tight_layout()
    fig.savefig(run.curves_path, dpi=90)
    plt.close(fig)
    return run.curves_path


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def finalize(
    project: Project,
    run: Run,
    include_test: bool = False,
    progress: Callable[[str], None] | None = print,
) -> Run:
    """Retrain a selected run's exact recipe on train+val (+test) for its best number of epochs.

    The result is the production model. Its "val" numbers are NOT held out (val is in its
    training data); quote the parent run's val/test numbers instead.
    """
    import yaml

    if run.meta.get("bag"):
        return _finalize_bag(project, run, include_test, progress)
    if run.meta.get("members"):
        return _finalize_ensemble(project, run, include_test, progress)
    if run.meta.get("kind") == "baseline":
        from .baselines import refit_baseline

        parts = ["train", "val"] + (["test"] if include_test else [])
        return refit_baseline(project, run, parts, progress)
    if run.meta.get("kind") != "neural":
        raise LifecycleError("finalize needs a completed neural or baseline run.")
    records = get_records(project)
    split = require_split(project, records)
    exp = ExperimentConfig.model_validate(yaml.safe_load(run.config_path.read_text()))
    epochs = int(run.summary.get("best_epoch") or exp.train.max_epochs)
    exp = exp.model_copy(
        update={
            "train": exp.train.model_copy(
                update={
                    "max_epochs": max(1, epochs),
                    "patience": None,
                    "time_budget": None,
                    "min_steps": 0,
                }
            )
        }
    )
    parts = ["train", "val"] + (["test"] if include_test else [])
    merged = dict(split)
    merged["train"] = sorted({i for p in parts for i in split[p]})
    return train(
        project,
        exp,
        name=f"final-{run.id}",
        parent=run.id,
        tags=["final"],
        hypothesis=f"production model: {run.id}'s recipe retrained on {'+'.join(parts)} for {epochs} epochs",
        progress=progress,
        split_override=merged,
        kind="final",
        extra_meta={
            "trained_on": parts,
            "includes_test": include_test,
            "split_hash": split["split_hash"],
        },
    )


def _finalize_bag(
    project: Project, run: Run, include_test: bool, progress: Callable[[str], None] | None
) -> Run:
    """A bag is already trained on train+val: its production model is the same k fold models."""
    if include_test and progress:
        progress("note: a bag's fold models are not retrained, so test is not added to training")
    final = Run.create(
        project.runs_dir,
        f"final-{run.id}",
        kind="final",
        parent=run.id,
        tags=["final", "bag"],
        hypothesis=f"production model: {run.id}'s fold models (already trained on train+val)",
        project_root=project.root,
        extra_meta={
            "members": run.meta["members"],
            "weights": run.meta["weights"],
            "model": run.meta.get("model"),
            "split_hash": run.meta.get("split_hash"),
            "data_fingerprint": run.meta.get("data_fingerprint"),
            "trained_on": ["train", "val"],
            "includes_test": False,
        },
    )
    final.write_summary(
        {
            "kind": "final",
            "model": run.meta.get("model"),
            "parent": run.id,
            "members": run.meta["members"],
            "note": f"the k fold models of {run.id}; quote {run.id}'s held-out numbers",
        }
    )
    final.set_status("completed")
    return final


def _finalize_ensemble(
    project: Project, run: Run, include_test: bool, progress: Callable[[str], None] | None
) -> Run:
    finals = [
        finalize(project, Run(project.runs_dir / m), include_test, progress)
        for m in run.meta["members"]
    ]
    parts = ["train", "val"] + (["test"] if include_test else [])
    final = Run.create(
        project.runs_dir,
        f"final-{run.id}",
        kind="final",
        parent=run.id,
        tags=["final", "ensemble"],
        hypothesis=f"production ensemble: {run.id}'s members retrained on {'+'.join(parts)}",
        project_root=project.root,
        extra_meta={
            "members": [f.id for f in finals],
            "weights": run.meta["weights"],
            "model": run.meta.get("model"),
            "split_hash": run.meta.get("split_hash"),
            "data_fingerprint": run.meta.get("data_fingerprint"),
            "trained_on": parts,
            "includes_test": include_test,
        },
    )
    final.write_summary(
        {
            "kind": "final",
            "model": run.meta.get("model"),
            "parent": run.id,
            "members": [f.id for f in finals],
            "note": f"trained on {'+'.join(parts)}; quote {run.id}'s held-out numbers",
        }
    )
    final.set_status("completed")
    return final
