"""Cross-validated bagging: k models on k folds of train+val, scored out-of-fold.

Fold j trains on the other folds and early-stops on fold j (stratified; duplicate and entity
groups stay within a fold). The bag's ``val`` is the out-of-fold score on the frozen val rows, so
it is comparable with other runs; ``oof`` covers all train+val rows. The fold models already saw
train+val, so ``finalize`` is free. Fold runs are kind ``fold`` and hidden from the leaderboard.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._utils import write_json
from .project import Project
from .tracking import Run

Say = Callable[[str], None]


def make_folds(project: Project, split: dict[str, Any], k: int, seed: int = 0) -> dict[str, int]:
    """dev id -> fold, stratified by class, keeping duplicate/entity groups within one fold."""
    from .pipeline import _dup_groups, get_records

    records = get_records(project)
    dev = list(split["train"]) + list(split["val"])
    pos = records.positions(dev)
    if records.groups is not None:
        units = np.asarray(records.groups)[pos].astype(str)
    else:
        dup, _ = _dup_groups(project, records)
        units = np.asarray(dup)[pos].astype(str) if dup is not None else np.asarray(dev, dtype=str)
    y = records.y[pos]
    rng = np.random.default_rng(seed)
    # one label per unit (its first member's), then deal units round-robin within each class
    unit_label: dict[str, Any] = {}
    for u, lab in zip(units, y):
        unit_label.setdefault(u, lab if project.config.task == "classification" else 0)
    by_label: dict[Any, list[str]] = {}
    for u, lab in unit_label.items():
        by_label.setdefault(str(lab), []).append(u)
    fold_of_unit: dict[str, int] = {}
    offset = 0
    for lab in sorted(by_label):
        us = by_label[lab]
        rng.shuffle(us)
        for i, u in enumerate(us):
            fold_of_unit[u] = (i + offset) % k
        offset += len(us)
    return {i: fold_of_unit[u] for i, u in zip(dev, units)}


def bag_val_outputs(project: Project, run: Run, split: str) -> tuple[np.ndarray, np.ndarray]:
    """OOF outputs for the frozen ``val`` (or ``train``) rows of a bag, in split order."""
    from ._utils import read_json
    from .pipeline import get_records, require_split

    folds: dict[str, int] = read_json(run.dir / "folds.json")
    split_d = require_split(project, get_records(project))
    ids = list(split_d[split])
    members = run.meta["members"]
    outs: np.ndarray | None = None
    ys: np.ndarray | None = None
    where = {i: k for k, i in enumerate(ids)}
    for j, mid in enumerate(members):
        rows = [i for i in ids if folds.get(i) == j]
        if not rows:
            continue
        o, yy = _predict_ids(project, Run(project.runs_dir / mid), rows)
        if outs is None:
            outs = np.zeros((len(ids), *o.shape[1:]), dtype=np.float64)
            ys = np.empty(len(ids), dtype=yy.dtype)
        idx = np.fromiter((where[i] for i in rows), dtype=np.int64)
        outs[idx] = o
        ys[idx] = yy  # type: ignore[index]
    assert outs is not None and ys is not None
    return outs, ys


def _predict_ids(project: Project, run: Run, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from .pipeline import predict_split

    res = predict_split(project, run, "val", ids=ids)
    y = res["y"]
    if project.config.task == "regression":
        y = res["dm"].target_scaler.decode(y)
    return np.asarray(res["outputs"], dtype=np.float64), np.asarray(y)


def build_bag(
    project: Project,
    ref: str = "best-neural",
    k: int = 5,
    time_budget_s: float | None = None,
    progress: Say | None = print,
) -> Run:
    import yaml

    from .config import ExperimentConfig
    from .leaderboard import resolve_ref
    from .metrics import bootstrap_se, compute_metrics
    from .pipeline import ensure_split, train

    say = progress or (lambda s: None)
    cfg = project.config
    if cfg.data.time_column:
        raise ValueError("Bagging shuffles rows into folds; it is not valid for a time split.")
    base = resolve_ref(project, ref)
    if base.meta.get("kind") != "neural":
        raise ValueError(f"bag needs a neural run to copy the recipe from; {base.id} is not one.")
    exp = ExperimentConfig.model_validate(yaml.safe_load(base.config_path.read_text()))
    if time_budget_s:
        exp = exp.model_copy(
            update={
                "train": exp.train.model_copy(update={"time_budget": f"{int(time_budget_s / k)}s"})
            }
        )
    split, _ = ensure_split(project)
    folds = make_folds(project, split, k, seed=int(exp.train.seed))
    members = []
    for j in range(k):
        sp = dict(split)
        sp["train"] = [i for i, f in folds.items() if f != j]
        sp["val"] = [i for i, f in folds.items() if f == j]
        say(f"bag fold {j + 1}/{k}: train={len(sp['train'])} early-stop on {len(sp['val'])}")
        r = train(
            project,
            exp.model_copy(
                update={"train": exp.train.model_copy(update={"seed": int(exp.train.seed) + j})}
            ),
            name=f"fold{j}-{base.id}"[:40],
            parent=base.id,
            tags=["bag-fold"],
            hypothesis=f"fold {j + 1}/{k} of a bag of {base.id}",
            progress=None,
            split_override=sp,
            kind="fold",
            extra_meta={"bag_fold": j},
        )
        members.append(r)
    run = Run.create(
        project.runs_dir,
        f"bag-{base.id}"[:40],
        kind="bag",
        parent=base.id,
        project_root=project.root,
        hypothesis=f"{k}-fold bag of {base.id}'s recipe (out-of-fold val; trained on train+val)",
        extra_meta={
            "members": [m.id for m in members],
            "weights": [1.0] * k,
            "bag": True,
            "model": f"bag[{k}x {exp.model.name}]",
            "split_hash": split["split_hash"],
            "data_fingerprint": split.get("data_fingerprint"),
            "trained_on": ["train", "val"],
        },
    )
    write_json(run.dir / "folds.json", folds)
    m, mode = cfg.metric, cfg.metric_mode
    out, y = bag_val_outputs(project, run, "val")
    n_classes = out.shape[1] if cfg.task == "classification" else None
    if cfg.task == "classification":
        from .preprocessing import LabelEncoder

        y = (
            LabelEncoder(split["classes"]).encode([str(v) for v in y])
            if y.dtype.kind in "OUS"
            else y
        )
    metrics = compute_metrics(cfg.task, y, out, n_classes, outputs_are_probs=True)
    se = bootstrap_se(cfg.task, m, y, out, n_classes)
    oof_tr, ytr = bag_val_outputs(project, run, "train")
    if cfg.task == "classification" and ytr.dtype.kind in "OUS":
        from .preprocessing import LabelEncoder

        ytr = LabelEncoder(split["classes"]).encode([str(v) for v in ytr])
    oof_all = compute_metrics(
        cfg.task,
        np.concatenate([ytr, y]),
        np.concatenate([oof_tr, out]),
        n_classes,
        outputs_are_probs=True,
    )
    run.write_summary(
        {
            "kind": "bag",
            "model": run.meta["model"],
            "members": run.meta["members"],
            "primary_metric": m,
            "metric_mode": mode,
            "val": metrics,
            "val_se": {m: se},
            "oof": oof_all,
            "n_oof": int(len(y) + len(ytr)),
            "train_time_s": sum(float(r.summary.get("train_time_s") or 0) for r in members),
            "note": "val = out-of-fold predictions on the frozen val rows (comparable with other runs); "
            "oof = out-of-fold over all train+val rows. The bag is already trained on train+val.",
        }
    )
    write_json(run.diagnosis_path, {"verdict": "bag", "findings": [], "next_experiments": []})
    run.set_status("completed")
    # same recipe, more data, k-model average: the bag replaces its parent for selection and
    # ensembling (its out-of-fold val is a single-model estimate, so it under-states the bag)
    base.update_meta(superseded_by=run.id)
    say(
        f"bag {run.id}: val_{m}={metrics[m]:.4f} ±{se:.4f} (out-of-fold), "
        f"oof_{m} over {len(y) + len(ytr)} rows = {oof_all[m]:.4f}"
    )
    return run
