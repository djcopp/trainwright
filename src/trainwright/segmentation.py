"""Segmentation helpers: the majority-class baseline and per-image scores.

Segmentation runs report per-image confusion rows ``[N, K*K]`` (true * K + predicted) instead
of full-size masks, so metrics, bootstrap error bars and error analysis stay cheap.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ._utils import write_json
from .metrics import bootstrap_se, segmentation_metrics
from .pipeline import get_records, monitor_key, require_split
from .project import Project
from .tracking import Run

EVAL_SIZE = 128  # baseline masks are compared at this resolution (a floor, not a contender)


def _masks(paths: Any, size: int = EVAL_SIZE) -> np.ndarray:
    from .data.modules import load_mask

    return np.stack([load_mask(p, size) for p in paths])


def constant_rows(masks: np.ndarray, cls: int, k: int) -> np.ndarray:
    rows = np.zeros((len(masks), k * k), dtype=np.float64)
    for i, m in enumerate(masks):
        v = m[m != 255].astype(np.int64)
        v = np.clip(v, 0, k - 1)
        rows[i] = np.bincount(v * k + cls, minlength=k * k)
    return rows


def image_miou(rows: np.ndarray, k: int) -> np.ndarray:
    """mIoU of each image on its own (classes present in its labels or predictions)."""
    return np.array([segmentation_metrics(r[None], k)["miou"] for r in rows])


def run_seg_baselines(
    project: Project, progress: Callable[[str], None] | None = print
) -> list[Run]:
    import joblib

    cfg = project.config
    project.ensure_dirs()
    records = get_records(project)
    split = require_split(project, records)
    classes = list(split["classes"])
    k = len(classes)
    train, val = records.subset(split["train"]), records.subset(split["val"])
    t0 = time.time()
    tr = _masks(train.y[: min(len(train), 500)], 64)
    counts = np.bincount(np.clip(tr[tr != 255].astype(np.int64), 0, k - 1), minlength=k)
    majority = int(counts.argmax())
    rows = constant_rows(_masks(val.y), majority, k)
    metrics = segmentation_metrics(rows, k)
    _, mode = monitor_key(cfg)
    se = bootstrap_se("segmentation", cfg.metric, np.zeros(len(rows)), rows, k)
    run = Run.create(
        project.runs_dir,
        "baseline-dummy",
        kind="baseline",
        hypothesis=f"Reference: every pixel is the most common class ({classes[majority]})",
        project_root=project.root,
        extra_meta={
            "model": "baseline:dummy",
            "split_hash": split["split_hash"],
            "data_fingerprint": records.fingerprint,
            "features_sha": None,
            "status": "completed",
        },
    )
    run.set_status("completed")
    joblib.dump(
        {
            "model": {"majority": majority},
            "modality": "image",
            "task": "segmentation",
            "classes": classes,
            "columns": None,
            "feature_hook": None,
        },
        run.dir / "model.joblib",
    )
    run.write_summary(
        {
            "kind": "baseline",
            "model": "baseline:dummy",
            "description": "predicts the most common class for every pixel: the absolute floor",
            "primary_metric": cfg.metric,
            "metric_mode": mode,
            "train_time_s": round(time.time() - t0, 2),
            "val": metrics,
            "train": {},
            "split_hash": split["split_hash"],
            "n_train": len(train),
            "n_val": len(val),
            "val_se": {cfg.metric: se},
        }
    )
    write_json(run.diagnosis_path, {"verdict": "baseline", "findings": []})
    if progress:
        progress(f"baseline dummy          val_{cfg.metric}={metrics[cfg.metric]:.4f}  -> {run.id}")
    return [run]


def baseline_rows(
    project: Project, run: Run, split: str
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    import joblib

    blob = joblib.load(run.dir / "model.joblib")
    records = get_records(project)
    recs = records.subset(require_split(project, records)[split])
    k = len(blob["classes"])
    rows = constant_rows(_masks(recs.y), int(blob["model"]["majority"]), k)
    paths = np.array([str(p) for p in recs.paths or []])
    return rows, np.zeros(len(rows)), recs.ids, paths
