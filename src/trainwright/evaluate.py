"""Evaluation with error analysis, and the sealed test set.

``evaluate(split="val")`` is free to call as often as you like. ``split="test"`` is the final
exam: each call is appended to ``.trainwright/test_ledger.jsonl`` and warnings are raised when
the test set is used for more than one model (that is model selection on test, which makes the
reported number optimistic).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._utils import append_jsonl, now_iso, read_jsonl, write_json
from .leaderboard import best_run
from .metrics import compute_metrics, confusion, per_class_report, top_confusions
from .pipeline import LifecycleError, get_records, predict_split, require_split
from .preprocessing import LabelEncoder
from .project import Project
from .tracking import Run


def _baseline_outputs(
    project: Project, run: Run, split: str
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    import joblib

    from .baselines import _pixels, _tabular_frame, _to_str_cats

    cfg = project.config
    blob = joblib.load(run.dir / "model.joblib")
    model = blob["model"]
    records = get_records(project)
    sp = require_split(project, records)
    recs = records.subset(sp[split])
    classes = sp.get("classes")
    name = run.meta.get("model", "")
    if name.endswith("dummy"):
        X: Any = np.zeros((len(recs), 1))
    elif cfg.modality == "tabular":
        from .featurehook import FeatureHook

        num, cat = blob["columns"]
        hook = FeatureHook.from_state(blob["feature_hook"]) if blob.get("feature_hook") else None
        X = _to_str_cats(_tabular_frame(recs, num + cat, hook), cat)
    elif cfg.modality == "text":
        X = recs.texts
    else:
        X = _pixels(recs)
    if cfg.task == "classification":
        assert classes is not None
        y = LabelEncoder(classes).encode(recs.y.tolist())
        p = model.predict_proba(X)
        out = np.zeros((len(recs), len(classes)))
        out[:, np.asarray(model.classes_, dtype=int)] = p
    else:
        y = recs.y.astype(float)
        out = model.predict(X)
    paths = np.array([str(p) for p in recs.paths]) if recs.paths is not None else np.array([])
    return out, y, recs.ids, paths


def test_ledger(project: Project) -> list[dict[str, Any]]:
    return read_jsonl(project.ledger_path)


def evaluate(
    project: Project, run: Run, split: str = "val", reason: str | None = None, force: bool = False
) -> dict[str, Any]:
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    cfg = project.config
    warnings: list[str] = []
    if run.meta.get("status") != "completed":
        raise LifecycleError(
            f"Run {run.id} is {run.meta.get('status')}; only completed runs can be evaluated."
        )

    if split == "test" and not force:
        cached = run.eval_result("test")
        if cached is not None:
            cached = dict(cached)
            cached["warnings"] = [
                "cached result: this run was already evaluated on test; no new test access was "
                "recorded (use --force to recompute)"
            ]
            return cached
    if split == "test":
        ledger = test_ledger(project)
        others = sorted({e["run"] for e in ledger if e["run"] != run.id})
        if others:
            warnings.append(
                f"The test set was already used to evaluate {len(others)} other run(s) {others[:5]}. "
                "Choosing between models by test score is selection on test: the reported test number is "
                "optimistic. Select on val; report test once for the final model."
            )
        from .leaderboard import leaderboard

        best = best_run(project)
        tied = {r["id"] for r in leaderboard(project) if r.get("tied_with_leader")}
        if best and best["id"] != run.id and run.id not in tied:
            warnings.append(
                f"{run.id} is not the best run on val ({best['id']} is). If this is your final model, "
                "say why in --reason."
            )

    kind = run.meta.get("kind")
    if kind == "final" and (split in run.meta.get("trained_on", [])):
        raise LifecycleError(
            f"{run.id} was trained on {split}; evaluating it there is meaningless. Quote the parent "
            f"run {run.meta.get('parent')}'s held-out numbers."
        )
    paths = np.array([])
    if kind == "baseline" or (run.dir / "model.joblib").exists():
        outputs, y, ids, paths = _baseline_outputs(project, run, split)
        loss = None
    else:
        res = predict_split(project, run, split)
        dm = res["dm"]
        outputs, ids = res["outputs"], dm.ids(split)
        y = res["y"]
        if cfg.task == "regression":
            y = dm.target_scaler.decode(y)
        loss = res["loss"]
        recs = dm.records[split]
        if recs.paths is not None:
            paths = np.array([str(p) for p in recs.paths])

    out_dir = run.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "run": run.id,
        "split": split,
        "n": len(ids),
        "evaluated_at": now_iso(),
        "primary_metric": cfg.metric,
    }
    if cfg.task == "classification":
        sp = (
            require_split(project, get_records(project))
            if kind == "baseline" or (run.dir / "model.joblib").exists()
            else {"classes": res["dm"].classes}
        )
        classes: list[str] = sp["classes"]
        probs = outputs
        metrics = compute_metrics("classification", y, probs, len(classes), outputs_are_probs=True)
        pred = probs.argmax(1)
        cm = confusion(y, pred, len(classes))
        result["per_class"] = per_class_report(y, pred, classes)
        result["top_confusions"] = top_confusions(cm, classes)
        conf = probs.max(1)
        p_true = probs[np.arange(len(y)), y.astype(int)]
        worst = np.argsort(p_true)[:25]
        result["worst_examples"] = [
            {
                "id": ids[i],
                "true": classes[int(y[i])],
                "predicted": classes[int(pred[i])],
                "confidence": round(float(conf[i]), 4),
                "p_true": round(float(p_true[i]), 4),
                **({"path": paths[i]} if len(paths) else {}),
            }
            for i in worst
            if pred[i] != y[i]
        ]
        # coverage if low-confidence predictions are deferred (e.g. to an LLM or a human)
        result["confidence_routing"] = _routing_table(conf, pred == y)
        _write_predictions(
            out_dir / f"{split}_predictions.csv",
            ids,
            [classes[int(i)] for i in y],
            [classes[int(i)] for i in pred],
            conf,
        )
        try:
            result["confusion_png"] = str(
                _plot_confusion(cm, classes, out_dir / f"{split}_confusion.png")
            )
            if len(paths) and result["worst_examples"]:
                result["errors_png"] = str(
                    _error_sheet(result["worst_examples"], out_dir / f"{split}_errors.png")
                )
        except Exception as e:  # pragma: no cover
            warnings.append(f"plotting failed: {e}")
    else:
        pred = np.asarray(outputs).reshape(-1)
        metrics = compute_metrics("regression", y, pred)
        err = pred - y
        worst = np.argsort(-np.abs(err))[:25]
        result["worst_examples"] = [
            {"id": ids[i], "true": float(y[i]), "predicted": float(pred[i]), "error": float(err[i])}
            for i in worst
        ]
        result["residuals"] = {
            "mean": float(err.mean()),
            "p95_abs": float(np.percentile(np.abs(err), 95)),
        }
        _write_predictions(
            out_dir / f"{split}_predictions.csv", ids, y.tolist(), pred.tolist(), None
        )
        try:
            result["residuals_png"] = str(
                _plot_residuals(y, pred, out_dir / f"{split}_residuals.png")
            )
        except Exception as e:  # pragma: no cover
            warnings.append(f"plotting failed: {e}")
    if loss is not None:
        metrics["loss"] = loss
    result["metrics"] = metrics
    result["warnings"] = warnings
    write_json(out_dir / f"{split}.json", result)
    if split == "test":
        append_jsonl(
            project.ledger_path,
            {
                "run": run.id,
                "at": now_iso(),
                "metric": cfg.metric,
                "score": metrics.get(cfg.metric),
                "reason": reason,
            },
        )
    return result


def _routing_table(conf: np.ndarray, correct: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        keep = conf >= t
        cov = float(keep.mean())
        acc = float(correct[keep].mean()) if keep.any() else float("nan")
        rows.append(
            {"threshold": t, "coverage": round(cov, 4), "accuracy_on_covered": round(acc, 4)}
        )
    return rows


def _write_predictions(
    path: Path, ids: list[str], y: list[Any], pred: list[Any], conf: np.ndarray | None
) -> None:
    import csv

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "true", "predicted"] + (["confidence"] if conf is not None else []))
        for i in range(len(ids)):
            w.writerow([ids[i], y[i], pred[i]] + ([f"{conf[i]:.5f}"] if conf is not None else []))


def _plot_confusion(cm: np.ndarray, classes: list[str], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(classes)
    if k > 40:  # keep the most confused classes readable
        idx = np.argsort(-(cm.sum(1) - np.diag(cm)))[:40]
        cm, classes = cm[np.ix_(idx, idx)], [classes[i] for i in idx]
        k = 40
    norm = cm / np.maximum(1, cm.sum(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(max(5, k * 0.45 + 2), max(4, k * 0.45 + 1.5)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_xticklabels([c[:18] for c in classes], rotation=90, fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([c[:18] for c in classes], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if k <= 20:
        for i in range(k):
            for j in range(k):
                if cm[i, j]:
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black",
                    )
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def _plot_residuals(y: np.ndarray, pred: np.ndarray, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y, pred, s=4, alpha=0.4)
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    axes[0].plot([lo, hi], [lo, hi], color="red", lw=1)
    axes[0].set_xlabel("true")
    axes[0].set_ylabel("predicted")
    axes[1].hist(pred - y, bins=60)
    axes[1].set_title("residuals (pred - true)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def _error_sheet(errors: list[dict[str, Any]], path: Path, thumb: int = 128, cols: int = 6) -> Path:
    from PIL import Image, ImageDraw

    errors = errors[:24]
    rows = (len(errors) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, e in enumerate(errors):
        r, c = divmod(k, cols)
        x, y = c * thumb, r * (thumb + 28)
        try:
            with Image.open(e["path"]) as src:
                im = src.convert("RGB")
                im.thumbnail((thumb, thumb))
                sheet.paste(im, (x, y))
        except Exception:
            pass
        draw.text((x + 2, y + thumb + 1), f"T:{e['true'][:16]}", fill="darkgreen")
        draw.text(
            (x + 2, y + thumb + 14), f"P:{e['predicted'][:12]} {e['confidence']:.2f}", fill="red"
        )
    sheet.save(path)
    return path
