"""One metric implementation for every run, baseline and evaluation, so numbers are comparable."""

from __future__ import annotations

from typing import Any

import numpy as np

CLASSIFICATION_METRICS = {
    "accuracy": "max",
    "balanced_accuracy": "max",
    "macro_f1": "max",
    "weighted_f1": "max",
    "roc_auc": "max",
    "log_loss": "min",
    "ece": "min",
}
REGRESSION_METRICS = {"rmse": "min", "mae": "min", "r2": "max", "mape": "min"}


def metric_mode(metric: str) -> str:
    m = metric.removeprefix("val_").removeprefix("test_")
    if m in CLASSIFICATION_METRICS:
        return CLASSIFICATION_METRICS[m]
    if m in REGRESSION_METRICS:
        return REGRESSION_METRICS[m]
    if m.endswith("loss"):
        return "min"
    raise KeyError(f"Unknown metric {metric!r}")


def valid_metrics(task: str) -> list[str]:
    return list(CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    y = np.asarray(y).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(axis=1)
    labels = list(range(n_classes))
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred))
        if len(np.unique(y)) > 1
        else float("nan"),
        "macro_f1": float(
            f1_score(y, pred, average="macro", labels=sorted(set(y.tolist())), zero_division=0)
        ),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    p = np.clip(probs, 1e-7, 1)
    p = p / p.sum(axis=1, keepdims=True)
    out["log_loss"] = float(log_loss(y, p, labels=labels))
    try:
        if n_classes == 2:
            out["roc_auc"] = float(roc_auc_score(y, probs[:, 1]))
        else:
            present = sorted(set(y.tolist()))
            if len(present) > 2:
                sub = probs[:, present]
                sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
                out["roc_auc"] = float(
                    roc_auc_score(y, sub, multi_class="ovr", average="macro", labels=present)
                )
    except ValueError:
        out["roc_auc"] = float("nan")
    out["ece"] = expected_calibration_error(probs, y)
    return out


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    err = pred - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    nz = np.abs(y) > 1e-8
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "mape": float(np.abs(err[nz] / y[nz]).mean()) if nz.any() else float("nan"),
    }


def compute_metrics(
    task: str,
    y: np.ndarray,
    outputs: np.ndarray,
    n_classes: int | None = None,
    outputs_are_probs: bool = False,
) -> dict[str, float]:
    """``outputs``: logits/probs [N, K] for classification, predictions [N] for regression
    (already on the original target scale)."""
    if task == "classification":
        probs = outputs if outputs_are_probs else _softmax(np.asarray(outputs, dtype=np.float64))
        return classification_metrics(y, probs, n_classes or probs.shape[1])
    return regression_metrics(y, outputs)


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(pred, int)), 1)
    return cm


def per_class_report(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> list[dict[str, Any]]:
    cm = confusion(y, pred, len(classes))
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        support = cm[i].sum()
        predicted = cm[:, i].sum()
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(
            {
                "class": c,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": int(support),
            }
        )
    return rows


def top_confusions(cm: np.ndarray, classes: list[str], k: int = 10) -> list[dict[str, Any]]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        out.append(
            {
                "true": classes[i],
                "predicted": classes[j],
                "count": int(off[i, j]),
                "share_of_true": float(off[i, j] / max(1, cm[i].sum())),
            }
        )
    return out


def _single_metric(task: str, metric: str, y: np.ndarray, out: np.ndarray, n_classes: int) -> float:
    """Fast path for the metrics used in model selection (used by the bootstrap)."""
    if task == "classification":
        pred = out.argmax(1)
        if metric == "accuracy":
            return float((pred == y).mean())
        if metric in ("macro_f1", "balanced_accuracy"):
            present = np.unique(y)
            cm = confusion(y, pred, n_classes)
            tp = np.diag(cm).astype(float)
            support = cm.sum(1).astype(float)
            if metric == "balanced_accuracy":
                return float((tp[present] / support[present]).mean())
            predicted = cm.sum(0).astype(float)
            prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
            rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
            f1 = np.divide(
                2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0
            )
            return float(f1[present].mean())
        return compute_metrics(task, y, out, n_classes, outputs_are_probs=True).get(
            metric, float("nan")
        )
    return regression_metrics(y, out)[metric]


def bootstrap_se(
    task: str,
    metric: str,
    y: np.ndarray,
    outputs: np.ndarray,
    n_classes: int | None = None,
    n_boot: int = 200,
    seed: int = 0,
    max_rows: int = 20000,
) -> float:
    """Bootstrap standard error of a validation metric: how much of a score difference is noise.

    ``outputs`` are probabilities [N, K] (classification) or predictions [N] (regression).
    """
    y = np.asarray(y)
    outputs = np.asarray(outputs)
    n = len(y)
    if n < 10:
        return float("nan")
    rng = np.random.default_rng(seed)
    k = n_classes or (outputs.shape[1] if outputs.ndim == 2 else 1)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=min(n, max_rows))
        try:
            v = _single_metric(task, metric, y[idx], outputs[idx], k)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(np.std(vals, ddof=1)) if len(vals) > 10 else float("nan")


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T minimising NLL of softmax(logits / T).

    Leaves the argmax (accuracy) unchanged; fixes over/under-confidence.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y).astype(int)

    def nll(log_t: float) -> float:
        s = z / np.exp(log_t)
        s = s - s.max(axis=1, keepdims=True)
        logp = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    lo, hi = np.log(0.05), np.log(20.0)
    g = (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(b)
    t = float(np.exp((lo + hi) / 2))
    return t if nll(np.log(t)) <= nll(0.0) else 1.0
