"""Compare runs on the primary validation metric (never on test)."""

from __future__ import annotations

import math
from typing import Any

from .data.split import load_split
from .project import Project
from .tracking import Run, list_runs


def run_row(run: Run, metric: str, current_split: str | None) -> dict[str, Any]:
    meta, summ = run.meta, run.summary
    val = summ.get("val", {})
    row: dict[str, Any] = {
        "id": run.id,
        "kind": meta.get("kind"),
        "model": meta.get("model") or summ.get("model"),
        "status": meta.get("status"),
        "hypothesis": meta.get("hypothesis"),
        "parent": meta.get("parent"),
        f"val_{metric}": val.get(metric),
        "val_loss": None,
        "n_params": summ.get("n_params"),
        "train_time_s": summ.get("train_time_s"),
        "best_epoch": summ.get("best_epoch"),
        "epochs_run": summ.get("epochs_run"),
        "stop_reason": summ.get("stop_reason"),
        "same_split": (meta.get("split_hash") == current_split) if current_split else None,
    }
    if "loss" in val:
        row["val_loss"] = val["loss"]
    test = run.eval_result("test")
    if test:
        row[f"test_{metric}"] = test["metrics"].get(metric)
    if run.diagnosis_path.exists():
        from ._utils import read_json

        d = read_json(run.diagnosis_path)
        row["diagnosis"] = d.get("verdict")
        row["findings"] = [f["code"] for f in d.get("findings", [])]
    return row


def leaderboard(
    project: Project, include_incomplete: bool = False, kind: str | None = None
) -> list[dict[str, Any]]:
    cfg = project.config
    split = load_split(project)
    current = split["split_hash"] if split else None
    rows = []
    for run in list_runs(project.runs_dir):
        row = run_row(run, cfg.metric, current)
        if not include_incomplete and row["status"] != "completed":
            continue
        if kind and row["kind"] != kind:
            continue
        rows.append(row)
    key = f"val_{cfg.metric}"
    sign = -1 if cfg.metric_mode == "max" else 1

    def sort_key(r: dict[str, Any]) -> tuple[int, int, float]:
        v = r.get(key)
        bad = v is None or (isinstance(v, float) and not math.isfinite(v))
        return (
            0 if r.get("same_split") in (True, None) else 1,
            1 if bad else 0,
            sign * (float(v) if not bad else 0.0),  # type: ignore[arg-type]
        )

    rows.sort(key=sort_key)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def best_run(project: Project, kind: str | None = None) -> dict[str, Any] | None:
    rows = [
        r
        for r in leaderboard(project, kind=kind)
        if r.get("same_split") is not False and r.get(f"val_{project.config.metric}") is not None
    ]
    return rows[0] if rows else None


def resolve_ref(project: Project, ref: str) -> Run:
    """Run ref: id / sequence / substring / 'latest' / 'best' / 'best-neural'."""
    if ref in ("best", "best-neural"):
        r = best_run(project, kind="neural" if ref == "best-neural" else None)
        if r is None:
            raise FileNotFoundError("No completed runs to pick 'best' from.")
        return Run(project.runs_dir / r["id"])
    return Run.open(project.runs_dir, ref)


def format_table(rows: list[dict[str, Any]], metric: str, limit: int = 30) -> str:
    if not rows:
        return "(no completed runs yet)"
    key = f"val_{metric}"
    has_test = any(f"test_{metric}" in r for r in rows)
    header = ["#", "run", "kind", key, "params", "time", "epochs", "diagnosis"] + (
        [f"test_{metric}"] if has_test else []
    )
    lines = []
    for r in rows[:limit]:
        v = r.get(key)
        ep = f"{r['best_epoch']}/{r['epochs_run']}" if r.get("best_epoch") else "-"
        params = f"{r['n_params'] / 1e6:.2f}M" if r.get("n_params") else "-"
        t = f"{r['train_time_s']:.0f}s" if r.get("train_time_s") is not None else "-"
        diag = ",".join(r.get("findings") or []) or (r.get("diagnosis") or "-")
        flag = "" if r.get("same_split") in (True, None) else " (old split)"
        cells = [
            str(r["rank"]),
            r["id"] + flag,
            r["kind"] or "-",
            f"{v:.4f}" if v is not None else "-",
            params,
            t,
            ep,
            diag[:40],
        ]
        if has_test:
            tv = r.get(f"test_{metric}")
            cells.append(f"{tv:.4f}" if tv is not None else "")
        lines.append(cells)
    widths = [max(len(h), *(len(c[i]) for c in lines)) for i, h in enumerate(header)]
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    out += ["  ".join(c.ljust(w) for c, w in zip(cells, widths)) for cells in lines]
    return "\n".join(out)
