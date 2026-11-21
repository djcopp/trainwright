"""Permutation feature importance on the validation split (tabular runs).

For each raw input column, shuffle it across validation rows, re-run preprocessing + model,
and measure how much the primary metric degrades. Columns whose shuffling does not hurt (or
helps) are candidates for ``data.drop``. Uses the run's own fitted preprocessing, so the
numbers reflect the deployed pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .pipeline import LifecycleError, load_task_and_data
from .project import Project
from .tracking import Run


def permutation_importance(
    project: Project, run: Run, repeats: int = 3, seed: int = 0, max_rows: int = 20000
) -> dict[str, Any]:
    cfg = project.config
    if cfg.modality != "tabular":
        raise LifecycleError("Feature importance is implemented for tabular projects.")
    if run.meta.get("kind") != "neural":
        raise LifecycleError("Pick a neural run (e.g. `best-neural`).")
    task, dm, _ = load_task_and_data(project, run, splits=("train", "val"))
    recs = dm.records["val"]
    n = len(recs)
    rng = np.random.default_rng(seed)
    rows = rng.choice(n, size=min(n, max_rows), replace=False) if n > max_rows else np.arange(n)
    assert recs.table is not None
    raw = recs.table.iloc[rows].reset_index(drop=True)
    y = dm.y_encoded("val")[rows]
    sign = 1 if cfg.metric_mode == "max" else -1

    @torch.no_grad()
    def score(df: Any) -> float:
        feats = dm.pre.transform(dm.featurize(df))
        xn = torch.from_numpy(feats["x_num"]).to(dm.device)
        xc = torch.from_numpy(feats["x_cat"]).to(dm.device)
        outs = []
        for i in range(0, len(y), 4096):
            outs.append(task.model(x_num=xn[i : i + 4096], x_cat=xc[i : i + 4096]).float().cpu())
        return float(task.compute_metrics(torch.cat(outs).numpy(), y)[cfg.metric])

    task.eval()
    base = score(raw)
    # permute *raw* input columns (so derived features from features.py are covered too)
    cols = [c for c in raw.columns if c not in dm.pre.dropped]
    results = []
    for c in cols:
        drops = []
        for _ in range(repeats):
            shuffled = raw.copy()
            shuffled[c] = raw[c].to_numpy()[rng.permutation(len(rows))]
            drops.append(sign * (base - score(shuffled)))
        results.append(
            {
                "feature": c,
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops)),
                "kind": "numeric"
                if c in dm.pre.numeric_columns
                else "categorical"
                if c in dm.pre.categorical_columns
                else "raw",
            }
        )
    results.sort(key=lambda r: -r["importance"])
    drop = [r["feature"] for r in results if r["importance"] + r["std"] <= 0]
    return {
        "run": run.id,
        "metric": cfg.metric,
        "base_score": base,
        "n_rows": len(rows),
        "importances": results,
        "suggested_drop": drop,
        "dropped_by_preprocessing": dm.pre.dropped,
    }


def format_importance(res: dict[str, Any], top: int = 25) -> str:
    lines = [
        f"{res['run']}: permutation importance on val ({res['n_rows']} rows), base {res['metric']}={res['base_score']:.4f}",
        "  (importance = metric lost when the column is shuffled; <= 0 means the model does not rely on it)",
    ]
    for r in res["importances"][:top]:
        bar = "#" * int(
            max(0, min(40, r["importance"] / max(1e-9, res["importances"][0]["importance"]) * 40))
        )
        lines.append(f"  {r['feature'][:28]:<28} {r['importance']:+.4f} ±{r['std']:.4f} {bar}")
    if len(res["importances"]) > top:
        lines.append(f"  ... {len(res['importances']) - top} more")
    if res["suggested_drop"]:
        lines.append(f"no measurable contribution: {res['suggested_drop']}")
    if res["dropped_by_preprocessing"]:
        lines.append(f"already ignored by preprocessing: {res['dropped_by_preprocessing']}")
    return "\n".join(lines)
