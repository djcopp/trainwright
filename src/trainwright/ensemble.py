"""Ensembles of runs: average calibrated probabilities (or target-space predictions).

Averaging a few diverse, individually strong models (e.g. an MLP and gradient boosting) is one
of the most reliable accuracy gains in tabular ML. Every run shares the frozen split, so their
validation predictions line up row for row and an ensemble is cheap to build and score.

* ``build_ensemble`` picks members (explicit, or the top runs on val), chooses how many of the
  top-k to average by val score, and records an ``ensemble`` run that competes on the leaderboard.
* ``ensemble_outputs`` produces averaged outputs for any split (used by ``evaluate``).
* ``finalize`` finalizes every member; ``export`` writes a bundle of member bundles.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._utils import write_json
from .metrics import bootstrap_se, compute_metrics
from .project import Project
from .tracking import Run


def member_outputs(project: Project, run: Run, split: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (outputs, y) for one member on a split. Probabilities or target-space predictions."""
    if run.meta.get("members"):
        return ensemble_outputs(project, run, split)
    if (run.dir / "model.joblib").exists():
        from .evaluate import _baseline_outputs

        out, y, _, _ = _baseline_outputs(project, run, split)
        return np.asarray(out, dtype=np.float64), np.asarray(y)
    from .pipeline import predict_split

    res = predict_split(project, run, split)
    y = res["y"]
    if project.config.task == "regression":
        y = res["dm"].target_scaler.decode(y)
    return np.asarray(res["outputs"], dtype=np.float64), np.asarray(y)


def _average(outs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return sum(wi * o for wi, o in zip(w, outs))  # type: ignore[return-value]


def ensemble_outputs(project: Project, run: Run, split: str) -> tuple[np.ndarray, np.ndarray]:
    meta = run.meta
    if meta.get("bag") and split in ("train", "val"):
        from .bagging import bag_val_outputs

        return bag_val_outputs(
            project, run, split
        )  # out-of-fold: no row scored by a model that saw it
    outs, y = [], None
    for mid in meta["members"]:
        o, yy = member_outputs(project, Run(project.runs_dir / mid), split)
        outs.append(o)
        y = yy if y is None else y
    assert y is not None
    return _average(outs, meta["weights"]), y


def _greedy_weights(
    outs: list[np.ndarray], objective: Callable[[np.ndarray], float], max_steps: int = 25
) -> list[int]:
    """Caruana et al. (2004): forward selection with replacement; counts become weights."""
    counts = [0] * len(outs)
    singles = [objective(o) for o in outs]
    j0 = int(np.argmin(singles))
    counts[j0] = 1
    total = outs[j0].copy()
    best = singles[j0]
    for _ in range(max_steps):
        n = sum(counts)
        cand = [objective((total + o) / (n + 1)) for o in outs]
        j = int(np.argmin(cand))
        if cand[j] >= best - 1e-9:
            break
        counts[j] += 1
        total = total + outs[j]
        best = cand[j]
    return counts


def build_ensemble(
    project: Project,
    refs: list[str] | None = None,
    top: int = 6,
    seeds: int = 1,
    progress: Callable[[str], None] | None = print,
) -> Run | None:
    """Greedy-weighted ensemble of the best runs. Returns None if no combination beats the best run."""
    import yaml

    from .config import ExperimentConfig, apply_overrides
    from .leaderboard import best_run, leaderboard, resolve_ref
    from .pipeline import train

    cfg = project.config
    m, mode = cfg.metric, cfg.metric_mode
    say = progress or (lambda s: None)
    if seeds > 1:
        b = best_run(project, kind="neural")
        if b is None:
            raise ValueError("--seeds needs a completed neural run.")
        base_run = Run(project.runs_dir / b["id"])
        base = yaml.safe_load(base_run.config_path.read_text())
        for i in range(1, seeds):
            exp = ExperimentConfig.model_validate(
                apply_overrides(base, {"train.seed": int(base["train"]["seed"]) + i})
            )
            say(f"seed replica {i}/{seeds - 1} of {base_run.id}")
            train(
                project,
                exp,
                name=f"seed{i}-{base_run.id}"[:40],
                parent=base_run.id,
                tags=["seed"],
                hypothesis=f"seed replica {i} of {base_run.id} (for ensembling)",
                progress=None,
            )
    if refs:
        members = [resolve_ref(project, r) for r in refs]
    else:
        rows = [
            r
            for r in leaderboard(project)
            if r.get("same_split") is not False
            and r["kind"] in ("neural", "baseline", "bag")
            and r.get("model") != "baseline:dummy"
            and not r.get("superseded_by")
            and r.get(f"val_{m}") is not None
            and not (cfg.modality == "image" and r["kind"] == "baseline")
        ]
        members = [Run(project.runs_dir / r["id"]) for r in rows[: max(top, seeds)]]
    if len(members) < 2:
        raise ValueError("An ensemble needs at least two completed runs on the current split.")
    cache: list[tuple[Run, np.ndarray]] = []
    y: np.ndarray | None = None
    for r in members:
        say(f"collecting val predictions from {r.id}")
        o, yy = member_outputs(project, r, "val")
        cache.append((r, o))
        y = yy if y is None else y
    assert y is not None
    n_classes = cache[0][1].shape[1] if cfg.task == "classification" else None
    if cfg.task == "classification":
        sel_metric = "log_loss"  # smooth proxy: accuracy-style metrics overfit a small val set
    else:
        sel_metric = m if m in ("rmse", "mae") else "rmse"

    def objective(out: np.ndarray) -> float:
        return float(
            compute_metrics(cfg.task, y, out, n_classes, outputs_are_probs=True)[sel_metric]
        )

    seed_spread = None
    seed_group: list[tuple[Run, np.ndarray]] = []
    seeded = {c[0].meta.get("parent") for c in cache if "seed" in c[0].meta.get("tags", [])}
    anchor = next((c[0].id for c in cache if c[0].id in seeded), None)
    if anchor is not None:
        seed_group = [
            c
            for c in cache
            if c[0].id == anchor
            or (c[0].meta.get("parent") == anchor and "seed" in c[0].meta.get("tags", []))
        ]
        vals = [
            float(compute_metrics(cfg.task, y, c[1], n_classes, outputs_are_probs=True)[m])
            for c in seed_group
        ]
        if len(vals) > 1:
            seed_spread = {"run": anchor, "min": min(vals), "max": max(vals), "n": len(vals)}
            say(
                f"seed spread of {anchor}'s recipe: val_{m} {min(vals):.4f}..{max(vals):.4f} over {len(vals)} "
                "seeds (single-run differences smaller than this are luck)"
            )
    counts = _greedy_weights([c[1] for c in cache], objective)
    chosen = [(c[0], c[1], k) for c, k in zip(cache, counts) if k > 0]
    if len(chosen) < 2 and seed_spread is not None:
        # val prefers the luckiest seed; the equal-weight seed average is the robust choice
        chosen = [(c[0], c[1], 1) for c in seed_group]
        say(
            "val prefers the luckiest single seed; recording the equal-weight seed average instead "
            "(more robust on unseen data)."
        )
    if len(chosen) < 2:
        say(
            f"no ensemble beats the best single run ({chosen[0][0].id}) on val {sel_metric}; nothing recorded"
        )
        return None
    avg = _average([c[1] for c in chosen], [float(c[2]) for c in chosen])
    metrics = compute_metrics(cfg.task, y, avg, n_classes, outputs_are_probs=True)
    se = bootstrap_se(cfg.task, m, y, avg, n_classes)
    ids = [c[0].id for c in chosen]
    weights = [float(c[2]) for c in chosen]
    run = Run.create(
        project.runs_dir,
        "ensemble",
        kind="ensemble",
        project_root=project.root,
        hypothesis="greedy-weighted ensemble of "
        + ", ".join(f"{i} x{int(w)}" for i, w in zip(ids, weights)),
        extra_meta={
            "members": ids,
            "weights": weights,
            "model": f"ensemble[{len(ids)}]",
            "split_hash": chosen[0][0].meta.get("split_hash"),
            "data_fingerprint": chosen[0][0].meta.get("data_fingerprint"),
        },
    )
    run.write_summary(
        {
            "kind": "ensemble",
            "model": f"ensemble[{len(ids)}]",
            "members": ids,
            "weights": weights,
            "selection_metric": sel_metric,
            "primary_metric": m,
            "metric_mode": mode,
            "val": metrics,
            "val_se": {m: se},
            "member_val": {c[0].id: c[0].summary.get("val", {}).get(m) for c in chosen},
            "seed_spread": seed_spread,
            "note": "weights chosen greedily on val (mildly optimistic); confirm on test once.",
        }
    )
    write_json(run.diagnosis_path, {"verdict": "ensemble", "findings": [], "next_experiments": []})
    run.set_status("completed")
    say(f"ensemble {dict(zip(ids, weights))}: val_{m}={metrics[m]:.4f} ±{se:.4f} -> {run.id}")
    return run
