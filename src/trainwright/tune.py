"""Automated recipe search: successive halving over a modality-specific recipe space.

Starts from the best (or given) run's config, samples ``n_trials`` variations (the unchanged
base is always one of them), trains all of them for a quarter of the epochs, keeps the best
half for half the epochs, and the best two for the full schedule. Every trial is an ordinary
run (tagged ``tune``) with a hypothesis listing exactly what it changed, so the leaderboard and
journal stay readable and any trial can be continued with ``--from-run``.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from typing import Any

import yaml

from ._utils import fmt_seconds, parse_duration
from .config import ExperimentConfig, apply_overrides, default_experiment
from .leaderboard import best_run, resolve_ref
from .pipeline import train
from .project import Project
from .tracking import Run

SPACES: dict[str, dict[str, list[Any]]] = {
    "tabular": {
        "optim.lr": [3e-4, 1e-3, 3e-3],
        "optim.weight_decay": [1e-6, 1e-5, 1e-4, 1e-3],
        "model.params.dropout": [0.0, 0.1, 0.2, 0.3],
        "model.params.hidden": [128, 256, 512],
        "model.params.depth": [2, 3, 4],
        "model.params.num_embedding": ["none", "periodic"],
        "model.params.k_ensemble": [1, 1, 16],  # TabM-mini: helps on some datasets, neutral on average
        "train.batch_size": [128, 256, 512],
    },
    "image": {
        "optim.lr": [3e-4, 1e-3, 3e-3],
        "optim.weight_decay": [0.01, 0.05, 0.1],
        "optim.backbone_lr_mult": [0.05, 0.1, 0.3],
        "preprocess.augment": ["light", "medium", "strong"],
        "train.label_smoothing": [0.0, 0.1],
    },
    "text": {
        "optim.lr": [1e-3, 3e-3, 1e-2],
        "model.params.dropout": [0.2, 0.3, 0.5],
        "model.params.embed_dim": [64, 128, 256],
        "preprocess.min_token_freq": [1, 2, 3],
        "preprocess.ngrams": [1, 2, 3],
    },
}


def _get(d: dict[str, Any], dotted: str) -> Any:
    for p in dotted.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def sample_candidates(
    base: dict[str, Any], modality: str, model_name: str, n: int, seed: int
) -> list[dict[str, Any]]:
    space = dict(SPACES[modality])
    if modality == "tabular" and model_name != "tabular_mlp":
        space = {k: v for k, v in space.items() if not k.startswith("model.")}
    if modality == "text" and model_name != "text_bag":
        space = {k: v for k, v in space.items() if k != "model.params.embed_dim"}
    rng = random.Random(seed)
    seen: set[tuple[tuple[str, Any], ...]] = {()}
    out: list[dict[str, Any]] = [{}]
    attempts = 0
    while len(out) < n and attempts < 500:
        attempts += 1
        k = rng.choice([1, 2, 2, 3])  # mostly small, local moves from the base recipe
        keys = rng.sample(sorted(space), min(k, len(space)))
        ch = {}
        for key in keys:
            opts = [v for v in space[key] if v != _get(base, key)]
            if opts:
                ch[key] = rng.choice(opts)
        sig = tuple(sorted(ch.items()))
        if ch and sig not in seen:
            seen.add(sig)
            out.append(ch)
    return out


def run_tune(
    project: Project,
    budget: str = "20m",
    from_run: str | None = None,
    n_trials: int = 8,
    seed: int = 0,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    cfg = project.config
    total = parse_duration(budget) or 1200.0
    t0 = time.time()
    base_run: Run | None
    if from_run:
        base_run = resolve_ref(project, from_run)
    else:
        b = best_run(project, kind="neural")
        base_run = resolve_ref(project, b["id"]) if b else None
    if base_run is not None:
        base = yaml.safe_load(base_run.config_path.read_text())
        parent = base_run.id
    else:
        base = default_experiment(cfg).model_dump()
        parent = None
    E = int(base["train"]["max_epochs"])
    rungs = sorted({max(1, E // 4), max(1, E // 2), E})
    cands = sample_candidates(base, cfg.modality, base["model"]["name"], n_trials, seed)
    say = progress or (lambda s: None)
    say(
        f"tune: {len(cands)} candidates from {parent or 'defaults'}, rungs {rungs} epochs, budget {fmt_seconds(total)}"
    )
    sign = 1 if cfg.metric_mode == "max" else -1
    history: list[dict[str, Any]] = []
    alive = list(range(len(cands)))
    for ri, epochs in enumerate(rungs):
        scores: dict[int, float] = {}
        for qi, ci in enumerate(alive):
            remaining = total - (time.time() - t0)
            if remaining < 15:
                say("tune: budget exhausted")
                break
            share = remaining / max(
                1,
                (len(alive) - qi)
                + sum(math.ceil(len(alive) / 2 ** (j + 1)) for j in range(len(rungs) - ri - 1)),
            )
            ch = cands[ci]
            ov = {
                **ch,
                "train.max_epochs": epochs,
                "train.patience": max(
                    3, min(int(base["train"].get("patience") or epochs), epochs // 3 + 2)
                ),
                "train.time_budget": f"{max(10, int(share))}s",
                # keep the step floor proportional so early rungs stay short on small data
                "train.min_steps": int(int(base["train"].get("min_steps") or 0) * epochs / E),
            }
            exp = ExperimentConfig.model_validate(apply_overrides(base, ov))
            desc = ", ".join(f"{k}={v}" for k, v in ch.items()) or "base recipe"
            say(f"\n--- tune rung {ri + 1}/{len(rungs)} ({epochs} ep) trial {ci}: {desc}")
            try:
                run = train(
                    project,
                    exp,
                    name=f"tune-r{ri + 1}-t{ci}",
                    parent=parent,
                    tags=["tune", f"rung{ri + 1}"],
                    hypothesis=f"tune rung {ri + 1} ({epochs} ep): {desc}",
                    progress=None,
                )
                score = run.summary.get("val", {}).get(cfg.metric)
            except Exception as e:  # a failing candidate must not kill the search
                say(f"trial {ci} failed: {e}")
                run, score = None, None
            ok = isinstance(score, (int, float)) and math.isfinite(score)
            scores[ci] = sign * score if ok else -math.inf
            history.append(
                {
                    "rung": ri + 1,
                    "epochs": epochs,
                    "trial": ci,
                    "changes": ch,
                    "run": run.id if run else None,
                    f"val_{cfg.metric}": score,
                }
            )
            say(
                f"    -> val_{cfg.metric}={score if score is None else round(score, 4)}  ({run.id if run else 'failed'})"
            )
        if not scores:
            break
        ranked = sorted(scores, key=lambda c: -scores[c])
        keep = max(1, math.ceil(len(ranked) / 2)) if ri < len(rungs) - 2 else min(2, len(ranked))
        alive = ranked[:keep] if ri < len(rungs) - 1 else ranked
    final_rung = max((h["rung"] for h in history if h[f"val_{cfg.metric}"] is not None), default=0)
    finals = [h for h in history if h["rung"] == final_rung and h[f"val_{cfg.metric}"] is not None]
    finals.sort(key=lambda h: -sign * h[f"val_{cfg.metric}"])
    best = finals[0] if finals else None
    base_score = base_run.summary.get("val", {}).get(cfg.metric) if base_run else None
    return {
        "base_run": parent,
        "base_score": base_score,
        "best_run": best["run"] if best else None,
        "best_changes": best["changes"] if best else None,
        "best_score": best[f"val_{cfg.metric}"] if best else None,
        "trials": history,
        "elapsed_s": round(time.time() - t0, 1),
        "rungs": rungs,
    }


def format_tune(res: dict[str, Any], metric: str) -> str:
    lines = [f"tune finished in {fmt_seconds(res['elapsed_s'])}; rungs {res['rungs']} epochs"]
    for h in res["trials"]:
        v = h[f"val_{metric}"]
        lines.append(
            f"  rung {h['rung']} trial {h['trial']:>2} {('%.4f' % v) if v is not None else 'failed':>8}  "
            f"{h['run'] or '-':<24} {', '.join(f'{k}={x}' for k, x in h['changes'].items()) or 'base'}"
        )
    if res["best_run"]:
        lines.append(
            f"best: {res['best_run']} val_{metric}={res['best_score']:.4f} with {res['best_changes'] or 'base recipe'}"
        )
        if res.get("base_score") is not None:
            lines.append(
                f"base run {res['base_run']}: val_{metric}={res['base_score']:.4f} (full schedule)"
            )
    return "\n".join(lines)
