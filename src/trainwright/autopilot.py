"""``trainwright auto``: the whole modelling lifecycle in one time-budgeted command.

Agents under a time limit get the most out of a strong, standard search that the tool runs end
to end, which leaves their own effort for what a tool cannot do: understanding the data, writing
``features.py``, spotting leakage and judging whether the result is good enough to ship.

The plan, each step skipped if already done (so ``auto`` is safe to rerun after edits):

1. profile (auto-drops id-like/constant columns) and freeze the split
2. reference baselines for the modality
3. a short, modality-specific ladder of strong neural recipes, each with its own time cap
4. seed replicas of the best neural recipe when time allows (cheap variance reduction)
5. a greedy val-weighted ensemble over everything
6. finalize the val-best model on train+val, so ``trainwright predict`` serves it

The test split is never touched here: evaluate it once yourself at the end.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ._utils import fmt_seconds, write_json
from .project import Project

Say = Callable[[str], None]

# (name, overrides, share of the search budget, hypothesis)
LADDERS: dict[str, list[tuple[str, dict[str, Any], float, str]]] = {
    "tabular": [
        ("mlp", {}, 0.30, "autopilot: default MLP (periodic numeric embeddings)"),
        (
            "mlp-plain",
            {"model.params.num_embedding": "none"},
            0.30,
            "autopilot: MLP on raw quantile features (smooth targets, tiny noisy data)",
        ),
    ],
    "image": [
        (
            "probe",
            {"model.name": "image_probe"},
            0.15,
            "autopilot: frozen DINOv2-small linear probe (features cached: fast on small image sets)",
        ),
        (
            "probe-base",
            {"model.name": "image_probe", "model.params.backbone": "dinov2_base"},
            0.20,
            "autopilot: frozen DINOv2-base linear probe (richer features, still one backbone pass)",
        ),
        (
            "finetune",
            {"model.name": "image_pretrained"},
            0.45,
            "autopilot: fine-tune a pretrained CNN (wins when images differ from natural photos)",
        ),
    ],
    "text": [
        (
            "encoder",
            {"model.name": "text_pretrained"},
            0.30,
            "autopilot: fine-tune a small pretrained sentence encoder (bge-small, 33M)",
        ),
        (
            "encoder-base",
            {"model.name": "text_pretrained", "model.params.encoder": "FacebookAI/roberta-base"},
            0.45,
            "autopilot: fine-tune a base-size encoder (roberta-base, 125M): nuance, sarcasm, tone",
        ),
    ],
}


def state_path(project: Project) -> Any:
    return project.state_dir / "auto.json"


def log_path(project: Project) -> Any:
    return project.state_dir / "auto.log"


def read_state(project: Project) -> dict[str, Any] | None:
    """The background/foreground auto run's state, with a liveness check on its pid."""
    import json
    import os

    p = state_path(project)
    if not p.exists():
        return None
    st = json.loads(p.read_text())
    if st.get("status") == "running":
        try:
            os.kill(int(st["pid"]), 0)
        except (OSError, ValueError, KeyError):
            st["status"] = "died"
            st["error"] = (
                "the auto process exited without finishing (killed or crashed); see auto.log"
            )
    return st


def profile_project(project: Project, keep_ids: bool = False) -> dict[str, Any]:
    """Profile the data, auto-drop id-like/constant columns into data.drop, save profile.json."""
    from .data.profile import profile as _profile
    from .pipeline import get_records

    records = get_records(project)
    prof = _profile(records, project.config, project.state_dir)
    prof["data_fingerprint"] = records.fingerprint
    cfg = project.config
    auto_dropped: list[str] = []
    if "tabular" in prof:
        # date/time text columns become numeric parts (year, month, weekday, hour, ...)
        from .preprocessing import detect_date_format

        found: dict[str, str] = {}
        for c, p in prof["tabular"]["columns"].items():
            if p.get("kind") == "datetime-like string" and c not in cfg.data.date_columns:
                if c in (cfg.data.target, cfg.data.time_column):
                    continue
                fmt = (
                    detect_date_format(records.table[c].tolist())
                    if records.table is not None
                    else None
                )
                if fmt:
                    found[c] = fmt
        if found:
            cfg.data.date_columns = {**cfg.data.date_columns, **found}
            project.save()
            prof["date_columns"] = found
            prof["recommendations"].insert(
                0,
                f"Expanded date columns {sorted(found)} into year/month/day/weekday/hour/dayofyear/ts "
                "features (data.date_columns). Consider data.time_column for a chronological split "
                "if you will predict the future.",
            )
    if "tabular" in prof and not keep_ids:
        for c, p in prof["tabular"]["columns"].items():
            if c not in cfg.data.drop and any(
                f.startswith(("id-like", "constant")) for f in p.get("flags", [])
            ):
                auto_dropped.append(c)
        if auto_dropped:
            cfg.data.drop = [*cfg.data.drop, *auto_dropped]
            project.save()
            prof["auto_dropped"] = auto_dropped
            prof["tabular"]["suggested_drop"] = [
                c for c in prof["tabular"]["suggested_drop"] if c not in auto_dropped
            ]
            prof["recommendations"] = [
                r for r in prof["recommendations"] if not r.startswith("Add to data.drop")
            ]
            if prof["tabular"]["suggested_drop"]:
                prof["recommendations"].insert(
                    0,
                    f"Consider adding to data.drop: {prof['tabular']['suggested_drop']} (leakage suspects).",
                )
    write_json(project.profile_path, prof)
    return prof


def _finalize_cost(run_row: dict[str, Any], project: Project) -> float:
    """Rough seconds to refit a run on train+val (members summed for ensembles)."""
    from .tracking import Run

    run = Run(project.runs_dir / run_row["id"])
    if run.meta.get("bag"):
        return 0.0  # fold models already trained on train+val
    members = run.meta.get("members")
    if members:
        return sum(
            _finalize_cost({"id": m}, project) for m in members
        )  # baselines refit in seconds
    s = run.summary
    t = float(s.get("train_time_s") or 0.0)
    if run.meta.get("kind") != "neural":
        return t * 1.3
    frac = (s.get("best_epoch") or 1) / max(1, s.get("epochs_run") or 1)
    return t * frac * 1.25  # +val data, no per-epoch val pass saved


def autopilot(
    project: Project,
    budget_s: float,
    overrides: dict[str, Any] | None = None,
    finish: bool = True,
    seeds: bool = True,
    progress: Say | None = print,
    bag: bool = True,
) -> dict[str, Any]:
    """Run the search; records running/done/failed in .trainwright/auto.json for `wait`/`status`."""
    import os
    import traceback

    t0 = time.time()
    try:
        res = _autopilot(project, budget_s, overrides, finish, seeds, progress, bag)
    except BaseException as e:
        write_json(
            state_path(project),
            {"status": "failed", "pid": os.getpid(), "started": t0, "finished": time.time(),
             "error": f"{type(e).__name__}: {e}"},
        )  # fmt: skip
        with log_path(project).open("a") as f:
            f.write(traceback.format_exc())
        raise
    write_json(
        state_path(project),
        {"status": "done", "pid": os.getpid(), "started": t0, "finished": time.time(),
         "budget_s": budget_s, "best": (res.get("best") or {}).get("id"),
         "final": res.get("final")},
    )  # fmt: skip
    return res


def _autopilot(
    project: Project,
    budget_s: float,
    overrides: dict[str, Any] | None = None,
    finish: bool = True,
    seeds: bool = True,
    progress: Say | None = print,
    bag: bool = True,
) -> dict[str, Any]:
    import os

    from .baselines import BASELINES, run_baselines
    from .config import resolve_experiment
    from .ensemble import build_ensemble
    from .leaderboard import best_run, leaderboard
    from .pipeline import ensure_split, finalize, get_records, train
    from .tracking import Run

    t0 = time.time()
    deadline = t0 + budget_s
    project.ensure_dirs()
    prev = read_state(project)
    if prev and prev.get("status") == "running" and int(prev.get("pid", -1)) != os.getpid():
        raise RuntimeError(
            f"auto is already running (pid {prev['pid']}); `trainwright wait` for it to finish"
        )
    write_json(
        state_path(project),
        {"status": "running", "pid": os.getpid(), "started": t0, "deadline": deadline,
         "budget_s": budget_s},
    )  # fmt: skip
    lp = log_path(project)

    def say(msg: str) -> None:
        with lp.open("a") as f:
            f.write(msg + "\n")
        if progress:
            progress(msg)

    reserve = 0.2 * budget_s if finish else 0.0  # kept back for finalize
    cfg = project.config
    overrides = dict(overrides or {})
    log: list[str] = []

    def note(msg: str) -> None:
        log.append(msg)
        say(f"[auto {fmt_seconds(time.time() - t0)}] {msg}")

    def left() -> float:
        return deadline - reserve - time.time()

    project.ensure_dirs()
    if not project.profile_path.exists():
        prof = profile_project(project)
        note(
            "profiled data"
            + (f"; auto-dropped {prof['auto_dropped']}" if prof.get("auto_dropped") else "")
        )
        for leak in (prof.get("tabular") or {}).get("leakage_suspects", []):
            note(f"LEAK? {leak['column']}: {leak['why']} (check before trusting any score)")
    sp, created = ensure_split(project)
    if created:
        note(f"froze {sp['strategy']} split {sp['counts']}")

    from .featurehook import FeatureHook

    hook = FeatureHook.from_project(project.root) if cfg.modality == "tabular" else None
    sha = hook.sha if hook is not None else None
    split_hash = sp["split_hash"]

    def current(r: dict[str, Any]) -> bool:
        meta = Run(project.runs_dir / r["id"]).meta
        return meta.get("features_sha") == sha and meta.get("split_hash") == split_hash

    rows = [r for r in leaderboard(project) if current(r)]
    done_models = {r.get("model") for r in rows}
    if sha:
        note(f"features.py in use (sha {sha}); steps already run with it are skipped")
    todo = [b for b in BASELINES[cfg.modality] if f"baseline:{b}" not in done_models]
    if todo:
        run_baselines(project, todo, progress=None)
        note(f"baselines: {', '.join(todo)}")

    hyps = {r.get("hypothesis") for r in rows}
    n_rows = len(get_records(project).y)
    for name, ov, share, hyp in LADDERS[cfg.modality]:
        if hyp in hyps:
            continue
        cap = min(share * budget_s, left())
        if cap < min(30.0, 0.15 * budget_s):
            note(f"skipped {name}: not enough time left")
            continue
        try:
            exp = resolve_experiment(cfg, overrides={**ov, **overrides})
            exp = exp.model_copy(
                update={"train": exp.train.model_copy(update={"time_budget": f"{int(cap)}s"})}
            )
            run = train(
                project, exp, name=f"auto-{name}", hypothesis=hyp, tags=["auto"], progress=None
            )
            m = cfg.metric
            note(
                f"{name}: val_{m}={run.summary['val'].get(m, float('nan')):.4f} "
                f"in {fmt_seconds(run.summary.get('train_time_s') or 0)} ({run.summary.get('stop_reason')})"
            )
        except Exception as e:  # one failing recipe must not sink the search
            note(f"{name} failed: {type(e).__name__}: {str(e)[:200]}")

    # spend spare budget on a recipe search around the best neural run (tabular: cheap runs)
    bn = best_run(project, kind="neural")
    if cfg.modality == "tabular" and bn is not None and current(bn):
        per = float(bn.get("train_time_s") or 0.0) + 5
        spare = left() * 0.55
        if spare > 6 * per and spare > 60:
            from .tune import run_tune

            res = run_tune(project, f"{int(spare)}s", from_run=bn["id"], n_trials=8, progress=None)
            b2 = best_run(project, kind="neural")
            m = cfg.metric
            note(
                f"tune: {len(res.get('trials', []))} trials in {fmt_seconds(spare)}; best neural now "
                f"{b2['id'] if b2 else '?'} val_{m}={(b2 or {}).get(f'val_{m}')}"
            )

    # cross-validated bag of the best neural recipe: k models on train+val, out-of-fold val
    bagged = False
    bn = best_run(project, kind="neural")
    if (
        bag
        and cfg.modality == "tabular"
        and not cfg.data.time_column
        and bn is not None
        and current(bn)
        and n_rows <= 100_000
    ):
        per = float(bn.get("train_time_s") or 0.0) + 5
        if left() > 5 * per * 1.3:
            from .bagging import build_bag

            try:
                b = build_bag(project, bn["id"], k=5, progress=None)
                m = cfg.metric
                note(
                    f"5-fold bag of {bn['id']}: val_{m}={b.summary['val'][m]:.4f} (out-of-fold), "
                    f"oof over {b.summary['n_oof']} rows {b.summary['oof'][m]:.4f}"
                )
                bagged = True
            except Exception as e:
                note(f"bag failed: {type(e).__name__}: {str(e)[:200]}")

    # seed replicas of the best neural recipe, then the ensemble over everything
    n_seeds = 1
    bn = best_run(project, kind="neural")
    if seeds and not bagged and bn is not None and current(bn):
        per = float(bn.get("train_time_s") or 0.0) + 5
        n_seeds = int(min(5, max(1, left() * 0.7 // max(per, 1.0))))
        if n_rows > 200_000:
            n_seeds = min(n_seeds, 2)
    try:
        ens = build_ensemble(project, top=6, seeds=n_seeds, progress=None)
        if n_seeds > 1:
            note(f"trained {n_seeds - 1} seed replica(s) of {bn['id'] if bn else '?'}")
        if ens is not None:
            m = cfg.metric
            note(f"ensemble {ens.meta.get('model')}: val_{m}={ens.summary['val'][m]:.4f}")
    except ValueError as e:
        note(f"no ensemble: {e}")

    best = best_run(project)
    result: dict[str, Any] = {"best": best, "log": log}
    if best is None:
        return result
    m = cfg.metric
    note(f"best on val: {best['id']} ({best.get('model')}) val_{m}={best.get(f'val_{m}'):.4f}")
    if finish:
        remaining = deadline - time.time()
        cands = [best] + [
            r
            for r in leaderboard(project)
            if r["id"] != best["id"]
            and r.get("tied_with_leader")
            and r.get("same_split") is not False
        ]
        pick = next(
            (r for r in cands if _finalize_cost(r, project) <= max(remaining, 0) * 1.5), None
        )
        if pick is None:
            note(
                "not enough budget left to refit on train+val; `trainwright predict` will use the best run as is"
            )
        else:
            if pick["id"] != best["id"]:
                note(
                    f"finalizing {pick['id']} (statistically tied with the leader, cheaper to refit)"
                )
            final = finalize(project, Run(project.runs_dir / pick["id"]), False, progress=None)
            result["final"] = final.id
            note(
                f"finalized {pick['id']} on train+val -> {final.id} (served by `trainwright predict`)"
            )
    result["seconds"] = round(time.time() - t0, 1)
    write_json(project.state_dir / "autopilot.json", result)
    return result
