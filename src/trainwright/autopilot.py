"""``trainwright auto``: the whole lifecycle in one time-boxed command.

profile -> split -> baselines -> recipe ladder -> tune/bag/seed replicas -> ensemble -> finalize.
Steps already current for the split and ``features.py`` are skipped, so reruns are cheap. Heavy
steps run in subprocesses with hard deadlines. The test split is never touched.
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
    "segmentation": [
        (
            "seg-probe",
            {"model.name": "seg_probe"},
            0.25,
            "autopilot: frozen DINOv2-small patch features + linear head (strong with few labels)",
        ),
        (
            "segmenter",
            {"model.name": "image_segmenter"},
            0.35,
            "autopilot: fine-tune LR-ASPP MobileNetV3 (fast, cheap to serve)",
        ),
        (
            "deeplab",
            {"model.name": "image_segmenter", "model.params.arch": "deeplabv3_resnet50"},
            0.40,
            "autopilot: fine-tune DeepLabV3-ResNet50 (slower, usually more accurate)",
        ),
    ],
    "text": [
        (
            "encoder",
            {"model.name": "text_pretrained"},
            0.45,
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
        texts = [
            c
            for c, p in prof["tabular"]["columns"].items()
            if p.get("kind") == "text"
            and c not in cfg.data.text_features
            and c not in cfg.data.drop
            and c != cfg.data.target
        ]
        if texts:
            cfg.data.text_features = [*cfg.data.text_features, *texts]
            project.save()
            prof["text_features"] = texts
            prof["recommendations"].insert(
                0,
                f"Free-text columns {texts} are turned into 32 TF-IDF+SVD features each "
                "(data.text_features); the raw text itself is not a model input.",
            )
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


def _guarded(
    project: Project, args: list[str], timeout_s: float
) -> tuple[dict[str, Any] | None, str | None]:
    """Run one trainwright command in its own process with a hard deadline.

    A step that overruns or hangs (a stuck accelerator, a model too big for the machine) is
    killed and its half-finished runs are marked failed, so the search always moves on and
    always reaches finalize with the best result found so far. -> (result, error)
    """
    import json
    import os
    import signal
    import subprocess
    import sys

    from .tracking import list_runs

    started = time.time()
    with (project.state_dir / "auto.stdout.log").open("a") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "trainwright", "--json", *args],
            cwd=project.root,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=max(30.0, timeout_s))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            for r in list_runs(project.runs_dir):
                if r.meta.get("status") == "running" and r.dir.stat().st_mtime >= started - 5:
                    r.update_meta(
                        error=f"killed by auto: exceeded its {fmt_seconds(timeout_s)} cap"
                    )
                    r.set_status("failed")
            return None, f"killed after {fmt_seconds(timeout_s)} (over its time cap)"
    start = out.rfind("\n{\n") + 1 if "\n{\n" in out else out.find("{")
    try:
        blob = json.loads(out[start:]) if start >= 0 else {}
    except json.JSONDecodeError:
        blob = {}
    if not blob.get("ok"):
        return None, str(blob.get("error") or f"exit code {proc.returncode}")[:300]
    return blob.get("result") or {}, None


def _ov_args(ov: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in ov.items():
        val = "null" if v is None else (str(v).lower() if isinstance(v, bool) else str(v))
        out += ["-o", f"{k}={val}"]
    return out


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


def _feature_step(
    project: Project, split_hash: str, budget_s: float, note: Callable[[str], None]
) -> dict[str, Any]:
    """Once per split: test the candidates `errors` writes and adopt clear wins (z > 2).

    Returns overrides for the neural runs (a log target). Runs in guarded subprocesses, so a
    slow or failing candidate costs at most its deadline.
    """
    from ._utils import read_json
    from .featurehook import FeatureHook

    def sha() -> str | None:
        hook = FeatureHook.from_project(project.root)
        return hook.sha if hook is not None else None

    mark = project.state_dir / "auto_features.json"
    if mark.exists():
        prev = read_json(mark)
        if prev.get("split_hash") == split_hash and prev.get("features_sha") == sha():
            return dict(prev.get("overrides") or {})
    cap = min(180.0, 0.1 * budget_s)
    rep, err = _guarded(project, ["errors", "--seeds", "1"], cap)
    overrides: dict[str, Any] = {}
    if rep is None:
        note(f"feature search skipped: {err}")
    else:
        wins = [c for c in rep.get("candidates", []) if c.get("verdict") == "better"]
        for c in wins:
            if c["name"] == "log_target":
                overrides["preprocess.target_transform"] = "log1p"
                note(f"adopted log1p target (proxy gain {c['delta']:+.4g} ±{c['se']:.2g})")
        groups = [c for c in wins if c.get("file")]
        if groups:
            best = groups[0]
            kept, err = _guarded(project, ["try", best["file"], "--keep", "--seeds", "2"], cap)
            if kept and kept.get("kept"):
                note(
                    f"adopted {best['name']} into features.py (proxy gain "
                    f"{kept['delta']:+.4g} ±{kept['se']:.2g}): {best['hypothesis']}. Valid only if "
                    "predictions are for groups seen in training; otherwise set data.group_column"
                )
            elif err:
                note(f"{best['name']} not adopted: {err}")
        if not wins:
            tested = ", ".join(c["name"] for c in rep.get("candidates", [])) or "none applicable"
            note(f"feature search: no clear win ({tested}); see `trainwright errors`")
    write_json(mark, {"split_hash": split_hash, "features_sha": sha(), "overrides": overrides})
    return overrides


SAFETY_CAP_S = 7200.0  # without --budget, auto runs until converged; this only stops runaways
MAX_ROUNDS = 4


def autopilot(
    project: Project,
    budget_s: float | None,
    overrides: dict[str, Any] | None = None,
    finish: bool = True,
    seeds: bool = True,
    progress: Say | None = print,
    bag: bool = True,
    features: bool = True,
) -> dict[str, Any]:
    """Run the search; records running/done/failed in .trainwright/auto.json for `wait`/`status`."""
    import os
    import traceback

    t0 = time.time()
    converge = budget_s is None
    budget_s = budget_s or SAFETY_CAP_S
    try:
        res = _autopilot(
            project, budget_s, overrides, finish, seeds, progress, bag, features, converge
        )
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
         "final": res.get("final"), "stopped": res.get("stopped"),
         "mode": "converge" if converge else "budget"},
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
    features: bool = True,
    converge: bool = False,
) -> dict[str, Any]:
    import os

    from .baselines import run_baselines
    from .leaderboard import best_run, leaderboard
    from .pipeline import ensure_split, get_records
    from .tracking import Run, list_runs

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
        dropped = prof.get("auto_dropped") or []
        shown = ", ".join(dropped[:5]) + (f" +{len(dropped) - 5} more" if len(dropped) > 5 else "")
        note("profiled data" + (f"; dropped id-like/constant columns: {shown}" if dropped else ""))
        for leak in (prof.get("tabular") or {}).get("leakage_suspects", []):
            note(f"LEAK? {leak['column']}: {leak['why']} (check before trusting any score)")
    sp, created = ensure_split(project)
    if created:
        note(f"froze {sp['strategy']} split {sp['counts']}")

    if features and cfg.modality == "tabular":
        for ov_key, ov_val in _feature_step(project, sp["split_hash"], budget_s, note).items():
            overrides.setdefault(ov_key, ov_val)

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
    from .baselines import baseline_names

    todo = [b for b in baseline_names(cfg) if f"baseline:{b}" not in done_models]
    if todo:
        run_baselines(project, todo, progress=None)
        note(f"baselines: {', '.join(todo)}")

    hyps = {r.get("hypothesis") for r in rows}
    n_rows = len(get_records(project).y)
    ladder = LADDERS["segmentation" if cfg.task == "segmentation" else cfg.modality]
    hyps_by_name = {n: h for n, _, _, h in ladder}
    for name, ov, share, hyp in ladder:
        if hyp in hyps:
            continue
        cap = min(share * budget_s, left())
        if cap < min(30.0, 0.15 * budget_s):
            note(f"skipped {name}: not enough time left")
            continue
        if name == "encoder-base":
            # ~3.5x the small encoder's epoch time; it needs about 3 epochs to be worth it
            small = next(
                (
                    r
                    for r in leaderboard(project)
                    if r.get("hypothesis") == hyps_by_name.get("encoder")
                ),
                None,
            )
            if small and small.get("epochs_run"):
                epoch_s = float(small.get("train_time_s") or 0) / max(1, int(small["epochs_run"]))
                if 3 * 3.5 * epoch_s > cap:
                    note(
                        f"skipped {name}: ~{fmt_seconds(3.5 * epoch_s)}/epoch here, 3 epochs won't fit in "
                        f"{fmt_seconds(cap)} (bigger --budget to try it)"
                    )
                    continue
        args = ["train", "--name", f"auto-{name}", "-H", hyp, "--tag", "auto"]
        args += _ov_args({**ov, **overrides, "train.time_budget": f"{int(cap)}s"})
        res, err = _guarded(project, args, cap * 1.3 + 120)  # + model load / final eval
        if res is None:  # one failing or stuck recipe must not sink the search
            note(f"{name} failed: {err}")
            continue
        summ = res.get("summary") or {}
        m = cfg.metric
        note(
            f"{name}: val_{m}={(summ.get('val') or {}).get(m, float('nan')):.4f} "
            f"in {fmt_seconds(summ.get('train_time_s') or 0)} ({summ.get('stop_reason')})"
        )

    # spend spare budget on a recipe search around the best neural run (tabular: cheap runs)
    bn = best_run(project, kind="neural")
    if cfg.modality == "tabular" and bn is not None and current(bn):
        per = float(bn.get("train_time_s") or 0.0) + 5
        spare = left() * 0.55
        if spare > 6 * per and spare > 60:
            args = ["tune", "--budget", f"{int(spare)}s", "--from-run", bn["id"], "--trials", "8"]
            res, err = _guarded(project, args, spare * 1.3 + 120)
            b2 = best_run(project, kind="neural")
            m = cfg.metric
            note(
                f"tune failed: {err}"
                if res is None
                else f"tune: {len(res.get('trials', []))} trials in {fmt_seconds(float(res.get('elapsed_s') or 0))}; best neural "
                f"now {b2['id'] if b2 else '?'} val_{m}={float((b2 or {}).get(f'val_{m}') or 'nan'):.4f}"
            )

    # cross-validated bags: the best neural recipe first, then (while the budget lasts) the next
    # best distinct recipes, so the final ensemble blends different models, not copies of one.
    # A bag supersedes its parent, so best_run(kind="neural") moves on to the next recipe.
    bagged = False
    m = cfg.metric
    sign = 1.0 if cfg.metric_mode == "max" else -1.0
    first: float | None = None
    for nb in range(3):
        bn = best_run(project, kind="neural")
        if not (
            bag
            and cfg.modality == "tabular"
            and not cfg.data.time_column
            and bn is not None
            and current(bn)
            and n_rows <= 100_000
        ):
            break
        score = float(bn.get(f"val_{m}") or float("nan"))
        if first is not None and not (sign * (score - first) >= -0.03 * abs(first)):
            break  # the next recipe is clearly worse: a bag of it won't help the blend
        per = float(bn.get("train_time_s") or 0.0) + 5
        if left() <= 5 * per * 1.3:
            break
        cap = left() * (0.8 if nb == 0 else 0.6)
        res, err = _guarded(
            project,
            ["bag", bn["id"], "--folds", "5", "--budget", f"{int(cap)}s"],
            cap * 1.3 + 120,
        )
        if res is None:
            note(f"bag failed: {err}")
            break
        summ = res.get("summary") or {}
        note(
            f"5-fold bag of {bn['id']}: val_{m}={summ['val'][m]:.4f} (out-of-fold), "
            f"oof over {summ['n_oof']} rows {summ['oof'][m]:.4f}"
        )
        bagged = True
        first = score if first is None else first

    # seed replicas of the best neural recipe, one at a time while the budget allows (always
    # leaving room to finalize), then the greedy ensemble over everything
    bn = best_run(project, kind="neural")
    replicas = 0
    seg = cfg.task == "segmentation"  # no averaging of segmentation outputs: no blends yet
    if seeds and not bagged and not seg and bn is not None and current(bn):
        per = float(bn.get("train_time_s") or 0.0) + 5
        base_seed = int(Run(project.runs_dir / bn["id"]).meta.get("seed") or 0)
        max_replicas = 2 if n_rows > 200_000 else 4
        while replicas < max_replicas:
            # finalizing an ensemble refits every member: keep room for that
            spare = deadline - time.time() - (replicas + 2) * per - 0.05 * budget_s
            cap = min(per * 1.3, spare)
            if cap < max(30.0, 0.6 * per):
                break
            k = replicas + 1
            args = [
                "train",
                "--from-run",
                bn["id"],
                "--tag",
                "seed",
                "--name",
                f"seed{k}-{bn['id']}"[:40],
            ]
            args += ["-H", f"seed replica {k} of {bn['id']} (for ensembling)"]
            args += _ov_args(
                {"train.seed": base_seed + 100 + k, "train.time_budget": f"{int(cap)}s"}
            )
            res, err = _guarded(project, args, cap * 1.3 + 120)
            if res is None:
                note(f"seed replica {k} failed: {err}")
                break
            replicas += 1
        if replicas:
            note(f"trained {replicas} seed replica(s) of {bn['id']}")
    res, err = (
        (None, "segmentation outputs can't be averaged yet")
        if seg
        else _guarded(project, ["ensemble", "--top", "8"], max(60.0, 0.1 * budget_s) + 120)
    )
    if res is None:
        note(f"no ensemble: {err}")
    elif res.get("run"):
        summ = res.get("summary") or {}
        m = cfg.metric
        note(f"ensemble {summ.get('model')}: val_{m}={summ['val'][m]:.4f}")

    # without a budget: more rounds (tune around the leader, bag the new best recipe, re-blend)
    # while each round beats the leader by more than its standard error
    stopped = "budget"
    if converge:
        stopped = "converged"
        if cfg.modality != "tabular":
            note("recipe ladder complete")
        for rnd in range(1, MAX_ROUNDS + 1) if cfg.modality == "tabular" else ():
            lead, bn = best_run(project), best_run(project, kind="neural")
            if lead is None or bn is None or not current(bn):
                break
            before = float(lead[f"val_{m}"])
            se = float(lead.get("val_se") or 0.0)
            per = float(bn.get("train_time_s") or 0.0) + 5
            tb = min(max(120.0, per * 8 * 3), left() * 0.5)
            if tb < 6 * per:
                stopped = "safety cap"
                note("stopped: the safety cap is close")
                break
            res, err = _guarded(
                project,
                ["tune", "--budget", f"{int(tb)}s", "--from-run", bn["id"], "--trials", "8"],
                tb * 1.3 + 120,
            )
            if res is None:
                note(f"round {rnd} tune failed: {err}")
                break
            new_best = best_run(project, kind="neural")
            if (
                bag
                and new_best is not None
                and new_best["id"] != bn["id"]
                and not cfg.data.time_column
                and n_rows <= 100_000
                and left() > 5 * per * 1.3
            ):
                cap = left() * 0.6
                _guarded(
                    project,
                    ["bag", new_best["id"], "--folds", "5", "--budget", f"{int(cap)}s"],
                    cap * 1.3 + 120,
                )
            _guarded(project, ["ensemble", "--top", "8"], max(60.0, 0.1 * budget_s) + 120)
            after_row = best_run(project) or lead
            after = float(after_row[f"val_{m}"])
            gain = sign * (after - before)
            note(
                f"round {rnd}: best val_{m} {before:.4f} -> {after:.4f} "
                f"(gain {gain:+.4f}, noise ±{se:.4f})"
            )
            if gain <= se:
                note("converged: the last round did not beat the leader by more than noise")
                break
        else:
            if cfg.modality == "tabular":
                stopped = "round limit"
                note(f"stopped after {MAX_ROUNDS} rounds that all still improved")

    best = best_run(project)
    result: dict[str, Any] = {"best": best, "log": log, "stopped": stopped}
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
        grace = 0.05 * budget_s  # finalize may run slightly past the deadline, never far
        pick = next(
            (r for r in cands if _finalize_cost(r, project) <= max(remaining, 0) + grace), None
        )
        existing = next(
            (
                r
                for r in reversed(list_runs(project.runs_dir))
                if pick is not None
                and r.meta.get("kind") == "final"
                and r.meta.get("parent") == pick["id"]
                and r.meta.get("status") == "completed"
            ),
            None,
        )
        if existing is not None:
            result["final"] = existing.id
            note(f"{pick['id']} is already finalized as {existing.id}")  # type: ignore[index]
        elif pick is None:
            note(
                "not enough budget left to refit on train+val; `trainwright predict` will use the best run as is"
            )
        else:
            if pick["id"] != best["id"]:
                note(
                    f"finalizing {pick['id']} (statistically tied with the leader, cheaper to refit)"
                )
            est = _finalize_cost(pick, project)
            res, err = _guarded(
                project,
                ["finalize", pick["id"], "--no-include-test"],
                est * 1.5 + 0.1 * budget_s + 120,
            )
            if res is None:
                note(f"finalize failed ({err}); `trainwright predict` will use {pick['id']} as is")
            else:
                result["final"] = res.get("run")
                note(
                    f"finalized {pick['id']} on train+val -> {res.get('run')} (served by `trainwright predict`)"
                )
    result["seconds"] = round(time.time() - t0, 1)
    write_json(project.state_dir / "autopilot.json", result)
    return result
