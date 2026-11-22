"""A/B harness: does an AI agent build better models with trainwright than without?

For each task, 15% of the data is hidden. Both conditions get the same vague, realistic prompt
and must write predictions for an unlabelled holdout; we score those against the hidden labels
and record time, turns and cost from the agent transcript.

* ``control``     : plain Python ML environment (numpy/pandas/sklearn/torch/torchvision) on PATH
* ``trainwright`` : trainwright installed + ``trainwright agent-setup`` (AGENTS.md, skill)

Usage::

    uv run python bench/agent_trials.py prepare --datasets scratch/datasets --out scratch/tasks
    uv run python bench/agent_trials.py run --task banking77 --condition trainwright
    uv run python bench/agent_trials.py score --trials scratch/trials
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

UNTIMED = (
    " Work fully autonomously: there is nobody to answer questions, so make reasonable decisions "
    "yourself. Keep individual commands under ~8 minutes (run longer jobs in the background)."
)
AUTONOMY = (
    " Work fully autonomously: there is nobody to answer questions, so make reasonable decisions "
    "yourself. You have a time limit of about {minutes} minutes, so keep individual commands under ~8 minutes."
)

TASKS = {
    "eye_movements": {
        "kind": "tabular",
        "source": "eye_movements/eye_movements.csv",
        "file": "eye_movements.csv",
        "target": "label",
        "task": "classification",
        "prompt": "I have this data in eye_movements.csv which has labels (the `label` column). Do feature "
        "exploration and train a model to predict it, and tell me how good it is. Finally write "
        "predictions for holdout.csv (same columns, no label) to predictions.csv with one row per "
        "holdout row, in the same order, in a column named `prediction`.",
    },
    "superconduct": {
        "kind": "tabular",
        "source": "superconduct/superconduct.csv",
        "file": "superconduct.csv",
        "target": "criticaltemp",
        "task": "regression",
        "prompt": "superconduct.csv has properties of superconductors and their critical temperature "
        "(`criticaltemp`). Explore it, train a model that predicts the critical temperature and "
        "tell me how accurate it is. Finally write predictions for holdout.csv (same columns, no "
        "target) to predictions.csv: one row per holdout row, same order, column `prediction`.",
    },
    "banking77": {
        "kind": "text",
        "source": "banking77/banking77.csv",
        "file": "banking77.csv",
        "target": "intent",
        "task": "classification",
        "prompt": "We route customer support messages to one of 77 intents with an LLM call today. "
        "banking77.csv has example messages and their intent. Train a small model that could "
        "replace the LLM call, and tell me how good it is and whether it's worth switching. "
        "Finally write predictions for holdout.csv (column `message`) to predictions.csv: one row "
        "per holdout row, same order, column `prediction`.",
    },
    "car_parts": {
        "kind": "image",
        "source": "car_parts/images",
        "file": "images",
        "target": None,
        "task": "classification",
        "prompt": "We currently identify car parts in engine-bay photos with a Sonnet agent, which is getting "
        "expensive as we scale. The labelled photos are in images/ (one folder per part). Can we "
        "train a smaller model that runs cheaply in the cloud instead? Train it, tell me how well "
        "it does and roughly what it would cost to run. Finally classify every image in holdout/ "
        "and write predictions.csv with columns `filename,prediction`.",
    },
    # ---- round 8+: wider coverage (bench/fetch_tasks.py)
    "beans": {
        "kind": "image",
        "source": "beans/images",
        "file": "images",
        "target": None,
        "task": "classification",
        "prompt": "Our field app sends photos of bean plant leaves to a vision LLM to flag disease, and "
        "farmers in low-connectivity areas need something that runs on a cheap phone or edge box. "
        "Labelled photos are in images/ (one folder per class). Train a small model for this, tell me "
        "how well it works and whether it is good enough to ship. Finally classify every image in "
        "holdout/ and write predictions.csv with columns `filename,prediction`.",
    },
    "eurosat": {
        "kind": "image",
        "source": "eurosat/images",
        "file": "images",
        "target": None,
        "task": "classification",
        "prompt": "images/ has small satellite image tiles sorted into land-use classes (one folder per "
        "class). We want to tag millions of tiles, which is far too expensive with an LLM. Train a "
        "model for it and tell me how accurate it is and what throughput/cost to expect. Finally "
        "classify every image in holdout/ and write predictions.csv with columns `filename,prediction`.",
    },
    "tweet_irony": {
        "kind": "text",
        "source": "tweet_irony/tweets.csv",
        "file": "tweets.csv",
        "target": "irony",
        "task": "classification",
        "prompt": "tweets.csv has tweets labelled as ironic or not (`irony`). Our moderation pipeline asks "
        "an LLM about every tweet. Can a small model do this instead? Train one, tell me how good it is. "
        "Finally write predictions for holdout.csv (column `tweet`) to predictions.csv: one row per "
        "holdout row, same order, column `prediction`.",
    },
    "app_reviews": {
        "kind": "text",
        "source": "app_reviews/reviews.csv",
        "file": "reviews.csv",
        "target": "star",
        "task": "regression",
        "prompt": "reviews.csv has Android app reviews with their star rating (`star`, 1-5). We want to "
        "estimate a star rating for feedback that arrives without one (support emails, tweets). Train "
        "a model that predicts the rating as a number and tell me how accurate it is. Finally write "
        "predictions for holdout.csv (same columns, no `star`) to predictions.csv: one row per holdout "
        "row, same order, numeric column `prediction`.",
    },
    "mercedes": {
        "kind": "tabular",
        "source": "mercedes/mercedes.csv",
        "file": "mercedes.csv",
        "target": "y",
        "task": "regression",
        "prompt": "mercedes.csv describes car configurations (anonymised columns) and `y`, the time in "
        "seconds the car spent on the test bench. Build a model that predicts `y` and tell me how good "
        "it is. Finally write predictions for holdout.csv (same columns, no `y`) to predictions.csv: one "
        "row per holdout row, same order, column `prediction`.",
    },
    "bike_sharing": {
        "kind": "tabular",
        "source": "bike_sharing/bike_sharing.csv",
        "file": "bike_sharing.csv",
        "target": "count",
        "task": "regression",
        "prompt": "bike_sharing.csv has hourly bike-rental counts (`count`) with weather and calendar "
        "information. Explore it, build a model that predicts the hourly count and tell me how accurate "
        "it is. Finally write predictions for holdout.csv (same columns, no `count`) to predictions.csv: "
        "one row per holdout row, same order, column `prediction`.",
    },
    "miniboone": {
        "kind": "tabular",
        "source": "miniboone/miniboone.csv",
        "file": "miniboone.csv",
        "target": "signal",
        "task": "classification",
        "prompt": "miniboone.csv has particle-detector measurements; `signal` says whether an event is a "
        "neutrino signal (True) or background (False). Train the best classifier you can and tell me how "
        "good it is. Finally write predictions for holdout.csv (same columns, no `signal`) to "
        "predictions.csv: one row per holdout row, same order, column `prediction` (True/False).",
    },
    "wine": {
        "kind": "tabular",
        "source": "wine/wine.csv",
        "file": "wine.csv",
        "target": "variety",
        "task": "classification",
        "prompt": "wine.csv has wine reviews: country, province, price, points and the taster's description. "
        "We currently ask an LLM to guess the grape `variety` for listings that don't state it. Train a "
        "model that predicts the variety and tell me how good it is and whether it could replace the "
        "LLM. Finally write predictions for holdout.csv (same columns, no `variety`) to predictions.csv: "
        "one row per holdout row, same order, column `prediction`.",
    },
    "telco_churn": {
        "kind": "tabular",
        "source": "telco_churn/telco_churn.csv",
        "file": "telco_churn.csv",
        "target": "Churn",
        "task": "classification",
        "prompt": "telco_churn.csv is our customer table; `Churn` says whether the customer left. Do some "
        "feature exploration and train a model to predict churn, and tell me how good it is. Finally "
        "write predictions for holdout.csv (same columns, no `Churn`) to predictions.csv: one row per "
        "holdout row, same order, column `prediction` (Yes/No).",
    },
}


# --------------------------------------------------------------------------- prepare


def prepare(
    datasets: Path,
    out: Path,
    hidden_frac: float = 0.15,
    seed: int = 0,
    only: list[str] | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in TASKS.items():
        if only and name not in only:
            continue
        tdir = out / name
        if tdir.exists():
            shutil.rmtree(tdir)
        (tdir / "data").mkdir(parents=True)
        src = datasets / spec["source"]
        if spec["kind"] in ("tabular", "text"):
            df = pd.read_csv(src).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            if spec["task"] == "classification":
                hidden_idx = (
                    df.groupby(spec["target"], group_keys=False)
                    .apply(lambda g: g.sample(frac=hidden_frac, random_state=seed))
                    .index
                )
            else:
                hidden_idx = df.sample(frac=hidden_frac, random_state=seed).index
            hidden = df.loc[hidden_idx].reset_index(drop=True)
            visible = df.drop(index=hidden_idx).reset_index(drop=True)
            visible.to_csv(tdir / "data" / spec["file"], index=False)
            hidden.drop(columns=[spec["target"]]).to_csv(tdir / "data" / "holdout.csv", index=False)
            hidden[[spec["target"]]].rename(columns={spec["target"]: "label"}).to_csv(
                tdir / "hidden_labels.csv", index=False
            )
        else:
            # hide whole near-duplicate groups (burst photos), otherwise the holdout is leaky
            from trainwright.data.dedup import duplicate_groups
            from trainwright.data.records import Records

            all_files = sorted(
                f for f in src.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            recs = Records(
                "image",
                "classification",
                [str(f) for f in all_files],
                np.array([f.parent.name for f in all_files]),
                "bench",
                paths=all_files,
            )
            groups, _ = duplicate_groups(recs)
            gid = dict(zip(map(str, all_files), groups))
            rows = []
            for cdir in sorted(d for d in src.iterdir() if d.is_dir()):
                files = sorted(
                    f for f in cdir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                )
                cgroups = sorted({gid[str(f)] for f in files})
                order = rng.permutation(len(cgroups))
                hidden_groups: set[str] = set()
                n_hidden = 0
                for k in order:
                    if len(cgroups) < 4 or n_hidden >= len(files) * hidden_frac:
                        break
                    hidden_groups.add(cgroups[k])
                    n_hidden += sum(1 for f in files if gid[str(f)] == cgroups[k])
                pick = {i for i, f in enumerate(files) if gid[str(f)] in hidden_groups}
                for i, f in enumerate(files):
                    if i in pick:
                        fname = f"img_{len(rows):04d}{f.suffix.lower()}"
                        dest = tdir / "data" / "holdout" / fname
                        rows.append({"filename": fname, "label": cdir.name})
                    else:
                        dest = tdir / "data" / "images" / cdir.name / f.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.link(f, dest)
            order = rng.permutation(len(rows))
            pd.DataFrame([rows[i] for i in order]).to_csv(tdir / "hidden_labels.csv", index=False)
        (tdir / "task.json").write_text(json.dumps({"name": name, **spec}, indent=2))
        print(f"prepared {name}")


# --------------------------------------------------------------------------- run


def run_trial(
    tasks: Path,
    trials: Path,
    task: str,
    condition: str,
    model: str = "sonnet",
    timeout_min: float = 45,
    tag: str = "",
    untimed: bool = False,
) -> Path:
    """``untimed``: the prompt states no time limit; ``timeout_min`` is only a hidden safety cap."""
    spec = json.loads((tasks / task / "task.json").read_text())
    tdir = trials / f"{task}-{condition}{('-' + tag) if tag else ''}"
    if tdir.exists():
        shutil.rmtree(tdir)
    shutil.copytree(tasks / task / "data", tdir, copy_function=os.link)
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    if condition == "trainwright":
        bin_dir = Path(os.environ.get("TW_TRIAL_BIN", str(ROOT / "scratch" / "tw-env" / "bin")))
        subprocess.run(
            [str(bin_dir / "trainwright"), "agent-setup", "--project"],
            cwd=tdir,
            check=True,
            capture_output=True,
        )
    else:
        bin_dir = ROOT / "scratch" / "ctrl-env" / "bin"
    # never let the dev checkout's venv (e.g. from `uv run`) leak into the trial
    clean = [
        p
        for p in env["PATH"].split(":")
        if ".venv" not in p and "tw-env" not in p and "ctrl-env" not in p
    ]
    env["PATH"] = ":".join([str(bin_dir), *clean])
    which = subprocess.run(
        ["which", "trainwright", "python"], env=env, capture_output=True, text=True
    ).stdout
    prompt = spec["prompt"] + (
        UNTIMED if untimed else AUTONOMY.format(minutes=int(max(10, timeout_min - 5)))
    )
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--setting-sources",
        "project,local",
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
    ]
    t0 = time.time()
    log = (tdir.parent / f"{tdir.name}.transcript.jsonl").open("w")
    try:
        p = subprocess.run(
            cmd, cwd=tdir, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_min * 60
        )
        status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
    log.close()
    meta = {
        "task": task,
        "condition": condition,
        "model": model,
        "status": status,
        "wall_s": round(time.time() - t0, 1),
        "tag": tag,
        "which": which.split(),
    }
    (tdir.parent / f"{tdir.name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(score_trial(tasks, tdir), indent=2))
    return tdir


# --------------------------------------------------------------------------- score


def _transcript_stats(path: Path) -> dict:
    stats = {
        "turns": None,
        "cost_usd": None,
        "duration_s": None,
        "bash_commands": 0,
        "used_trainwright": False,
        "used_sklearn_models": set(),
        "torch_used": False,
    }
    if not path.exists():
        return stats
    for line in path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            stats["turns"] = ev.get("num_turns")
            stats["cost_usd"] = ev.get("total_cost_usd")
            stats["duration_s"] = round((ev.get("duration_ms") or 0) / 1000, 1)
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") != "tool_use":
                    continue
                inp = json.dumps(block.get("input", {}))
                if block.get("name") == "Bash":
                    stats["bash_commands"] += 1
                if "trainwright " in inp:
                    stats["used_trainwright"] = True
                for m in (
                    "RandomForest",
                    "GradientBoosting",
                    "HistGradientBoosting",
                    "LogisticRegression",
                    "XGB",
                    "LGBM",
                    "SVC",
                    "KNeighbors",
                    "Ridge",
                    "TfidfVectorizer",
                    "MLPClassifier",
                    "MLPRegressor",
                    "ExtraTrees",
                ):
                    if m in inp:
                        stats["used_sklearn_models"].add(m)
                if "import torch" in inp or "torch.nn" in inp:
                    stats["torch_used"] = True
    stats["used_sklearn_models"] = sorted(stats["used_sklearn_models"])
    return stats


def score_trial(tasks: Path, tdir: Path) -> dict:
    meta_path = tdir.parent / f"{tdir.name}.meta.json"
    meta = (
        json.loads(meta_path.read_text())
        if meta_path.exists()
        else {"task": tdir.name.split("-")[0]}
    )
    spec = json.loads((tasks / meta["task"] / "task.json").read_text())
    truth = pd.read_csv(tasks / meta["task"] / "hidden_labels.csv")
    res = {**meta, **_transcript_stats(tdir.parent / f"{tdir.name}.transcript.jsonl")}
    pred_path = tdir / "predictions.csv"
    if not pred_path.exists():
        res["score"] = None
        res["score_error"] = "no predictions.csv"
        return res
    try:
        pred = pd.read_csv(pred_path)
        if spec["kind"] == "image":
            col = "filename" if "filename" in pred.columns else pred.columns[0]
            pred["key"] = pred[col].astype(str).map(lambda s: Path(s).name)
            merged = truth.merge(pred, left_on="filename", right_on="key", how="left")
            y, p = merged["label"].astype(str), merged["prediction"].astype(str)
            res["coverage"] = float(merged["prediction"].notna().mean())
        else:
            if len(pred) != len(truth):
                raise ValueError(f"{len(pred)} predictions for {len(truth)} holdout rows")
            y, p = truth["label"], pred["prediction"]
        if spec["task"] == "classification":
            from sklearn.metrics import accuracy_score, f1_score

            y, p = y.astype(str), p.astype(str).str.replace(r"\.0$", "", regex=True)
            res["score"] = {
                "accuracy": round(float(accuracy_score(y, p)), 4),
                "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
            }
        else:
            y, p = y.astype(float).to_numpy(), p.astype(float).to_numpy()
            rmse = float(np.sqrt(((y - p) ** 2).mean()))
            r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            res["score"] = {"rmse": round(rmse, 4), "r2": round(float(r2), 4)}
    except Exception as e:
        res["score"] = None
        res["score_error"] = f"{type(e).__name__}: {e}"
    return res


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "-"
    rng = f" ({min(xs):.3f} to {max(xs):.3f})" if len(xs) > 1 else ""
    return f"{np.mean(xs):.3f}{rng}"


def summary_table(tasks: Path, trial_dirs: list[Path], tw_tags: list[str]) -> str:
    rows = []
    for d in trial_dirs:
        if d.exists():
            rows += [
                score_trial(tasks, t)
                for t in sorted(d.iterdir())
                if t.is_dir() and (t.parent / f"{t.name}.meta.json").exists()
            ]
    out = [
        "| task | metric | straight prompt (control) | trainwright | runs (control / tw) |",
        "|---|---|---|---|---|",
    ]
    for task, spec in TASKS.items():
        key = "rmse" if spec["task"] == "regression" else "accuracy"
        ctrl = [
            r["score"][key]
            for r in rows
            if r.get("task") == task and r.get("condition") == "control" and r.get("score")
        ]
        tw = [
            r["score"][key]
            for r in rows
            if r.get("task") == task
            and r.get("condition") == "trainwright"
            and r.get("tag") in tw_tags
            and r.get("score")
        ]
        better = None
        if ctrl and tw:
            better = bool(
                (np.mean(tw) < np.mean(ctrl)) if key == "rmse" else (np.mean(tw) > np.mean(ctrl))
            )
        tw_cell = _fmt(tw)
        if better:
            tw_cell = f"**{tw_cell}**"
        ctrl_cell = f"**{_fmt(ctrl)}**" if better is False else _fmt(ctrl)
        out.append(
            f"| {task} | {key}{' ↓' if key == 'rmse' else ''} | {ctrl_cell} | {tw_cell} | {len(ctrl)} / {len(tw)} |"
        )
    return "\n".join(out)


GENERIC_PROMPT = {
    "classification": "I have this data in {file} with labels in the `{target}` column. Do feature exploration, "
    "train a model to predict it and tell me how good it is. Finally write predictions for holdout.csv (same "
    "columns, no label) to predictions.csv with one row per holdout row, in the same order, in a column named "
    "`prediction`.",
    "regression": "I have this data in {file}; the target is the `{target}` column. Explore it, train a model that "
    "predicts the target and tell me how accurate it is. Finally write predictions for holdout.csv (same columns, no "
    "target) to predictions.csv: one row per holdout row, same order, column `prediction`.",
}


def prepare_suite(suite_data: Path, out: Path, hidden_frac: float = 0.15, seed: int = 0) -> int:
    """Turn bench/suite.py datasets into agent tasks with a hidden 15% holdout and a generic prompt."""
    n = 0
    for d in sorted(suite_data.iterdir()):
        if not (d / "data.csv").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        tdir = out / d.name
        if (tdir / "task.json").exists():
            n += 1
            continue
        (tdir / "data").mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(d / "data.csv").sample(frac=1.0, random_state=seed).reset_index(drop=True)
        target = meta["target"]
        if meta["task"] == "classification":
            hidden_idx = (
                df.groupby(target, group_keys=False)
                .apply(lambda g: g.sample(frac=hidden_frac, random_state=seed))
                .index
            )
        else:
            hidden_idx = df.sample(frac=hidden_frac, random_state=seed).index
        hidden, visible = (
            df.loc[hidden_idx].reset_index(drop=True),
            df.drop(index=hidden_idx).reset_index(drop=True),
        )
        visible.to_csv(tdir / "data" / "data.csv", index=False)
        hidden.drop(columns=[target]).to_csv(tdir / "data" / "holdout.csv", index=False)
        hidden[[target]].rename(columns={target: "label"}).to_csv(
            tdir / "hidden_labels.csv", index=False
        )
        spec = {
            "name": d.name,
            "kind": "tabular",
            "task": meta["task"],
            "target": target,
            "file": "data.csv",
            "prompt": GENERIC_PROMPT[meta["task"]].format(file="data.csv", target=target),
        }
        (tdir / "task.json").write_text(json.dumps(spec, indent=2))
        n += 1
    return n


def batch(
    tasks: Path,
    trials: Path,
    conditions: list[str],
    reps: int,
    parallel: int,
    max_trials: int,
    model: str,
    timeout_min: float,
    only: list[str] | None = None,
    prefix: str = "b",
    untimed: bool = False,
) -> None:
    """Resumable: runs every (task, condition, rep) without a .meta.json, at most max_trials this call."""
    jobs = []
    for tdir in sorted(tasks.iterdir()):
        if not (tdir / "task.json").exists() or (only and tdir.name not in only):
            continue
        for rep in range(1, reps + 1):
            for cond in conditions:
                tag = f"{prefix}{rep}"
                if not (trials / f"{tdir.name}-{cond}-{tag}.meta.json").exists():
                    jobs.append((tdir.name, cond, tag))
    jobs = jobs[:max_trials]
    print(f"{len(jobs)} trials to run ({parallel} at a time)", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(parallel) as ex:
        list(
            ex.map(
                lambda j: run_trial(
                    tasks, trials, j[0], j[1], model, timeout_min, j[2], untimed=untimed
                ),
                jobs,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("prepare-suite", help="agent tasks from bench/suite.py datasets")
    ps.add_argument("--suite-data", type=Path, default=ROOT / "scratch" / "suite" / "data")
    ps.add_argument("--out", type=Path, default=ROOT / "scratch" / "suite-tasks")
    b = sub.add_parser("batch", help="resumable, capped batch of agent trials")
    b.add_argument("--tasks", type=Path, default=ROOT / "scratch" / "suite-tasks")
    b.add_argument(
        "--trials", type=Path, default=Path.home() / ".cache" / "trainwright-suite-trials"
    )
    b.add_argument("--conditions", nargs="+", default=["control", "trainwright"])
    b.add_argument("--reps", type=int, default=1)
    b.add_argument("--parallel", type=int, default=4)
    b.add_argument(
        "--max-trials", type=int, default=20, help="hard cap per invocation (usage control)"
    )
    b.add_argument("--model", default="sonnet")
    b.add_argument("--timeout-min", type=float, default=30)
    b.add_argument("--only", nargs="*", default=None, help="task names to include")
    b.add_argument("--prefix", default="b", help="trial tag prefix, e.g. r8-")
    b.add_argument(
        "--untimed",
        action="store_true",
        help="no time limit in the prompt; --timeout-min becomes a hidden safety cap",
    )
    p = sub.add_parser("prepare")
    p.add_argument("--datasets", type=Path, default=ROOT / "scratch" / "datasets")
    p.add_argument("--out", type=Path, default=ROOT / "scratch" / "tasks")
    p.add_argument("--only", nargs="*", default=None)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True, choices=list(TASKS))
    r.add_argument("--condition", required=True, choices=["control", "trainwright"])
    r.add_argument("--model", default="sonnet")
    r.add_argument("--timeout-min", type=float, default=45)
    r.add_argument("--tag", default="")
    r.add_argument("--tasks", type=Path, default=ROOT / "scratch" / "tasks")
    # trials run OUTSIDE the repo so agents can't wander into hidden labels or other trials
    r.add_argument("--trials", type=Path, default=Path.home() / ".cache" / "trainwright-trials")
    t = sub.add_parser("summary", help="Markdown table averaging replicates across trial dirs")
    t.add_argument("--tasks", type=Path, default=ROOT / "scratch" / "tasks")
    t.add_argument(
        "--trials",
        type=Path,
        nargs="+",
        default=[ROOT / "scratch" / "trials", Path.home() / ".cache" / "trainwright-trials"],
    )
    t.add_argument("--trainwright-tags", nargs="+", default=["v5a", "v5b"])
    s = sub.add_parser("score")
    s.add_argument("--tasks", type=Path, default=ROOT / "scratch" / "tasks")
    s.add_argument("--trials", type=Path, default=Path.home() / ".cache" / "trainwright-trials")
    a = ap.parse_args()
    if a.cmd == "prepare-suite":
        print(f"{prepare_suite(a.suite_data, a.out)} agent tasks ready in {a.out}")
    elif a.cmd == "batch":
        a.trials.mkdir(parents=True, exist_ok=True)
        batch(
            a.tasks,
            a.trials,
            a.conditions,
            a.reps,
            a.parallel,
            a.max_trials,
            a.model,
            a.timeout_min,
            a.only,
            a.prefix,
            a.untimed,
        )
    elif a.cmd == "prepare":
        prepare(a.datasets, a.out, only=a.only)
    elif a.cmd == "run":
        a.trials.mkdir(parents=True, exist_ok=True)
        run_trial(a.tasks, a.trials, a.task, a.condition, a.model, a.timeout_min, a.tag)
    elif a.cmd == "summary":
        print(summary_table(a.tasks, a.trials, a.trainwright_tags))
    else:
        rows = [score_trial(a.tasks, d) for d in sorted(a.trials.iterdir()) if d.is_dir()]
        print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
