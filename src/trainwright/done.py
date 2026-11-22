"""Definition of done: what a finished project has, checked from the project's own records.

``check`` reads the profile, split, feature-trial log, auto state, leaderboard, test ledger and
decision log, and lists what is still missing. Judgement calls the tool can't verify
(the deployment scenario, a suspected leak that is legitimate, stopping feature work early)
are recorded with ``trainwright decide <topic> "<reason>"`` and count as resolved, so a
reviewer can read every one of them.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ._utils import read_json, read_jsonl
from .data.split import load_split
from .leaderboard import best_run, leaderboard
from .project import Project
from .tracking import list_runs

MIN_FEATURE_TRIALS = 3


def decisions(project: Project) -> dict[str, dict[str, Any]]:
    """Latest decision per topic."""
    out: dict[str, dict[str, Any]] = {}
    for d in read_jsonl(project.state_dir / "decisions.jsonl"):
        out[d["topic"]] = d
    return out


def record_decision(project: Project, topic: str, reason: str) -> dict[str, Any]:
    project.ensure_dirs()
    d = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "topic": topic.strip(), "reason": reason}
    with open(project.state_dir / "decisions.jsonl", "a") as f:
        f.write(json.dumps(d) + "\n")
    return d


def _item(name: str, status: str, detail: str, fix: str | None = None) -> dict[str, Any]:
    return {"item": name, "status": status, "detail": detail, "fix": fix}


def check(project: Project) -> dict[str, Any]:
    cfg = project.config
    dec = decisions(project)
    items: list[dict[str, Any]] = []
    tabular = cfg.modality == "tabular"

    # 1. the data was profiled -----------------------------------------------------------
    prof = read_json(project.profile_path) if project.profile_path.exists() else None
    items.append(
        _item("profile", "pass", "data profiled")
        if prof
        else _item("profile", "fail", "data not profiled", "trainwright profile")
    )
    tab = (prof or {}).get("tabular") or {}

    # 2. every suspected leak is dropped or explained -----------------------------------------
    open_leaks = []
    for leak in tab.get("leakage_suspects", []):
        cols = [c.strip() for c in str(leak["column"]).split(",")]
        if all(c in cfg.data.drop for c in cols) or f"leak:{leak['column']}" in dec:
            continue
        open_leaks.append(leak["column"])
    if open_leaks:
        items.append(
            _item(
                "leaks",
                "fail",
                f"suspected leaks not resolved: {open_leaks}",
                f'add to data.drop, or `trainwright decide leak:{open_leaks[0]} "why it is '
                'legitimate at prediction time"`',
            )
        )
    elif prof:
        items.append(_item("leaks", "pass", "no unresolved leak suspects"))

    # 3. the split mimics deployment ----------------------------------------------------------
    groups = [g["column"] for g in tab.get("group_suspects", [])]
    dates = list(cfg.data.date_columns)
    if cfg.data.group_column or cfg.data.time_column:
        how = (
            f"group split on {cfg.data.group_column}"
            if cfg.data.group_column
            else f"time split on {cfg.data.time_column}"
        )
        items.append(_item("deployment", "pass", how))
    elif "deployment" in dec:
        items.append(_item("deployment", "pass", f"decided: {dec['deployment']['reason']}"))
    elif groups or dates:
        flagged = ", ".join(
            ([f"group keys {groups}"] if groups else []) + ([f"dates {dates}"] if dates else [])
        )
        items.append(
            _item(
                "deployment",
                "fail",
                f"{flagged} flagged; the split is random. Will predictions be for new groups or "
                "future dates?",
                "set data.group_column / data.time_column and `trainwright split --force`, or "
                '`trainwright decide deployment "predictions are for ... so a random split is right"`',
            )
        )
    else:
        items.append(_item("deployment", "pass", "random split; no group keys or dates flagged"))

    # 4. the errors were read, and ideas were tested ----------------------------------------------
    errors = project.state_dir / "errors.json"
    trials = read_jsonl(project.state_dir / "feature_trials.jsonl")
    if not errors.exists():
        items.append(
            _item(
                "error analysis",
                "fail",
                "`trainwright errors` never run",
                "trainwright errors" if tabular else "trainwright errors best",
            )
        )
    elif tabular and len(trials) < MIN_FEATURE_TRIALS and "features" not in dec:
        items.append(
            _item(
                "error analysis",
                "fail",
                f"{len(trials)} feature idea(s) tested; aim for {MIN_FEATURE_TRIALS} "
                "(from the slices and worst rows `errors` shows)",
                'trainwright try cand.py --hypothesis "..."  (or `trainwright decide features '
                '"why no more ideas are worth testing"`)',
            )
        )
    else:
        kept = sum(1 for t in trials if t.get("kept"))
        detail = (
            f"errors read; {len(trials)} feature idea(s) tested, {kept} kept"
            if tabular
            else "errors read"
        )
        if "features" in dec and len(trials) < MIN_FEATURE_TRIALS:
            detail += f"; decided: {dec['features']['reason']}"
        items.append(_item("error analysis", "pass", detail))

    # 5. the search converged ---------------------------------------------------------------------
    auto = (
        read_json(project.state_dir / "auto.json")
        if (project.state_dir / "auto.json").exists()
        else {}
    )
    split = load_split(project)
    sh = split["split_hash"] if split else None
    runs = [r for r in list_runs(project.runs_dir) if r.meta.get("split_hash") == sh]
    neural = [
        r for r in runs if r.meta.get("kind") == "neural" and r.meta.get("status") == "completed"
    ]
    if auto.get("status") == "running":
        items.append(_item("search", "fail", "auto is still running", "trainwright wait"))
    elif auto.get("status") == "done" and auto.get("stopped") == "converged":
        items.append(_item("search", "pass", "auto converged (further rounds stopped improving)"))
    elif "search" in dec:
        items.append(_item("search", "pass", f"decided: {dec['search']['reason']}"))
    elif auto.get("status") == "done":
        items.append(
            _item(
                "search",
                "fail",
                f"auto stopped on {auto.get('stopped') or 'its budget'}, not convergence",
                "trainwright auto  (no --budget: it continues while results improve), or "
                '`trainwright decide search "why this is enough"`',
            )
        )
    else:
        items.append(
            _item(
                "search",
                "fail",
                f"{len(neural)} neural run(s); no converged search",
                "trainwright auto --background, then trainwright wait",
            )
        )

    # 6. a real model leads, and it is finalized ---------------------------------------------------
    best = best_run(project)
    if best is None:
        items.append(_item("model", "fail", "no completed runs", "trainwright auto"))
    else:
        m = cfg.metric
        if best.get("model") == "baseline:dummy":
            items.append(
                _item("model", "fail", "the leader is the dummy baseline: nothing learned", None)
            )
        else:
            items.append(
                _item(
                    "model",
                    "pass",
                    f"leader {best['id']} ({best.get('model')}) val_{m}={best.get(f'val_{m}'):.4f}",
                )
            )
        tied = {best["id"]} | {r["id"] for r in leaderboard(project) if r.get("tied_with_leader")}
        finals = [
            r
            for r in runs
            if r.meta.get("kind") == "final"
            and r.meta.get("status") == "completed"
            and r.meta.get("parent") in tied
        ]
        items.append(
            _item("finalized", "pass", f"{finals[-1].id} (refit on train+val)")
            if finals
            else _item(
                "finalized",
                "fail",
                "the leader is not refit on train+val",
                f"trainwright finalize {best['id']}",
            )
        )

        # 7. the test set was used once, for the model being shipped ------------------------------
        ledger = read_jsonl(project.ledger_path)
        tested = sorted({e["run"] for e in ledger})
        if not tested:
            items.append(
                _item(
                    "test",
                    "fail",
                    "no final test evaluation",
                    f'trainwright evaluate {best["id"]} --split test --reason "final model"',
                )
            )
        elif len(tested) > 1:
            items.append(
                _item(
                    "test",
                    "warn",
                    f"test evaluated for {len(tested)} runs {tested[:4]}: choosing between them "
                    "by test score is selection on test; quote it as optimistic",
                )
            )
        else:
            items.append(_item("test", "pass", f"evaluated once ({tested[0]})"))

    # 8. predictions (only when the task asked for them) -------------------------------------------
    pf = project.state_dir / "predictions.json"
    if pf.exists():
        p = read_json(pf)
        items.append(
            _item("predictions", "pass", f"{p['n']} rows in {p['path']} from {p['model']}")
        )

    done = not any(i["status"] == "fail" for i in items)
    return {"done": done, "items": items, "decisions": list(dec.values())}


def format_check(res: dict[str, Any]) -> str:
    mark = {"pass": "✓", "fail": "✗", "warn": "!"}
    lines = [f"{mark[i['status']]} {i['item']:<15} {i['detail']}" for i in res["items"]]
    for i in res["items"]:
        if i["status"] == "fail" and i["fix"]:
            lines.append(f"  to do ({i['item']}): {i['fix']}")
    if res["decisions"]:
        lines.append("decisions on record:")
        lines += [f"  {d['topic']}: {d['reason']}" for d in res["decisions"]]
    lines.append(
        "DONE: have a reviewer check the report against these records before you finish"
        if res["done"]
        else "NOT DONE"
    )
    return "\n".join(lines)
