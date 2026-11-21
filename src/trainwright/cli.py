"""``trainwright`` command line. Designed to be driven by AI agents (and humans).

Conventions:
* concise human-readable output by default, ``--json`` for a machine-readable envelope
  ``{"ok", "command", "result", "next_steps", "warnings"}``;
* every command ends with ``next:`` suggestions;
* errors say what went wrong *and* how to fix it, with exit code 1.
"""

from __future__ import annotations

import json
import sys
import traceback
from importlib import resources
from pathlib import Path
from typing import Any

import typer

from . import __version__
from ._utils import to_json, write_json

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
    help="trainwright: an agent-first PyTorch framework for training small task-specific models.",
)

STATE: dict[str, Any] = {"json": False, "command": None}


class CLIError(Exception):
    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


def emit(
    result: Any,
    text: str | None = None,
    next_steps: list[dict[str, str]] | None = None,
    warnings: list[str] | None = None,
) -> None:
    next_steps = next_steps or []
    warnings = warnings or []
    if STATE["json"]:
        print(
            to_json(
                {
                    "ok": True,
                    "command": STATE["command"],
                    "result": result,
                    "next_steps": next_steps,
                    "warnings": warnings,
                }
            )
        )
        return
    if text:
        print(text)
    for w in warnings:
        print(f"WARNING: {w}")
    if next_steps:
        print("\nnext:")
        for n in next_steps:
            print(f"  $ {n['command']}")
            if n.get("why"):
                print(f"      # {n['why']}")


def progress(msg: str) -> None:
    # progress goes to stderr in JSON mode so stdout stays parseable
    print(msg, file=sys.stderr if STATE["json"] else sys.stdout, flush=True)


def _rel(p: Any) -> str:
    """Paths relative to the cwd keep output short (agents pay per token)."""
    if p is None:
        return ""
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def _project() -> Any:
    from .project import Project

    return Project.find()


def _playbook() -> str:
    return resources.files("trainwright").joinpath("templates/PLAYBOOK.md").read_text()


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
) -> None:
    STATE["json"] = json_out
    if version:
        print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


# --------------------------------------------------------------------------- init


def _infer_project(
    data: Path, target: str | None, text_column: str | None, image_column: str | None
) -> dict[str, Any]:
    from .data.profile import infer_task, recommend_metric
    from .data.records import read_table
    from .preprocessing import IMAGE_EXTENSIONS, infer_column_kind

    if data.is_dir():
        classes = [d for d in data.iterdir() if d.is_dir() and not d.name.startswith(".")]
        counts = {
            d.name: sum(1 for f in d.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
            for d in classes
        }
        if not classes or not sum(counts.values()):
            raise CLIError(
                f"{data} is a directory but has no <class>/<image> sub-folders.",
                "Arrange images as data/<class_name>/<file>.jpg or pass a CSV manifest with --image-column.",
            )
        metric, mode, why = recommend_metric("classification", counts)
        return {
            "modality": "image",
            "task": "classification",
            "target": None,
            "metric": metric,
            "mode": mode,
            "why": why,
            "text_column": None,
            "image_column": None,
        }
    df = read_table(data)
    if not target:
        for cand in ("label", "target", "y", "class", "outcome"):
            if cand in df.columns:
                target = cand
                break
        else:
            raise CLIError(
                "Could not guess the target column.",
                f"Pass --target. Columns: {list(df.columns)[:40]}",
            )
    if target not in df.columns:
        raise CLIError(f"Target {target!r} not in data.", f"Columns: {list(df.columns)[:40]}")
    task = infer_task(df[target])
    modality = "tabular"
    if image_column:
        modality = "image"
    elif text_column:
        modality = "text"
    else:
        feats = [c for c in df.columns if c != target]
        text_cols = [
            c
            for c in feats
            if infer_column_kind(df[c].dropna().astype(str).head(2000).tolist()) == "text"
            and not __import__("pandas").api.types.is_numeric_dtype(df[c])
        ]
        if len(text_cols) == 1 and len(feats) <= 3:
            modality, text_column = "text", text_cols[0]
    tcounts: Any = (
        df[target].astype(str).value_counts().to_dict() if task == "classification" else None
    )
    metric, mode, why = recommend_metric(task, tcounts)
    return {
        "modality": modality,
        "task": task,
        "target": target,
        "metric": metric,
        "mode": mode,
        "why": why,
        "text_column": text_column,
        "image_column": image_column,
    }


AGENT_FILES_NOTE = "Agent instructions written: AGENTS.md, .claude/skills/trainwright/SKILL.md"

SKILL_FRONTMATTER = """---
name: trainwright
description: Train, evaluate and deploy small task-specific ML models (tabular, image or text classification/regression) with the trainwright CLI and a disciplined lifecycle (profile, frozen split, baselines, hypothesis-driven experiments, curve diagnosis, sealed test set, ONNX export, cost benchmark vs LLM). Use whenever the user has labelled data and wants a model trained or improved, asks to "predict X from this data", or asks whether a cheaper small model could replace an LLM call.
---

"""


def _write_agent_files(root: Path, global_skill: bool = False) -> list[str]:
    written = []
    pb = _playbook()
    agents = root / "AGENTS.md"
    if not agents.exists():
        agents.write_text(pb)
        written.append(str(agents))
    claude_md = root / "CLAUDE.md"
    line = "@AGENTS.md"
    if not claude_md.exists():
        claude_md.write_text(
            f"# Project instructions\n\nThis is a trainwright ML project. Follow the playbook:\n\n{line}\n"
        )
        written.append(str(claude_md))
    elif line not in claude_md.read_text():
        with claude_md.open("a") as f:
            f.write(f"\n{line}\n")
        written.append(str(claude_md))
    targets = [root / ".claude" / "skills" / "trainwright"]
    if global_skill:
        targets.append(Path.home() / ".claude" / "skills" / "trainwright")
    for t in targets:
        t.mkdir(parents=True, exist_ok=True)
        (t / "SKILL.md").write_text(SKILL_FRONTMATTER + pb)
        written.append(str(t / "SKILL.md"))
    gi = root / ".gitignore"
    lines = ["runs/*/checkpoints/", ".trainwright/cache/", "artifacts/", "__pycache__/"]
    existing = gi.read_text().splitlines() if gi.exists() else []
    add = [x for x in lines if x not in existing]
    if add:
        with gi.open("a") as f:
            f.write(
                ("\n" if existing and existing[-1] else "")
                + "# trainwright\n"
                + "\n".join(add)
                + "\n"
            )
    return written


@app.command()
def init(
    data: Path = typer.Option(
        ..., "--data", help="CSV/Parquet file, image folder, or image manifest CSV."
    ),
    target: str | None = typer.Option(None, help="Target column (guessed if omitted)."),
    name: str | None = typer.Option(None, help="Project name (default: data file/folder name)."),
    task: str | None = typer.Option(None, help="classification | regression (inferred)."),
    modality: str | None = typer.Option(None, help="tabular | image | text (inferred)."),
    text_column: str | None = typer.Option(None, help="Text column (text modality)."),
    image_column: str | None = typer.Option(None, help="Image path column (image manifest CSV)."),
    metric: str | None = typer.Option(None, help="Primary metric (recommended automatically)."),
    drop: list[str] = typer.Option([], help="Columns never to use as features (repeatable)."),
    group_column: str | None = typer.Option(None, help="Keep rows of a group in one split."),
    time_column: str | None = typer.Option(None, help="Split chronologically on this column."),
    force: bool = typer.Option(False, help="Overwrite an existing trainwright.yaml."),
    global_skill: bool = typer.Option(
        False, "--global-skill", help="Also install the agent skill in ~/.claude/skills."
    ),
) -> None:
    """Create trainwright.yaml (+ agent instructions) in the current directory."""
    from .config import DataConfig, ProjectConfig
    from .metrics import metric_mode, valid_metrics
    from .project import CONFIG_NAME, Project

    STATE["command"] = "init"
    root = Path.cwd()
    if (root / CONFIG_NAME).exists() and not force:
        raise CLIError(
            f"{CONFIG_NAME} already exists here.",
            "Edit it directly, or pass --force to recreate it.",
        )
    data_abs = data.resolve()
    if not data_abs.exists():
        raise CLIError(f"Data path {data} does not exist.")
    inf = _infer_project(data_abs, target, text_column, image_column)
    task = task or inf["task"]
    modality = modality or inf["modality"]
    if metric:
        if metric not in valid_metrics(task):
            raise CLIError(
                f"Metric {metric!r} is not valid for {task}.",
                f"Choose one of {valid_metrics(task)}",
            )
        mode, why = metric_mode(metric), "chosen by user"
    else:
        metric, mode, why = inf["metric"], inf["mode"], inf["why"]
        if task != inf["task"]:
            from .data.profile import recommend_metric

            metric, mode, why = recommend_metric(
                task, None if task == "regression" else {"a": 1, "b": 1}
            )
    try:
        data_rel = str(data_abs.relative_to(root))
    except ValueError:
        data_rel = str(data_abs)
    cfg = ProjectConfig(
        name=name or data_abs.stem,
        task=task,
        modality=modality,
        data=DataConfig(
            path=data_rel,
            target=inf["target"] if modality != "image" or image_column else None,
            text_column=text_column or inf["text_column"],
            image_column=image_column,
            drop=list(drop),
            group_column=group_column,
            time_column=time_column,
        ),
        metric=metric,
        metric_mode=mode,
    )
    project = Project(root, cfg)
    project.save()
    project.ensure_dirs()
    written = _write_agent_files(root, global_skill)
    emit(
        {"config": cfg.model_dump(), "metric_rationale": why, "agent_files": written},
        f"created {CONFIG_NAME}: {cfg.task} on {cfg.modality} data, target={cfg.data.target}, "
        f"metric={cfg.metric} ({cfg.metric_mode})\n  why this metric: {why}\n{AGENT_FILES_NOTE}",
        [{"command": "trainwright profile", "why": "inspect the data before anything else"}],
    )


# --------------------------------------------------------------------------- status / guide


@app.command()
def status() -> None:
    """Lifecycle checklist for this project and the next commands to run."""
    from .lifecycle import format_status
    from .lifecycle import status as _status

    STATE["command"] = "status"
    s = _status(_project())
    emit(
        s,
        format_status(s) if not STATE["json"] else None,
        [] if not STATE["json"] else s["next_steps"],
    )


@app.command()
def guide() -> None:
    """Print the agent playbook (rules, the loop, what to try per modality)."""
    STATE["command"] = "guide"
    pb = _playbook()
    emit({"playbook": pb}, pb)


@app.command("agent-setup")
def agent_setup(
    global_skill: bool = typer.Option(
        True, "--global/--project", help="Install skill in ~/.claude/skills too."
    ),
) -> None:
    """(Re)write AGENTS.md, CLAUDE.md import and the Claude Code skill."""
    STATE["command"] = "agent-setup"
    written = _write_agent_files(Path.cwd(), global_skill)
    emit(
        {"written": written},
        "\n".join(f"wrote {w}" for w in written) or "agent files already up to date",
    )


# --------------------------------------------------------------------------- data


@app.command()
def profile(
    keep_ids: bool = typer.Option(
        False, "--keep-ids", help="Don't auto-drop id-like/constant columns."
    ),
) -> None:
    """Profile the data: types, missingness, leakage suspects, imbalance, recommendations."""
    from .data.profile import profile as _profile
    from .pipeline import get_records

    STATE["command"] = "profile"
    project = _project()
    records = get_records(project)
    prof = _profile(records, project.config, project.state_dir)
    prof["data_fingerprint"] = records.fingerprint
    cfg = project.config
    auto_dropped: list[str] = []
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
    rec_metric = prof["recommended_metric"]
    lines = [f"{prof['n']} examples ({cfg.modality}, {cfg.task})"]
    if prof.get("dropped_missing_target"):
        lines.append(f"dropped {prof['dropped_missing_target']} rows with a missing target")
    t = prof["target"]
    if cfg.task == "classification":
        top = list(t["counts"].items())[:8]
        lines.append(
            f"target: {t['n_classes']} classes, imbalance {t['imbalance_ratio']}:1, counts {dict(top)}"
            + (" ..." if t["n_classes"] > 8 else "")
        )
    else:
        lines.append(
            f"target: mean {t['mean']:.4g} std {t['std']:.4g} range [{t['min']:.4g}, {t['max']:.4g}] skew {t['skew']:.2f}"
        )
    if "tabular" in prof:
        tb = prof["tabular"]
        lines.append(
            f"features: {tb['n_features']} columns {tb['feature_kinds']}, {tb['duplicate_rows']} duplicate rows"
        )
        for c, p in tb["columns"].items():
            if p.get("flags"):
                lines.append(f"  - {c}: {'; '.join(p['flags'])}")
        for leak in tb["leakage_suspects"]:
            lines.append(f"  - LEAK? {leak['column']}: {leak['why']}")
    if "image" in prof:
        im = prof["image"]
        lines.append(
            f"images: median {im['width']['p50']:.0f}x{im['height']['p50']:.0f}px, modes {im['modes']}"
        )
    if "text" in prof:
        tx = prof["text"]
        lines.append(
            f"text: {tx['words_per_doc']['p50']:.0f} words median, p95 {tx['words_per_doc']['p95']:.0f}; "
            f"{tx['empty_docs']} empty, {tx['duplicate_docs']} duplicates"
        )
    if auto_dropped:
        lines.append(
            f"added to data.drop in trainwright.yaml (id-like/constant): {auto_dropped}  (--keep-ids to disable)"
        )
    if rec_metric["metric"] != cfg.metric:
        lines.append(
            f"NOTE: recommended metric is {rec_metric['metric']} ({rec_metric['why']}); config uses {cfg.metric}"
        )
    lines.append("recommendations:")
    lines += [f"  * {r}" for r in prof["recommendations"]]
    emit(
        prof if STATE["json"] else {},
        "\n".join(lines),
        [
            {
                "command": "trainwright split",
                "why": "freeze train/val/test (after editing data.drop if needed)",
            }
        ],
    )


@app.command()
def split(
    force: bool = typer.Option(False, help="Re-split (invalidates comparisons with earlier runs)."),
) -> None:
    """Freeze the train/val/test split (once per project)."""
    from .data.split import split_summary
    from .pipeline import ensure_split

    STATE["command"] = "split"
    project = _project()
    project.ensure_dirs()
    sp, created = ensure_split(project, force=force)
    summ = split_summary(sp)
    dd = sp.get("dedup") or {}
    dd_msg = (
        f"\n  duplicate-aware: {dd['items_with_duplicates']} examples in near-duplicate groups kept within one split"
        if dd.get("items_with_duplicates")
        else ""
    )
    msg = (
        f"{'froze new' if created else 'existing'} {sp['strategy']} split: {sp['counts']} (hash {sp['split_hash']})"
        + ("" if created else "  (use --force to re-split)")
        + dd_msg
    )
    emit(
        summ,
        msg,
        [{"command": "trainwright baseline", "why": "establish the floor to beat"}],
        sp.get("warnings"),
    )


# --------------------------------------------------------------------------- training


@app.command()
def baseline(
    only: str | None = typer.Option(None, help="Comma-separated subset, e.g. 'dummy,hist_gbm'."),
) -> None:
    """Fit reference baselines on the frozen split (recorded as runs)."""
    from .baselines import run_baselines
    from .leaderboard import format_table, leaderboard

    STATE["command"] = "baseline"
    project = _project()
    runs = run_baselines(project, only.split(",") if only else None, progress=progress)
    rows = leaderboard(project)
    emit(
        {"runs": [r.id for r in runs], "leaderboard": rows},
        "\n" + format_table(rows, project.config.metric),
        [
            {
                "command": 'trainwright train --hypothesis "default recipe"',
                "why": "train the default neural model for this modality",
            }
        ],
    )


def _resolve_exp(
    project: Any, config: Path | None, overrides: list[str], from_run: str | None
) -> tuple[Any, str | None]:
    import yaml

    from .config import ExperimentConfig, resolve_experiment
    from .leaderboard import resolve_ref

    base = None
    parent = None
    if from_run:
        run = resolve_ref(project, from_run)
        if run.meta.get("kind") == "baseline":
            raise CLIError(f"{run.id} is a baseline; --from-run needs a neural run.")
        base = ExperimentConfig.model_validate(yaml.safe_load(run.config_path.read_text()))
        parent = run.id
    if config is None:
        return resolve_experiment(project.config, None, overrides, base=base), parent
    path = config if config.exists() else project.experiments_dir / config
    if not path.exists() and not str(config).endswith(".yaml"):
        path = project.experiments_dir / f"{config}.yaml"
    if not path.exists():
        raise CLIError(f"Experiment config {config} not found.")
    return resolve_experiment(project.config, path, overrides, base=base), parent


@app.command()
def train(
    override: list[str] = typer.Option(
        [], "-o", "--override", help="key=value, e.g. -o optim.lr=3e-4 (repeatable)."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Experiment YAML (path or name in experiments/)."
    ),
    from_run: str | None = typer.Option(
        None, "--from-run", help="Start from this run's exact config."
    ),
    hypothesis: str | None = typer.Option(
        None, "--hypothesis", "-H", help="What this run tests and why."
    ),
    name: str | None = typer.Option(None, help="Short run name (default: model name)."),
    tag: list[str] = typer.Option([], help="Tags (repeatable)."),
    dry_run: bool = typer.Option(False, help="Print the resolved config and exit."),
) -> None:
    """Train a model on the frozen split, logging everything to runs/<id>/."""
    from .config import dump_yaml
    from .pipeline import train as _train

    STATE["command"] = "train"
    project = _project()
    project.load_user_plugins()
    exp, parent = _resolve_exp(project, config, override, from_run)
    if dry_run:
        emit(exp.model_dump(), dump_yaml(exp))
        return
    warnings = []
    if not hypothesis:
        warnings.append(
            "no --hypothesis given: say what this run tests so the experiment log stays meaningful"
        )
    run = _train(
        project, exp, name=name, hypothesis=hypothesis, parent=parent, tags=tag, progress=progress
    )
    summ = run.summary
    from ._utils import read_json

    diag = read_json(run.diagnosis_path)
    m = project.config.metric
    lines = [
        f"\nrun {run.id}: best val_{m}={summ['val'].get(m, float('nan')):.4f} at epoch {summ['best_epoch']}/"
        f"{summ['epochs_run']} ({summ['stop_reason']})",
        f"curves: {_rel(run.curves_path)}",
        f"diagnosis: {diag['verdict']}",
    ]
    lines += [f"  - [{f['severity']}] {f['code']}: {f['message']}" for f in diag["findings"]]
    nxt = [{"command": n["command"], "why": n["why"]} for n in diag.get("next_experiments", [])]
    if not nxt:
        nxt = [{"command": "trainwright runs", "why": "compare against baselines and earlier runs"}]
    emit({"run": run.id, "summary": summ, "diagnosis": diag}, "\n".join(lines), nxt, warnings)


@app.command()
def sweep(
    override: list[str] = typer.Option(
        [], "-o", help="key=v1,v2,... grid axes; key=value for fixed overrides."
    ),
    from_run: str | None = typer.Option(None, "--from-run"),
    hypothesis: str | None = typer.Option(None, "--hypothesis", "-H"),
    max_runs: int = typer.Option(12, help="Refuse grids larger than this."),
) -> None:
    """Grid over a few overrides, e.g. -o optim.lr=1e-3,3e-3 -o model.params.dropout=0.1,0.3."""
    import itertools

    from .leaderboard import format_table, leaderboard
    from .pipeline import train as _train

    STATE["command"] = "sweep"
    project = _project()
    axes: list[tuple[str, list[str]]] = []
    fixed: list[tuple[str, list[str]]] = []
    for o in override:
        k, v = o.split("=", 1)
        vals = (
            [x for x in v.split(",") if x != ""]
            if "," in v and not v.strip().startswith("[")
            else [v]
        )
        (axes if len(vals) > 1 else fixed).append((k, vals))
    combos = list(itertools.product(*[[f"{k}={v}" for v in vals] for k, vals in axes])) or [()]
    if len(combos) > max_runs:
        raise CLIError(
            f"Grid has {len(combos)} runs (> --max-runs {max_runs}).",
            "Narrow the grid or use `trainwright tune`.",
        )
    ids: list[str] = []
    for combo in combos:
        ov = [f"{k}={vals[0]}" for k, vals in fixed] + list(combo)
        exp, parent = _resolve_exp(project, None, ov, from_run)
        hyp = f"{hypothesis or 'sweep'} [{', '.join(combo)}]"
        progress(f"\n=== sweep {len(ids) + 1}/{len(combos)}: {', '.join(combo) or '(fixed)'}")
        run = _train(
            project,
            exp,
            name="sweep-" + "-".join(c.split("=")[-1] for c in combo)[:30],
            hypothesis=hyp,
            parent=parent,
            tags=["sweep"],
            progress=progress,
        )
        ids.append(run.id)
    rows = [r for r in leaderboard(project) if r["id"] in ids]
    emit(
        {"runs": ids, "leaderboard": rows},
        "\n" + format_table(rows, project.config.metric),
        [{"command": f"trainwright diagnose {rows[0]['id']}", "why": "inspect the best sweep run"}]
        if rows
        else [],
    )


@app.command()
def tune(
    budget: str = typer.Option("20m", help="Total wall-clock budget, e.g. 20m, 1h."),
    from_run: str | None = typer.Option(
        None, "--from-run", help="Centre the search on this run (default: best neural)."
    ),
    trials: int = typer.Option(8, help="Number of candidate recipes."),
    seed: int = typer.Option(0),
) -> None:
    """Automated recipe search (successive halving) around the best run."""
    from .tune import format_tune, run_tune

    STATE["command"] = "tune"
    project = _project()
    res = run_tune(
        project, budget=budget, from_run=from_run, n_trials=trials, seed=seed, progress=progress
    )
    emit(
        res,
        format_tune(res, project.config.metric),
        [
            {
                "command": f"trainwright diagnose {res['best_run']}",
                "why": "check the tuned run's curves",
            }
        ]
        if res.get("best_run")
        else [],
    )


# --------------------------------------------------------------------------- runs


@app.command("runs")
def runs_cmd(
    all_: bool = typer.Option(False, "--all", help="Include failed/running runs."),
    limit: int = typer.Option(30),
) -> None:
    """Validation leaderboard of all runs (baselines + neural)."""
    from .leaderboard import format_table, leaderboard

    STATE["command"] = "runs"
    project = _project()
    rows = leaderboard(project, include_incomplete=all_)
    nxt = []
    if rows:
        nxt.append(
            {
                "command": f"trainwright diagnose {rows[0]['id']}"
                if rows[0]["kind"] == "neural"
                else 'trainwright train --hypothesis "..."',
                "why": "inspect / improve on the leader",
            }
        )
    emit(rows[:limit], format_table(rows, project.config.metric, limit), nxt)


@app.command()
def show(
    ref: str = typer.Argument("latest", help="Run id, number, 'latest' or 'best'."),
    epochs: int = typer.Option(5, help="Show the last N epochs."),
) -> None:
    """Show one run: status, hypothesis, metrics, recent epochs, diagnosis, files."""
    from ._utils import read_json
    from .leaderboard import resolve_ref

    STATE["command"] = "show"
    project = _project()
    run = resolve_ref(project, ref)
    meta, summ, hist = run.meta, run.summary, run.history
    diag = read_json(run.diagnosis_path) if run.diagnosis_path.exists() else None
    m = project.config.metric
    lines = [
        f"{run.id}  [{meta.get('kind')}] {meta.get('model')}  status={meta.get('status')}",
        f"hypothesis: {meta.get('hypothesis') or '-'}"
        + (f"   parent: {meta['parent']}" if meta.get("parent") else ""),
    ]
    if meta.get("error"):
        lines.append(f"error: {meta['error']}")
    if summ:
        val = summ.get("val", {})
        lines.append(
            "val: "
            + "  ".join(f"{k}={v:.4f}" for k, v in val.items() if isinstance(v, (int, float)))
        )
        if summ.get("best_epoch"):
            lines.append(
                f"best epoch {summ['best_epoch']}/{summ['epochs_run']} ({summ['stop_reason']}), "
                f"{summ.get('n_params', 0):,} params, {summ.get('train_time_s', 0):.0f}s"
            )
    if hist:
        lines.append(f"last {min(epochs, len(hist))} of {len(hist)} epochs:")
        for r in hist[-epochs:]:
            lines.append(
                f"  epoch {r['epoch']:>3}  train_loss {r['train_loss']:.4f}  val_loss {r['val_loss']:.4f}  "
                f"val_{m} {r.get(f'val_{m}', float('nan')):.4f}{'*' if r.get('is_best') else ''}"
            )
    if diag:
        lines.append(
            f"diagnosis: {diag['verdict']}  "
            + ", ".join(f["code"] for f in diag.get("findings", []))
        )
    files = {
        "dir": _rel(run.dir),
        "curves": _rel(run.curves_path) if run.curves_path.exists() else None,
        "config": _rel(run.config_path) if run.config_path.exists() else None,
    }
    for s in ("val", "test"):
        if (run.eval_dir / f"{s}.json").exists():
            files[f"eval_{s}"] = _rel(run.eval_dir / f"{s}.json")
    lines.append("files: " + "  ".join(f"{k}={v}" for k, v in files.items() if v))
    emit(
        {
            "meta": meta,
            "summary": summ,
            "history_tail": hist[-epochs:],
            "diagnosis": diag,
            "files": files,
        },
        "\n".join(lines),
    )


@app.command()
def diagnose(ref: str = typer.Argument("latest")) -> None:
    """Analyse a run's curves: overfitting, underfitting, instability, baseline comparison, next experiments."""
    from .diagnose import diagnose_run
    from .leaderboard import resolve_ref

    STATE["command"] = "diagnose"
    project = _project()
    run = resolve_ref(project, ref)
    d = diagnose_run(project, run)
    write_json(run.diagnosis_path, d)
    lines = [f"{run.id}: {d['verdict']}  (curves: {_rel(run.curves_path)})"]
    for f in d["findings"]:
        lines.append(f"- [{f['severity']}] {f['code']}: {f['message']}")
        for s in f["suggestions"][:3]:
            lines.append(f"    try -o {s['override']}   # {s['why']}")
    emit(
        d,
        "\n".join(lines),
        [{"command": n["command"], "why": n["why"]} for n in d.get("next_experiments", [])],
    )


@app.command()
def features(ref: str = typer.Argument("best-neural"), repeats: int = typer.Option(3)) -> None:
    """Permutation feature importance on val (tabular). Suggests features to drop."""
    from .featsel import format_importance, permutation_importance
    from .leaderboard import resolve_ref

    STATE["command"] = "features"
    project = _project()
    run = resolve_ref(project, ref)
    res = permutation_importance(project, run, repeats=repeats)
    nxt = []
    if res["suggested_drop"]:
        nxt.append(
            {
                "command": "edit trainwright.yaml: data.drop += "
                + json.dumps(res["suggested_drop"]),
                "why": "these features hurt or do not help on val; then re-train from this run",
            }
        )
    emit(res, format_importance(res), nxt)


@app.command()
def evaluate(
    ref: str = typer.Argument("best"),
    split_: str = typer.Option("val", "--split", help="val (free) or test (final, recorded)."),
    reason: str | None = typer.Option(
        None, help="Why this run is the final model (for --split test)."
    ),
) -> None:
    """Error analysis on val, or the one-time final evaluation on the sealed test split."""
    from .evaluate import evaluate as _evaluate
    from .leaderboard import resolve_ref

    STATE["command"] = "evaluate"
    project = _project()
    run = resolve_ref(project, ref)
    res = _evaluate(project, run, split_, reason)
    m = project.config.metric
    met = res["metrics"]
    lines = [
        f"{run.id} on {split_} (n={res['n']}): {m}={met.get(m, float('nan')):.4f}",
        "  "
        + "  ".join(
            f"{k}={v:.4f}" for k, v in met.items() if isinstance(v, (int, float)) and k != m
        ),
    ]
    if res.get("per_class"):
        worst = sorted(res["per_class"], key=lambda r: r["f1"])[:5]
        lines.append(
            "weakest classes: "
            + ", ".join(f"{r['class']} (f1 {r['f1']:.2f}, n={r['support']})" for r in worst)
        )
    if res.get("top_confusions"):
        lines.append(
            "top confusions: "
            + "; ".join(
                f"{c['true']}->{c['predicted']} x{c['count']}" for c in res["top_confusions"][:5]
            )
        )
    if res.get("confidence_routing"):
        r = [x for x in res["confidence_routing"] if x["threshold"] in (0.8, 0.9, 0.95)]
        lines.append(
            "confidence routing: "
            + "; ".join(
                f">={x['threshold']}: {x['coverage']:.0%} covered at {x['accuracy_on_covered']:.1%} acc"
                for x in r
            )
        )
    for k in ("confusion_png", "errors_png", "residuals_png"):
        if res.get(k):
            lines.append(f"{k.replace('_png', '')}: {_rel(res[k])}")
    lines.append(f"predictions: {_rel(run.eval_dir / f'{split_}_predictions.csv')}")
    nxt = (
        [{"command": f"trainwright export {run.id}", "why": "package the final model"}]
        if split_ == "test"
        else [
            {
                "command": f'trainwright evaluate {run.id} --split test --reason "final model"',
                "why": "only once you have finished iterating",
            }
        ]
    )
    emit(res, "\n".join(lines), nxt, res.get("warnings"))


# --------------------------------------------------------------------------- deploy


def _bundle_dir(project: Any, ref: str, auto_export: bool = False) -> Path:
    p = Path(ref)
    if (p / "bundle.json").exists():
        return p
    from .leaderboard import resolve_ref

    run = resolve_ref(project, ref)
    d = project.artifacts_dir / run.id
    if not (d / "bundle.json").exists() and auto_export:
        from .export import export_run

        progress(f"exporting {run.id} first...")
        export_run(project, run)
    if not (d / "bundle.json").exists():
        raise CLIError(
            f"{run.id} has not been exported.", f"Run `trainwright export {run.id}` first."
        )
    return d


@app.command()
def export(
    ref: str = typer.Argument("best"),
    force: bool = typer.Option(False),
    out: Path | None = typer.Option(None, help="Output directory (default artifacts/<run>)."),
) -> None:
    """Export a run to a torch-free ONNX bundle with FastAPI server, Dockerfile and model card."""
    from .export import export_run
    from .leaderboard import resolve_ref

    STATE["command"] = "export"
    project = _project()
    run = resolve_ref(project, ref)
    warnings = []
    if not run.eval_result("test"):
        warnings.append(f"{run.id} has no test evaluation; the model card will say so")
    res = export_run(project, run, out, force)
    if res.get("parity_ok") is False:
        warnings.append(f"ONNX output differs from torch by {res['parity_max_abs_diff']:.2e}")
    rd = _rel(res["dir"])
    emit(
        res,
        f"exported {run.id} -> {rd}  (parity max diff {res.get('parity_max_abs_diff')}, "
        f"smoke test ok)\n  serve: trainwright serve {run.id}\n  docker: docker build -t {project.config.name} {rd}",
        [
            {
                "command": f"trainwright benchmark {run.id}",
                "why": "latency + cost vs the LLM it replaces",
            }
        ],
        warnings,
    )


@app.command()
def benchmark(
    ref: str = typer.Argument("best"),
    batch_size: int = typer.Option(64),
    threads: int = typer.Option(2, help="CPU threads (match your target instance's vCPUs)."),
) -> None:
    """Measure latency/throughput of an exported bundle and estimate cost vs an LLM API."""
    from .benchmark import benchmark_bundle, format_benchmark

    STATE["command"] = "benchmark"
    project = _project()
    res = benchmark_bundle(project, _bundle_dir(project, ref), batch_size, threads)
    emit(
        res,
        format_benchmark(res),
        [{"command": "trainwright report", "why": "write up the project for the human"}],
    )


@app.command()
def serve(ref: str = typer.Argument("best"), host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve an exported bundle locally with uvicorn."""
    import os

    import uvicorn

    STATE["command"] = "serve"
    bdir = _bundle_dir(_project(), ref).resolve()
    sys.path.insert(0, str(bdir))
    os.environ["TW_BUNDLE"] = str(bdir)
    import importlib.util

    spec = importlib.util.spec_from_file_location("tw_bundle_server", bdir / "server.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    uvicorn.run(mod.app, host=host, port=port)


@app.command()
def predict(
    ref: str = typer.Argument("best"),
    input_: Path = typer.Option(
        ..., "--input", "-i", help="JSON list / CSV / text file (one per line) / image."
    ),
    out_path: Path | None = typer.Option(
        None, "--out", "-o", help="Write ALL predictions to this CSV (id, prediction, confidence)."
    ),
    limit: int = typer.Option(20, help="Rows to print when --out is not given."),
) -> None:
    """Predict with a run's exported bundle (exported automatically if needed).

    Input: CSV/Parquet (tabular or text column), JSON list, text file (one per line),
    an image file, or a folder of images (recursively).
    """
    import csv

    from .preprocessing import IMAGE_EXTENSIONS
    from .runtime import Predictor

    STATE["command"] = "predict"
    project = _project()
    pred = Predictor(_bundle_dir(project, ref, auto_export=True))
    suf = input_.suffix.lower()
    ids: list[str]
    if input_.is_dir():
        files = sorted(
            f for f in input_.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        items: list[Any] = [str(f) for f in files]
        ids = [str(f.relative_to(input_)) for f in files]
    elif suf == ".json":
        data = json.loads(input_.read_text())
        items = data["instances"] if isinstance(data, dict) else data
        ids = [str(i) for i in range(len(items))]
    elif suf in (".csv", ".parquet", ".tsv", ".pq", ".jsonl"):
        from .data.records import read_table

        df = read_table(input_)
        if pred.modality == "tabular":
            items = df.astype(object).where(df.notna(), None).to_dict(orient="records")
        else:
            col = project.config.data.text_column or project.config.data.image_column
            if col not in df.columns:
                raise CLIError(
                    f"Column {col!r} not in {input_.name}", f"Columns: {list(df.columns)}"
                )
            items = df[col].astype(str).tolist()
        ids = [str(i) for i in range(len(items))]
    elif pred.modality == "image":
        items, ids = [str(input_)], [input_.name]
    else:
        items = [line for line in input_.read_text().splitlines() if line.strip()]
        ids = [str(i) for i in range(len(items))]
    if out_path is None:
        out = pred.predict(items[:limit])
        emit(out, "\n".join(json.dumps(o) for o in out))
        return
    rows = []
    for i in range(0, len(items), 512):
        rows += pred.predict(items[i : i + 512])
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        if pred.task == "classification":
            w.writerow(["id", "prediction", "confidence"])
            w.writerows([ids[k], r["label"], f"{r['confidence']:.5f}"] for k, r in enumerate(rows))
        else:
            w.writerow(["id", "prediction"])
            w.writerows([ids[k], r["prediction"]] for k, r in enumerate(rows))
    emit(
        {"path": str(out_path), "n": len(rows)},
        f"wrote {len(rows)} predictions to {_rel(out_path)}",
    )


@app.command()
def report() -> None:
    """Write reports/REPORT.md: data, experiments, final evaluation, deployment cost."""
    from .lifecycle import write_report

    STATE["command"] = "report"
    path = write_report(_project())
    emit({"path": str(path)}, f"wrote {_rel(path)}")


# --------------------------------------------------------------------------- reference


@app.command()
def models(modality: str | None = typer.Option(None, help="tabular | image | text")) -> None:
    """List registered model architectures, their params and when to use them."""
    from .models import list_models

    STATE["command"] = "models"
    try:
        project = _project()
        project.load_user_plugins()
        modality = modality or project.config.modality
    except Exception:
        pass
    ms = list_models(modality)
    text = "\n\n".join(
        f"{m['name']} [{m['modality']}]: {m['description']}\n  when: {m['when_to_use']}\n  params: {m['params']}"
        for m in ms
    )
    emit(ms, text)


@app.command()
def schema() -> None:
    """JSON schema of the experiment config (every key you can override with -o)."""
    from .config import ExperimentConfig, ProjectConfig

    STATE["command"] = "schema"
    s = {
        "experiment": ExperimentConfig.model_json_schema(),
        "project": ProjectConfig.model_json_schema(),
    }
    if STATE["json"]:
        emit(s)
        return
    lines = []
    exp = ExperimentConfig.model_json_schema()
    for sec, ref in exp["properties"].items():
        name = ref.get("$ref", "").split("/")[-1]
        props = exp.get("$defs", {}).get(name, {}).get("properties", {})
        for k, p in props.items():
            typ = p.get("type") or "/".join(
                str(x.get("type", x.get("const", ""))) for x in p.get("anyOf", [])
            )
            enum = p.get("enum")
            desc = p.get("description", "")
            lines.append(
                f"{sec}.{k}: {('|'.join(map(str, enum))) if enum else typ} = {p.get('default')!r}  {desc}"
            )
    emit(s, "\n".join(lines))


@app.command()
def mcp() -> None:
    """Run an MCP server (stdio) exposing trainwright as tools. Needs `trainwright[mcp]`."""
    from .mcp_server import main as mcp_main

    mcp_main()


# --------------------------------------------------------------------------- entry


def main() -> None:
    import click

    try:
        rv = app(standalone_mode=False)
        if isinstance(rv, int) and rv:
            sys.exit(rv)
    except SystemExit as e:
        raise e
    except typer.Exit as e:
        sys.exit(e.exit_code)
    except (click.exceptions.Abort, KeyboardInterrupt):
        print("interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        # click/typer usage errors (typer may vendor its own click, so duck-type them)
        if hasattr(e, "show") and hasattr(e, "exit_code"):
            e.show()
            sys.exit(e.exit_code)
        _handle_error(e)


def _handle_error(e: Exception) -> None:
    from pydantic import ValidationError

    from .config import ConfigError
    from .pipeline import LifecycleError
    from .project import ProjectNotFound

    try:
        raise e
    except (
        CLIError,
        ConfigError,
        LifecycleError,
        ProjectNotFound,
        FileNotFoundError,
        KeyError,
        ValueError,
        ValidationError,
    ) as e:
        hint = getattr(e, "hint", None)
        msg = str(e).strip("'\"")
        if STATE["json"]:
            print(
                to_json(
                    {
                        "ok": False,
                        "command": STATE["command"],
                        "error": msg,
                        "hint": hint,
                        "type": type(e).__name__,
                    }
                )
            )
        else:
            print(f"error: {msg}", file=sys.stderr)
            if hint:
                print(f"hint: {hint}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if STATE["json"]:
            print(
                to_json(
                    {
                        "ok": False,
                        "command": STATE["command"],
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-2000:],
                    }
                )
            )
        else:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
