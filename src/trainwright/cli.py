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
from ._utils import fmt_seconds, to_json, write_json

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


def _estimate_llm_tokens(cfg: Any, data: Path) -> int:
    """Rough tokens for the LLM call this model would replace: instructions + label list + input."""
    base = 250
    n_classes = 0
    try:
        from .data.records import read_table

        if cfg.modality == "image":
            return base + 1600  # ~1092x1092 image ≈ 1600 tokens
        df = read_table(data)
        if cfg.task == "classification" and cfg.data.target:
            n_classes = int(df[cfg.data.target].nunique())
        if cfg.modality == "text" and cfg.data.text_column:
            avg = float(df[cfg.data.text_column].astype(str).str.len().mean())
        else:
            avg = (
                float(
                    df.drop(columns=[cfg.data.target], errors="ignore")
                    .head(200)
                    .to_json(orient="records")
                    .__len__()
                )
                / 200
            )
        return int(base + 6 * n_classes + avg / 4)
    except Exception:
        return 800


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
    cfg.cost.llm_input_tokens_per_call = _estimate_llm_tokens(cfg, data_abs)
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
    from .autopilot import profile_project

    STATE["command"] = "profile"
    project = _project()
    prof = profile_project(project, keep_ids=keep_ids)
    cfg = project.config
    auto_dropped: list[str] = prof.get("auto_dropped", [])
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
                "command": "trainwright auto --budget 15m",
                "why": "after fixing data.drop / group_column / features.py: split, baselines, strong "
                "recipes, ensemble and finalize in one time-boxed command",
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


@app.command()
def auto(
    budget: str = typer.Option("20m", help="Total wall-clock budget, e.g. 10m, 1h."),
    override: list[str] = typer.Option(
        [], "-o", "--override", help="key=value applied to every neural run (repeatable)."
    ),
    finish: bool = typer.Option(
        True, help="Refit the val-best model on train+val at the end (served by `predict`)."
    ),
    seeds: bool = typer.Option(
        True, help="Add seed replicas of the best neural recipe if time allows."
    ),
    background: bool = typer.Option(
        False,
        "--background",
        help="Detach and return at once; follow with `trainwright wait` (for budgets longer than "
        "your tool timeout).",
    ),
) -> None:
    """Run the whole search in one command: profile, split, baselines, strong recipes, ensemble, finalize."""
    import subprocess
    import sys

    from ._utils import parse_duration
    from .autopilot import autopilot, log_path, read_state
    from .config import parse_override
    from .leaderboard import format_table, leaderboard

    STATE["command"] = "auto"
    project = _project()
    project.load_user_plugins()
    budget_s = parse_duration(budget) or 1200.0
    ov = dict(parse_override(o) for o in override)
    if background:
        st = read_state(project)
        if st and st.get("status") == "running":
            raise CLIError(f"auto is already running (pid {st['pid']}): `trainwright wait`")
        project.ensure_dirs()
        args = [sys.executable, "-m", "trainwright", "auto", "--budget", f"{int(budget_s)}s"]
        args += [x for o in override for x in ("-o", o)]
        args += [] if finish else ["--no-finish"]
        args += [] if seeds else ["--no-seeds"]
        lp = log_path(project)
        with lp.open("a") as f:
            proc = subprocess.Popen(
                args, cwd=project.root, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
            )
        import time as _time

        from .autopilot import state_path

        now = _time.time()  # written now so `status`/`wait` see it before the child starts up
        write_json(
            state_path(project),
            {
                "status": "running",
                "pid": proc.pid,
                "started": now,
                "deadline": now + budget_s,
                "budget_s": budget_s,
            },
        )
        emit(
            {"pid": proc.pid, "log": str(lp), "budget_s": budget_s},
            f"auto started in the background (pid {proc.pid}, budget {fmt_seconds(budget_s)}); "
            f"log: {_rel(lp)}",
            [
                {
                    "command": "trainwright wait --timeout 7m",
                    "why": "blocks until auto finishes (or the timeout), printing progress; repeat as needed",
                }
            ],
        )
        return
    res = autopilot(project, budget_s, ov, finish=finish, seeds=seeds, progress=progress)
    rows = leaderboard(project)
    best = res.get("best") or {}
    lines = ["", format_table(rows, project.config.metric)]
    nxt = [
        {
            "command": f'trainwright evaluate {best.get("id", "best")} --split test --reason "final estimate"',
            "why": "the one honest number: score the val-picked model on the sealed test split",
        },
        {
            "command": "trainwright predict <new data>",
            "why": "serves the finalized model" if res.get("final") else "serves the best run",
        },
    ]
    if project.config.modality == "tabular":
        nxt.append(
            {
                "command": "trainwright features best-neural  (then write features.py and rerun auto)",
                "why": "domain features are where you beat the search; auto skips steps already done",
            }
        )
    else:
        nxt.append(
            {
                "command": f"trainwright evaluate {best.get('id', 'best')}",
                "why": "val error analysis: confusions and worst examples (label noise? missing classes?)",
            }
        )
    emit(res, "\n".join(lines), nxt)


@app.command()
def wait(
    timeout: str = typer.Option("7m", help="Give up waiting after this long (auto keeps running)."),
) -> None:
    """Wait for a background `trainwright auto` to finish, printing its progress."""
    import time as _time

    from ._utils import parse_duration
    from .autopilot import log_path, read_state
    from .leaderboard import format_table, leaderboard

    STATE["command"] = "wait"
    project = _project()
    st = read_state(project)
    if st is None:
        raise CLIError("No auto run found. Start one with `trainwright auto --background`.")
    lp = log_path(project)
    seen = lp.stat().st_size if lp.exists() else 0
    end = _time.time() + (parse_duration(timeout) or 420.0)
    while st and st.get("status") == "running" and _time.time() < end:
        _time.sleep(5)
        if lp.exists() and lp.stat().st_size > seen:
            with lp.open() as f:
                f.seek(seen)
                for line in f.read().splitlines():
                    if line.startswith("[auto"):
                        progress(line)
            seen = lp.stat().st_size
        st = read_state(project)
    assert st is not None
    status_ = st.get("status")
    if status_ == "running":
        left = max(0.0, float(st.get("deadline", 0)) - _time.time())
        emit(
            st,
            f"auto still running (about {fmt_seconds(left)} of budget left)",
            [{"command": "trainwright wait --timeout 7m", "why": "keep waiting"}],
        )
        return
    if status_ in ("failed", "died"):
        tail = lp.read_text().splitlines()[-15:] if lp.exists() else []
        raise CLIError(f"auto {status_}: {st.get('error')}\n" + "\n".join(tail))
    rows = leaderboard(project)
    best = st.get("best") or "best"
    emit(
        st,
        "auto finished\n\n" + format_table(rows, project.config.metric),
        [
            {
                "command": f'trainwright evaluate {best} --split test --reason "final estimate"',
                "why": "the one honest number for the val-picked model (once)",
            },
            {"command": "trainwright predict -i <new data>", "why": "serves the finalized model"},
        ],
    )


@app.command()
def bag(
    ref: str = typer.Argument("best-neural", help="Run whose recipe to bag."),
    folds: int = typer.Option(5, help="Number of folds (= models in the bag)."),
) -> None:
    """Cross-validated bag: k copies of a recipe on k folds of train+val, scored out-of-fold."""
    from .bagging import build_bag
    from .leaderboard import format_table, leaderboard

    STATE["command"] = "bag"
    project = _project()
    project.load_user_plugins()
    run = build_bag(project, ref, k=folds, progress=progress)
    rows = leaderboard(project)
    emit(
        {"run": run.id, "summary": run.summary},
        "\n" + format_table(rows, project.config.metric),
        [
            {
                "command": "trainwright ensemble",
                "why": "the bag competes in greedy ensembles with its out-of-fold predictions",
            },
            {
                "command": f"trainwright finalize {run.id}",
                "why": "free: the fold models are already trained on train+val",
            },
        ],
    )


@app.command("leaderboard", hidden=True)
def leaderboard_alias(
    limit: int = typer.Option(30, help="Rows to show."),
) -> None:
    """Alias for `trainwright runs`."""
    runs_cmd(all_=False, limit=limit)


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
    force: bool = typer.Option(
        False, help="Recompute a test evaluation (records another test access)."
    ),
) -> None:
    """Error analysis on val, or the one-time final evaluation on the sealed test split.

    Re-running ``--split test`` for the same run returns the cached result without a new access."""
    from .evaluate import evaluate as _evaluate
    from .leaderboard import resolve_ref

    STATE["command"] = "evaluate"
    project = _project()
    run = resolve_ref(project, ref)
    res = _evaluate(project, run, split_, reason, force)
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
        [
            {
                "command": f"trainwright finalize {run.id}",
                "why": "retrain this recipe on all the data (incl. test, now scored): the production model to predict/export with",
            }
        ]
        if split_ == "test"
        else [
            {
                "command": f'trainwright evaluate {run.id} --split test --reason "final model"',
                "why": "only once you have finished iterating",
            }
        ]
    )
    emit(res, "\n".join(lines), nxt, res.get("warnings"))


@app.command()
def finalize(
    ref: str = typer.Argument("best"),
    include_test: bool | None = typer.Option(
        None,
        "--include-test/--no-include-test",
        help="Train on the test split too. Default: yes once this run's test evaluation is recorded "
        "(the number is already reported, so holding the data back only costs accuracy).",
    ),
) -> None:
    """Retrain the selected run's recipe on all usable data for its best epoch count: the production model."""
    from .pipeline import finalize as _finalize

    STATE["command"] = "finalize"
    project = _project()
    from .leaderboard import resolve_ref

    run = resolve_ref(project, ref)
    warnings = []
    if include_test is None:
        include_test = run.eval_result("test") is not None
    if include_test and not run.eval_result("test"):
        raise CLIError(
            f"{run.id} has no recorded test evaluation.",
            f"Run `trainwright evaluate {run.id} --split test` first: once test is in the training data it can never be evaluated.",
        )
    if not run.eval_result("test"):
        warnings.append(
            f"{run.id} has no test evaluation yet; record it before shipping the final model"
        )
    final = _finalize(project, run, include_test, progress=progress)
    emit(
        {"run": final.id, "parent": run.id, "trained_on": final.meta.get("trained_on")},
        f"final model {final.id}: {run.id}'s recipe on {'+'.join(final.meta.get('trained_on', []))} "
        f"({final.summary.get('epochs_run')} epochs). Quote {run.id}'s held-out val/test numbers for it.",
        [{"command": f"trainwright export {final.id}", "why": "ship the production model"}],
        warnings,
    )


@app.command()
def ensemble(
    refs: list[str] = typer.Argument(None, help="Member runs (default: the top runs on val)."),
    top: int = typer.Option(6, help="Candidate pool: this many top runs on val."),
    seeds: int = typer.Option(1, help="Also train N-1 seed replicas of the best neural run first."),
) -> None:
    """Average the best runs' calibrated predictions; recorded as a run that competes on the leaderboard."""
    from .ensemble import build_ensemble
    from .leaderboard import format_table, leaderboard

    STATE["command"] = "ensemble"
    project = _project()
    run = build_ensemble(project, refs or None, top, seeds, progress=progress)
    if run is None:
        emit(
            {"run": None},
            "no ensemble beat the best single run on val; keep the single model",
            [{"command": "trainwright runs", "why": "the leader stays the final candidate"}],
        )
        return
    summ = run.summary
    m = project.config.metric
    rows = leaderboard(project)
    emit(
        {"run": run.id, "summary": summ},
        "\n".join(
            [
                f"{run.id}: weights {dict(zip(summ['members'], summ['weights']))} "
                f"(chosen on val {summ['selection_metric']})",
                "",
                format_table(rows, m, 8),
            ]
        ),
        [
            {
                "command": f'trainwright evaluate {run.id} --split test --reason "final model"',
                "why": "if the ensemble leads (and isn't just tied), make it the final model",
            }
        ],
    )


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
    from .tracking import Run as _Run

    src = run
    if run.meta.get("kind") == "final" and run.meta.get("parent"):
        src = _Run(project.runs_dir / run.meta["parent"])
    if not src.eval_result("test"):
        warnings.append(f"{src.id} has no test evaluation; the model card will say so")
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
    use_final: bool = typer.Option(
        True, "--final/--no-final", help="Use the run's finalized production model if one exists."
    ),
    columns: str | None = typer.Option(
        None,
        "--columns",
        help="Comma-separated output columns for --out, e.g. 'prediction' or 'filename,prediction'. "
        "Available: id, filename (image basename), prediction, confidence.",
    ),
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
    warnings: list[str] = []
    if not use_final:
        target = ref
    else:
        target = ref
        try:
            from .leaderboard import resolve_ref
            from .tracking import list_runs

            run = resolve_ref(project, ref)
            if run.meta.get("kind") != "final":
                finals = [
                    r
                    for r in list_runs(project.runs_dir)
                    if r.meta.get("kind") == "final"
                    and r.meta.get("parent") == run.id
                    and r.meta.get("status") == "completed"
                ]
                if finals:
                    target = finals[-1].id
                    progress(
                        f"using production model {target} (finalized from {run.id}); --no-final to opt out"
                    )
                else:
                    warnings.append(
                        f"{run.id} was trained on the train split only. For production predictions run "
                        f"`trainwright finalize {run.id}` first (more data, usually better)."
                    )
        except FileNotFoundError:
            pass  # a bundle directory path
    pred = Predictor(_bundle_dir(project, target, auto_export=True))
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
        emit(out, "\n".join(json.dumps(o) for o in out), warnings=warnings)
        return
    rows = []
    for i in range(0, len(items), 512):
        rows += pred.predict(items[i : i + 512])
    clf = pred.task == "classification"
    cols = (
        [c.strip() for c in columns.split(",") if c.strip()]
        if columns
        else (["id", "prediction", "confidence"] if clf else ["id", "prediction"])
    )
    avail = {"id", "filename", "prediction"} | ({"confidence"} if clf else set())
    if bad := [c for c in cols if c not in avail]:
        raise CLIError(f"Unknown --columns {bad}", f"Available: {sorted(avail)}")

    def cell(k: int, r: dict[str, Any], c: str) -> Any:
        if c == "id":
            return ids[k]
        if c == "filename":
            return Path(ids[k]).name
        if c == "confidence":
            return f"{r['confidence']:.5f}"
        return r["label"] if clf else r["prediction"]

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([cell(k, r, c) for c in cols] for k, r in enumerate(rows))
    emit(
        {"path": str(out_path), "n": len(rows), "model": target},
        f"wrote {len(rows)} predictions to {_rel(out_path)} (model {target})",
        warnings=warnings,
    )


@app.command()
def report() -> None:
    """Write reports/REPORT.md: data, experiments, final evaluation, deployment cost."""
    from .lifecycle import write_report

    STATE["command"] = "report"
    path = write_report(_project())
    emit({"path": str(path)}, f"wrote {_rel(path)}")


# --------------------------------------------------------------------------- distillation


def _read_instructions(text: str) -> str:
    if text.startswith("@"):
        return Path(text[1:]).read_text()
    return text


@app.command()
def label(
    input_: Path = typer.Option(
        ..., "--input", "-i", help="Unlabelled CSV/Parquet or a folder of images."
    ),
    labels: str | None = typer.Option(
        None, help="Comma-separated label set (default: the project's classes)."
    ),
    column: str | None = typer.Option(
        None, help="Text column to label (default: whole row for tables)."
    ),
    instructions: str = typer.Option("", help="Labelling guidelines, or @file.md."),
    model: str = typer.Option(
        "claude-opus-5-5", help="Claude model id (use the one you run in production to distil it)."
    ),
    effort: str = typer.Option(
        "low", help="low | medium | high (classification rarely needs more than low)."
    ),
    max_cost_usd: float = typer.Option(
        5.0, help="Hard spend cap; labelling stops before exceeding it."
    ),
    limit: int | None = typer.Option(None, help="Only label the first N items (pilot run)."),
    out: Path = typer.Option(Path("labelled.csv"), help="Output CSV."),
    estimate: bool = typer.Option(False, "--estimate", help="Only print the cost estimate."),
    concurrency: int = typer.Option(8),
) -> None:
    """Label data with Claude (structured output, confidence, rationale) under a spend cap."""
    import pandas as pd

    from .label import estimate_cost, label_items
    from .preprocessing import IMAGE_EXTENSIONS

    STATE["command"] = "label"
    if labels:
        label_set = [x.strip() for x in labels.split(",") if x.strip()]
    else:
        try:
            from .data.split import load_split

            sp = load_split(_project())
            label_set = list(sp["classes"]) if sp and sp.get("classes") else []
        except Exception:
            label_set = []
    if len(label_set) < 2:
        raise CLIError(
            "Need at least two labels.",
            "Pass --labels 'a,b,c' (or run inside a project with a split).",
        )
    if input_.is_dir():
        files = sorted(
            f for f in input_.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        modality, items = "image", [str(f) for f in files]
        df = pd.DataFrame({"path": [str(f.resolve()) for f in files]})
    else:
        from .data.records import read_table

        df = read_table(input_)
        if column:
            if column not in df.columns:
                raise CLIError(f"Column {column!r} not found.", f"Columns: {list(df.columns)}")
            modality, items = "text", df[column].astype(str).tolist()
        else:
            modality = "tabular"
            items = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    if limit:
        items, df = items[:limit], df.iloc[:limit].copy()
    instr = _read_instructions(instructions)
    est = estimate_cost(items, modality, label_set, instr, model)
    if estimate:
        emit(
            est,
            f"{est['n_items']} items x ~{est['avg_input_tokens']} input tokens on {model}: "
            f"~${est['estimated_usd']:.2f} (cap ${max_cost_usd})",
        )
        return
    if est["estimated_usd"] > max_cost_usd:
        raise CLIError(
            f"Estimated cost ${est['estimated_usd']:.2f} exceeds --max-cost-usd {max_cost_usd}.",
            "Pilot with --limit 50, use a cheaper --model, or raise --max-cost-usd deliberately.",
        )
    progress(
        f"labelling {len(items)} {modality} items with {model} (estimate ${est['estimated_usd']:.2f}, cap ${max_cost_usd})"
    )
    rows, spend = label_items(
        items,
        label_set,
        instr,
        modality,
        model,
        effort,
        max_cost_usd,
        concurrency,
        progress=progress,
    )
    df = df.reset_index(drop=True)
    df["label"] = [r.get("label") for r in rows]
    df["llm_confidence"] = [r.get("confidence") for r in rows]
    df["llm_rationale"] = [r.get("rationale") for r in rows]
    df["label_source"] = [f"llm:{model}" if r.get("label") else r.get("status") for r in rows]
    df.to_csv(out, index=False)
    ok = int(df["label"].notna().sum())
    data_flag = (
        f"--data {_rel(out)} --target label"
        + (" --image-column path" if modality == "image" else "")
        + (f" --text-column {column}" if modality == "text" else "")
    )
    emit(
        {"out": str(out), "labelled": ok, "spend": spend, "estimate": est},
        f"labelled {ok}/{len(rows)} items -> {_rel(out)}  (spent ${spend['usd']:.3f}, {spend['refusals']} refusals, "
        f"{spend['errors']} errors{', BUDGET HIT' if spend['budget_hit'] else ''})\n"
        "NOTE: models trained on these labels learn to agree with the LLM. Hand-check a sample "
        "and use `trainwright review` after training to find label errors.",
        [
            {
                "command": f"trainwright init {data_flag}",
                "why": "turn the labelled data into a project",
            }
        ],
    )


@app.command()
def review(
    ref: str = typer.Argument("best-neural"),
    top: int = typer.Option(50, help="Queue size."),
    adjudicate_model: str | None = typer.Option(
        None,
        "--adjudicate",
        help="Ask this Claude model to re-label disputed items blind (costs money).",
    ),
    instructions: str = typer.Option("", help="Guidelines for the adjudicator, or @file.md."),
    max_cost_usd: float = typer.Option(2.0),
    apply_agreed: bool = typer.Option(
        False, help="Append items where the adjudicator sides with the model to corrections.csv."
    ),
) -> None:
    """Rank likely label errors (confident val disagreements); optionally adjudicate with an LLM."""
    import csv

    from .label import adjudicate, review_queue, write_review
    from .leaderboard import resolve_ref

    STATE["command"] = "review"
    project = _project()
    run = resolve_ref(project, ref)
    q = review_queue(project, run, top)
    spend = None
    agreed: list[dict[str, Any]] = []
    if adjudicate_model:
        from .data.split import load_split

        classes = (load_split(project) or {}).get("classes") or []
        q, spend = adjudicate(
            q,
            classes,
            _read_instructions(instructions),
            project.config.modality,
            adjudicate_model,
            max_cost_usd,
        )
        agreed = [x for x in q["queue"] if x.get("llm_verdict") == x["model_prediction"]]
        if apply_agreed and agreed:
            corr = project.root / "corrections.csv"
            new = not corr.exists()
            with corr.open("a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["id", "label", "source", "note"])
                for x in agreed:
                    w.writerow(
                        [
                            x["id"],
                            x["model_prediction"],
                            f"llm:{adjudicate_model}",
                            f"was {x['given_label']}; model {x['model_confidence']:.2f}",
                        ]
                    )
    paths = write_review(project, q)
    lines = [
        f"{q['n_val_disagreements']} val items where {run.id} disagrees with the label; top {len(q['queue'])} "
        f"by model confidence -> {_rel(paths['queue'])}"
    ]
    if paths.get("sheet"):
        lines.append(f"image grid: {_rel(paths['sheet'])}")
    for x in q["queue"][:8]:
        lines.append(
            f"  {x['id'][:40]:<40} label={x['given_label']:<18} model={x['model_prediction']} "
            f"({x['model_confidence']:.2f})"
            + (f" llm={x['llm_verdict']}" if x.get("llm_verdict") else "")
        )
    if spend:
        lines.append(
            f"adjudicator {adjudicate_model}: sided with the model on {len(agreed)}/{len(q['queue'])} "
            f"(likely label errors), spent ${spend['usd']:.3f}"
        )
    nxt = [
        {
            "command": f"edit {_rel(paths['corrections'])} (id,label,source,note), then set data.corrections: corrections.csv "
            "in trainwright.yaml and re-train",
            "why": "fix label errors; the split stays frozen",
        }
    ]
    emit({**q, "paths": paths, "spend": spend, "agreed": len(agreed)}, "\n".join(lines), nxt)


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
