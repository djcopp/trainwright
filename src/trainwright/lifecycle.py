"""Where is this project in the ML lifecycle, and what should happen next?

``status()`` is the agent's compass: it checks each stage against what is on disk and returns
the next commands to run with the reason for each. ``write_report()`` renders the whole
project (data, experiments, final evaluation, deployment cost) as one Markdown report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._utils import read_json, read_jsonl
from .data.split import load_split
from .leaderboard import best_run, format_table, leaderboard
from .project import Project
from .tracking import Run, list_runs


def _step(
    name: str, done: bool, detail: str, command: str | None = None, why: str | None = None
) -> dict[str, Any]:
    return {"stage": name, "done": done, "detail": detail, "command": command, "why": why}


def status(project: Project) -> dict[str, Any]:
    cfg = project.config
    split = load_split(project)
    profile = read_json(project.profile_path) if project.profile_path.exists() else None
    runs = list_runs(project.runs_dir)
    split_hash = split["split_hash"] if split else None
    current = [
        r
        for r in runs
        if r.meta.get("split_hash") == split_hash and r.meta.get("status") == "completed"
    ]
    baselines = [r for r in current if r.meta.get("kind") == "baseline"]
    neural = [r for r in current if r.meta.get("kind") == "neural"]
    failed = [r for r in runs if r.meta.get("status") in ("failed", "interrupted")]
    running = [r for r in runs if r.meta.get("status") == "running"]
    ledger = read_jsonl(project.ledger_path)
    best = best_run(project)
    best_neural = best_run(project, kind="neural")
    exported = (
        sorted(p.name for p in project.artifacts_dir.iterdir())
        if project.artifacts_dir.exists()
        else []
    )
    bench = (
        sorted(p.stem for p in (project.state_dir / "benchmark").glob("*.json"))
        if (project.state_dir / "benchmark").exists()
        else []
    )

    steps = [
        _step(
            "profile",
            profile is not None,
            f"profiled {profile.get('n')} examples" if profile else "data not profiled",
            "trainwright profile",
            "look at the data first: types, leakage, imbalance, the recommended metric",
        ),
        _step(
            "split",
            split is not None,
            f"{split['strategy']} split {split['counts']} (hash {split_hash})"
            if split
            else "no frozen split",
            "trainwright auto --budget 15m",
            "one command: freeze the split, baselines, strong recipes, ensemble, finalize (stops on time)",
        ),
        _step(
            "baselines",
            bool(baselines),
            f"{len(baselines)} baseline run(s)",
            "trainwright auto --budget 15m",
            "baselines plus the strong recipes for this modality, ensemble and finalize",
        ),
        _step(
            "train",
            bool(neural),
            f"{len(neural)} neural run(s)",
            "trainwright auto --budget 15m",
            "train the strong default recipes for this modality (or `trainwright train` by hand)",
        ),
    ]
    n_with_hyp = sum(1 for r in neural if r.meta.get("hypothesis"))
    steps.append(
        _step(
            "iterate",
            len(neural) >= 3,
            f"{len(neural)} neural runs, {n_with_hyp} with a hypothesis",
            f"trainwright diagnose {best_neural['id']}"
            if best_neural
            else "trainwright diagnose latest",
            "run 2+ deliberate follow-up experiments chosen from the diagnosis (change one thing at a time)",
        )
    )
    test_runs = sorted({e["run"] for e in ledger})
    steps.append(
        _step(
            "final_evaluation",
            bool(ledger),
            f"test evaluated for {test_runs}" if ledger else "test split still sealed",
            f'trainwright evaluate {best["id"] if best else "<best-run>"} --split test --reason "final model"',
            "evaluate the single selected model on test exactly once",
        )
    )
    finals = [
        r for r in runs if r.meta.get("kind") == "final" and r.meta.get("status") == "completed"
    ]
    steps.append(
        _step(
            "finalize",
            bool(finals),
            f"production model(s): {[r.id for r in finals]}"
            if finals
            else "no production model yet",
            f"trainwright finalize {best_neural['id'] if best_neural else '<run>'}",
            "retrain the chosen recipe on train+val for the production model",
        )
    )
    target = finals[-1].id if finals else (best["id"] if best else "<run>")
    steps.append(
        _step(
            "export",
            bool(exported),
            f"bundles: {exported}" if exported else "nothing exported",
            f"trainwright export {target}",
            "package a torch-free ONNX bundle + FastAPI server",
        )
    )
    steps.append(
        _step(
            "benchmark",
            bool(bench),
            f"benchmarked: {bench}" if bench else "not benchmarked",
            f"trainwright benchmark {exported[0] if exported else '<run>'}",
            "measure latency/throughput and estimate cost vs the LLM call it replaces",
        )
    )
    steps.append(
        _step(
            "report",
            (project.reports_dir / "REPORT.md").exists(),
            "reports/REPORT.md",
            "trainwright report",
            "write the project report for the human",
        )
    )

    todo = [s for s in steps if not s["done"]]
    next_steps = [{"command": s["command"], "why": s["why"]} for s in todo[:2]]
    warnings = []
    if failed:
        warnings.append(
            f"{len(failed)} failed/interrupted run(s): {[r.id for r in failed][-3:]}; "
            "`trainwright show <id>` for the error"
        )
    if running:
        warnings.append(
            f"{len(running)} run(s) marked running: {[r.id for r in running]} "
            "(if no process is training them, they crashed)"
        )
    if len(test_runs) > 1:
        warnings.append(
            f"test set used for {len(test_runs)} different runs: treat test numbers as optimistic"
        )
    if best and best.get("kind") == "baseline" and neural:
        warnings.append(
            f"best run is a baseline ({best['id']}); the neural model has not beaten it yet"
        )
    return {
        "project": cfg.name,
        "task": f"{cfg.task} / {cfg.modality}",
        "metric": f"{cfg.metric} ({cfg.metric_mode})",
        "best_run": {k: best.get(k) for k in ("id", "kind", "model", f"val_{cfg.metric}")}
        if best
        else None,
        "stages": steps,
        "next_steps": next_steps,
        "warnings": warnings,
    }


def format_status(s: dict[str, Any]) -> str:
    lines = [f"project {s['project']}  [{s['task']}]  metric {s['metric']}"]
    if s["best_run"]:
        b = s["best_run"]
        val = [v for k, v in b.items() if k.startswith("val_")][0]
        lines.append(f"best: {b['id']} ({b['model']}) val={val:.4f}")
    lines.append("")
    for st in s["stages"]:
        lines.append(f"  [{'x' if st['done'] else ' '}] {st['stage']:<17} {st['detail']}")
    for w in s["warnings"]:
        lines.append(f"WARNING: {w}")
    if s["next_steps"]:
        lines.append("")
        lines.append("next:")
        for n in s["next_steps"]:
            lines.append(f"  $ {n['command']}\n      # {n['why']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- report


def write_report(project: Project) -> Path:
    cfg = project.config
    rd = project.reports_dir
    rd.mkdir(parents=True, exist_ok=True)
    rel = lambda p: Path(*[".."]) / Path(p).relative_to(project.root)  # noqa: E731
    m = cfg.metric
    split = load_split(project)
    profile = read_json(project.profile_path) if project.profile_path.exists() else {}
    rows = leaderboard(project)
    best = best_run(project)
    st = status(project)
    L: list[str] = [f"# {cfg.name}: {cfg.task} on {cfg.modality} data", ""]
    if cfg.description:
        L += [cfg.description, ""]
    L += ["## Summary", ""]
    if best:
        L.append(
            f"- Best model on validation: **{best['id']}** (`{best['model']}`), val {m} = **{best[f'val_{m}']:.4f}**"
        )
        b_base = best_run(project, kind="baseline")
        b_neural = best_run(project, kind="neural")
        if b_base and b_neural:
            sign = 1 if cfg.metric_mode == "max" else -1
            delta = sign * (b_neural[f"val_{m}"] - b_base[f"val_{m}"])
            L.append(
                f"- Best neural vs best baseline: {b_neural[f'val_{m}']:.4f} vs {b_base[f'val_{m}']:.4f} "
                f"({'+' if delta >= 0 else ''}{delta:.4f} in favour of {'neural' if delta >= 0 else 'the baseline'})"
            )
    ledger = read_jsonl(project.ledger_path)
    for e in ledger:
        L.append(
            f"- Test {m} for {e['run']}: **{e['score']:.4f}** ({e.get('reason') or 'no reason given'})"
        )
    if len({e["run"] for e in ledger}) > 1:
        L.append("- Note: test was used for more than one run, so test numbers are optimistic.")
    L.append("")

    L += ["## Data", ""]
    if profile:
        L.append(f"- {profile.get('n')} labelled examples; target: {profile.get('target')}")
        for r in profile.get("recommendations", []):
            L.append(f"- {r}")
    if split:
        L.append(
            f"- Frozen {split['strategy']} split (seed {split['seed']}): {split['counts']}, hash `{split['split_hash']}`"
        )
    L.append("")

    L += ["## Experiments (validation leaderboard)", "", "```", format_table(rows, m), "```", ""]
    L += ["### Experiment journal", ""]
    for r in list_runs(project.runs_dir):
        meta = r.meta
        s = r.summary
        v = (s.get("val") or {}).get(m) if meta.get("kind") != "final" else None
        L.append(
            f"- `{r.id}` [{meta.get('status')}] {meta.get('hypothesis') or '(no hypothesis)'}"
            + (f" → val {m} {v:.4f}" if isinstance(v, (int, float)) else "")
            + (
                " (production model; val is in its training data)"
                if meta.get("kind") == "final"
                else ""
            )
        )
    L.append("")

    if best and best["kind"] == "neural":
        run = Run(project.runs_dir / best["id"])
        L += [f"## Best run: {run.id}", ""]
        if run.curves_path.exists():
            L += [f"![curves]({rel(run.curves_path)})", ""]
        if run.diagnosis_path.exists():
            d = read_json(run.diagnosis_path)
            L.append(f"Diagnosis: **{d['verdict']}**")
            for f in d["findings"]:
                L.append(f"- [{f['severity']}] `{f['code']}`: {f['message']}")
            L.append("")
        for split_name in ("val", "test"):
            ev = run.eval_result(split_name)
            if ev and ev.get("confusion_png"):
                L += [
                    f"{split_name} confusion matrix:",
                    "",
                    f"![confusion]({rel(ev['confusion_png'])})",
                    "",
                ]
            if ev and ev.get("top_confusions"):
                L.append(
                    f"Top {split_name} confusions: "
                    + "; ".join(
                        f"{c['true']} → {c['predicted']} ({c['count']})"
                        for c in ev["top_confusions"][:5]
                    )
                )
                L.append("")

    bench_dir = project.state_dir / "benchmark"
    if bench_dir.exists():
        for bp in sorted(bench_dir.glob("*.json")):
            b = read_json(bp)
            c = b["cost_estimate"]
            a = c["assumptions"]
            L += [
                f"## Deployment: {b['run']}",
                "",
                "| | small model | LLM |",
                "|---|---|---|",
                f"| what | `{b['model']}` ({b['model_size_mb']} MB, ONNX on CPU) | {a['llm_name']} |",
                f"| latency p50 | {b['latency_ms_batch1']['p50']:.2f} ms | ~{a['llm_latency_s'] * 1000:.0f} ms (assumed) |",
                f"| cost / 1M predictions | ${c['small_model_usd_per_1m_predictions']:,.4f} | ${c['llm_usd_per_1m_predictions']:,.2f} |",
                "",
                f"Estimate only. Assumptions: {a['instance_name']} at ${a['instance_usd_per_hour']}/h, "
                f"{a['instance_utilisation']:.0%} utilisation, CPU slowdown x{a['cpu_slowdown_factor']}; LLM "
                f"{a['llm_input_tokens_per_call']} input + {a['llm_output_tokens_per_call']} output tokens at "
                f"${a['llm_usd_per_mtok_input']}/${a['llm_usd_per_mtok_output']} per MTok. {a['pricing_note']}",
                "",
            ]

    L += ["## Lifecycle status", ""]
    for s in st["stages"]:
        L.append(f"- [{'x' if s['done'] else ' '}] {s['stage']}: {s['detail']}")
    L.append("")
    path = rd / "REPORT.md"
    L = [line.replace(str(project.root) + "/", "") for line in L]  # project-relative paths
    L = [
        line.replace("- [ ] report: reports/REPORT.md", "- [x] report: reports/REPORT.md")
        for line in L
    ]
    path.write_text("\n".join(L))
    return path
