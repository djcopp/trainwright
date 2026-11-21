"""Large-scale, LLM-free benchmark of trainwright's automated pipeline on standard OpenML suites.

Suites: OpenML-CC18 (72 classification datasets) and OpenML-CTR23 (35 regression datasets).
For every dataset, trainwright's own lifecycle runs end to end through the CLI in an isolated
project (init -> profile -> split -> baselines -> default model -> greedy ensemble). The winner is
picked on **val only** and scored once on the sealed test split, alongside the strongest
scikit-learn default (early-stopped HistGradientBoosting), which is what a blind prompt usually reaches for.

    uv run python bench/suite.py prepare                 # download (resumable)
    uv run python bench/suite.py run --workers 4         # benchmark (resumable; no LLM, no tokens)
    uv run python bench/suite.py report                  # aggregate table

This measures the framework's modelling quality at scale. The agent A/B trials
(``bench/agent_trials.py``) measure what an agent does with it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUITES = {"cc18": 99, "ctr23": 353}
MAX_CELLS = 6_000_000
MAX_COLS = 1000


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def prepare(out: Path, suites: list[str]) -> None:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    index = []
    for suite in suites:
        ids = _get(f"https://www.openml.org/api/v1/json/study/{SUITES[suite]}")["study"]["data"][
            "data_id"
        ]
        for did in ids:
            d = out / f"{suite}-{did}"
            meta_path = d / "meta.json"
            if meta_path.exists():
                index.append(json.loads(meta_path.read_text()))
                continue
            d.mkdir(exist_ok=True)
            try:
                desc = _get(f"https://www.openml.org/api/v1/json/data/{did}")[
                    "data_set_description"
                ]
                q = {
                    x["name"]: x["value"]
                    for x in _get(f"https://www.openml.org/api/v1/json/data/qualities/{did}")[
                        "data_qualities"
                    ]["quality"]
                    if x.get("value") is not None
                }
                rows, cols = (
                    int(float(q.get("NumberOfInstances", 0))),
                    int(float(q.get("NumberOfFeatures", 0))),
                )
                meta = {
                    "suite": suite,
                    "id": did,
                    "name": desc["name"],
                    "target": desc.get("default_target_attribute"),
                    "task": "classification" if suite == "cc18" else "regression",
                    "rows": rows,
                    "cols": cols,
                    "licence": desc.get("licence"),
                }
                if rows * cols > MAX_CELLS or cols > MAX_COLS:
                    meta["skipped"] = f"too large for the CPU benchmark ({rows}x{cols})"
                else:
                    pq = d / "raw.pq"
                    urllib.request.urlretrieve(desc["parquet_url"], pq)
                    df = pd.read_parquet(pq)
                    df.columns = [str(c).replace(" ", "_") for c in df.columns]
                    meta["target"] = str(meta["target"]).replace(" ", "_")
                    df.to_csv(d / "data.csv", index=False)
                    pq.unlink()
            except Exception as e:
                meta = {
                    "suite": suite,
                    "id": did,
                    "skipped": f"download failed: {type(e).__name__}: {e}",
                }
            meta_path.write_text(json.dumps(meta, indent=2))
            index.append(meta)
            print(f"{suite}-{did}: {meta.get('name')} {meta.get('skipped', 'ok')}", flush=True)
    ok = [m for m in index if not m.get("skipped")]
    print(f"{len(ok)} datasets ready, {len(index) - len(ok)} skipped")


def _tw(args: list[str], cwd: Path, timeout: float) -> dict[str, Any]:
    tw = os.environ.get("TW_BIN", str(ROOT / ".venv" / "bin" / "trainwright"))
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    p = subprocess.run(
        [tw, "--json", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env
    )
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": (p.stderr or p.stdout)[-800:]}


def run_one(data_dir: Path, work: Path, budget: str) -> dict[str, Any]:
    meta = json.loads((data_dir / "meta.json").read_text())
    proj = work / data_dir.name
    proj.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {
        "dataset": data_dir.name,
        "name": meta.get("name"),
        "task": meta.get("task"),
        "rows": meta.get("rows"),
        "cols": meta.get("cols"),
    }
    t0 = time.time()
    try:
        if not (proj / "trainwright.yaml").exists():
            r = _tw(
                [
                    "init",
                    "--data",
                    str(data_dir / "data.csv"),
                    "--target",
                    meta["target"],
                    "--task",
                    meta["task"],
                ],
                proj,
                300,
            )
            if not r.get("ok"):
                raise RuntimeError(f"init: {r.get('error')}")
        for step, args, to in [
            ("profile", ["profile"], 900),
            ("split", ["split"], 900),
            ("baseline", ["baseline", "--only", "dummy,linear,hist_gbm"], 1800),
            (
                "train",
                [
                    "train",
                    "-o",
                    "train.device=cpu",
                    "-o",
                    f"train.time_budget={budget}",
                    "--hypothesis",
                    "autopilot default",
                ],
                3600,
            ),
            ("ensemble", ["ensemble", "--top", "4"], 1800),
        ]:
            r = _tw(args, proj, to)
            if not r.get("ok") and step not in ("ensemble",):
                raise RuntimeError(f"{step}: {r.get('error')}")
        runs = _tw(["runs"], proj, 300)["result"]
        cfg_metric = _tw(["status"], proj, 120)["result"]["metric"].split()[0]
        key = f"val_{cfg_metric}"
        pick = runs[0]
        gbm = next((x for x in runs if x.get("model") == "baseline:hist_gbm"), None)
        mlp = next((x for x in runs if x.get("kind") == "neural"), None)
        res.update(
            metric=cfg_metric, pick=pick["id"], pick_kind=pick["kind"], pick_val=pick.get(key)
        )
        for label, run in [("pick", pick), ("hist_gbm", gbm), ("mlp", mlp)]:
            if run is None:
                continue
            ev = _tw(
                [
                    "evaluate",
                    run["id"],
                    "--split",
                    "test",
                    "--reason",
                    f"suite benchmark: {label}",
                    "--force",
                ],
                proj,
                1800,
            )
            if ev.get("ok"):
                res[f"{label}_test"] = ev["result"]["metrics"].get(cfg_metric)
        res["status"] = "ok"
    except Exception as e:
        res.update(status="failed", error=str(e)[:500])
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run(
    data: Path, work: Path, results: Path, workers: int, budget: str, limit: int | None
) -> None:
    done = set()
    if results.exists():
        done = {
            json.loads(line)["dataset"] for line in results.read_text().splitlines() if line.strip()
        }
    todo = [d for d in sorted(data.iterdir()) if (d / "data.csv").exists() and d.name not in done]
    todo.sort(
        key=lambda d: (
            json.loads((d / "meta.json").read_text()).get("rows", 0)
            * json.loads((d / "meta.json").read_text()).get("cols", 0)
        )
    )  # small first
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} done, {len(todo)} to run with {workers} workers", flush=True)

    def task(d: Path) -> None:
        r = run_one(d, work, budget)
        with results.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(
            f"{r['dataset']:<14} {r.get('name', '')[:28]:<28} {r['status']:<7} pick={r.get('pick_kind')} "
            f"test={r.get('pick_test')} gbm={r.get('hist_gbm_test')} ({r['seconds']:.0f}s)",
            flush=True,
        )

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(task, todo))


def report(results: Path) -> str:
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    ok = [
        r
        for r in rows
        if r.get("status") == "ok"
        and r.get("pick_test") is not None
        and r.get("hist_gbm_test") is not None
    ]
    lines = [f"{len(ok)} datasets scored ({len(rows) - len(ok)} failed or incomplete)", ""]
    lower = {"rmse", "mae", "log_loss"}
    rel, err_red, wins = [], [], 0
    for r in ok:
        a, b = float(r["pick_test"]), float(r["hist_gbm_test"])
        if r["metric"] in lower:
            rel.append((b - a) / abs(b) if b else 0.0)
            wins += a < b
        else:
            rel.append((a - b) / abs(b) if b else 0.0)
            wins += a > b
            if r["metric"] in ("accuracy", "balanced_accuracy", "macro_f1") and b < 1:
                err_red.append(((1 - b) - (1 - a)) / (1 - b))
    ties = sum(1 for x in rel if abs(x) < 1e-9)
    lines.append("trainwright pick vs HistGradientBoosting default, on sealed test sets:")
    lines.append(f"  better on {wins}/{len(ok)} datasets ({ties} exact ties)")
    lines.append(f"  mean relative improvement {np.mean(rel):+.2%}, median {np.median(rel):+.2%}")
    if err_red:
        lines.append(
            f"  classification error reduction: median {np.median(err_red):+.1%} over {len(err_red)} datasets"
        )
    kinds: dict[str, int] = {}
    for r in ok:
        kinds[r["pick_kind"]] = kinds.get(r["pick_kind"], 0) + 1
    lines.append(f"  picked model family: {kinds}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, default=ROOT / "scratch" / "suite" / "data")
    p.add_argument("--suites", nargs="+", default=["cc18", "ctr23"])
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, default=ROOT / "scratch" / "suite" / "data")
    r.add_argument("--work", type=Path, default=Path.home() / ".cache" / "trainwright-suite")
    r.add_argument("--results", type=Path, default=ROOT / "scratch" / "suite" / "results.jsonl")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--budget", default="4m", help="time budget for the default neural run")
    r.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("report")
    s.add_argument("--results", type=Path, default=ROOT / "scratch" / "suite" / "results.jsonl")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.out, a.suites)
    elif a.cmd == "run":
        run(a.data, a.work, a.results, a.workers, a.budget, a.limit)
    else:
        print(report(a.results))


if __name__ == "__main__":
    sys.exit(main())
