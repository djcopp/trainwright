"""Compare recipe variants across many suite datasets, on **val only** (test stays sealed).

Each variant is a named list of ``-o key=value`` overrides. Variants train inside the existing
suite projects (``bench/suite.py run`` must have run first), so splits and baselines are shared.
The report scores each variant against early-stopped HistGradientBoosting on the same val split.

    uv run python bench/variants.py run --variant base= --variant steps=train.min_steps=2000 \
        --datasets ctr23-44960,cc18-31 --workers 4
    uv run python bench/variants.py report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path.home() / ".cache" / "trainwright-suite"
OUT = ROOT / "scratch" / "variants.jsonl"
LOWER = {"rmse", "mae", "log_loss"}


def _tw(args: list[str], cwd: Path, timeout: float = 3600) -> dict[str, Any]:
    tw = os.environ.get("TW_BIN", str(ROOT / ".venv" / "bin" / "trainwright"))
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    p = subprocess.run(
        [tw, "--json", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env
    )
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": (p.stderr or p.stdout)[-600:]}


def _val(runs: list[dict[str, Any]], key: str, pred: Any) -> float | None:
    r = next((x for x in runs if pred(x)), None)
    return None if r is None else r.get(key)


def run_one(ds: str, name: str, overrides: list[str], budget: str) -> dict[str, Any]:
    proj = WORK / ds
    t0 = time.time()
    args = ["train", "--name", f"v-{name}", "-o", "train.device=cpu"]
    args += ["-o", f"train.time_budget={budget}"]
    for o in overrides:
        args += ["-o", o]
    r = _tw(args, proj)
    res: dict[str, Any] = {"dataset": ds, "variant": name, "overrides": overrides}
    if not r.get("ok"):
        res["error"] = str(r.get("error"))[:300]
        return res
    metric = _tw(["status"], proj)["result"]["metric"].split()[0]
    runs = _tw(["runs"], proj)["result"]
    key = f"val_{metric}"
    rid = r["result"].get("run") or r["result"].get("id")
    res.update(
        metric=metric,
        val=_val(runs, key, lambda x: x["id"] == rid),
        gbm_val=_val(runs, key, lambda x: x.get("model") == "baseline:hist_gbm"),
        seconds=round(time.time() - t0, 1),
    )
    return res


def run(variants: dict[str, list[str]], datasets: list[str], workers: int, budget: str) -> None:
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            d = json.loads(line)
            done.add((d["dataset"], d["variant"]))
    jobs = [(d, v) for d in datasets for v in variants if (d, v) not in done]
    print(f"{len(jobs)} jobs", flush=True)

    def task(job: tuple[str, str]) -> None:
        d, v = job
        r = run_one(d, v, variants[v], budget)
        with OUT.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(
            f"{d:<14} {v:<12} val={r.get('val')} gbm={r.get('gbm_val')} {r.get('error', '')[:80]}",
            flush=True,
        )

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(task, jobs))


def _rel(m: str, a: float, b: float) -> float:
    return (b - a) / abs(b) if m in LOWER else (a - b) / abs(b)


def report(base: str | None = None) -> None:
    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        if r.get("val") is not None and r.get("gbm_val") is not None:
            by.setdefault(r["variant"], {})[r["dataset"]] = r
    common = set.intersection(*(set(v) for v in by.values())) if by else set()
    print(f"{len(common)} datasets scored by every variant (val only)\n")
    print(f"{'variant':<14} {'vs gbm mean':>12} {'median':>8} {'beats gbm':>10}", end="")
    print(f" {'vs ' + base:>14} {'wins':>6}" if base else "")
    for v, d in sorted(by.items()):
        rel = [_rel(d[k]["metric"], d[k]["val"], d[k]["gbm_val"]) for k in common]
        wins = sum(x > 0 for x in rel)
        line = f"{v:<14} {np.mean(rel):>+12.2%} {np.median(rel):>+8.2%} {wins:>6}/{len(rel)}"
        if base and base in by:
            rb = [_rel(d[k]["metric"], d[k]["val"], by[base][k]["val"]) for k in common]
            line += f" {np.mean(rb):>+14.2%} {sum(x > 0 for x in rb):>3}/{len(rb)}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument(
        "--variant", action="append", required=True, help="name=ov1,ov2 (name= for none)"
    )
    r.add_argument("--datasets", required=True, help="comma list, or @file")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--budget", default="4m")
    s = sub.add_parser("report")
    s.add_argument("--base", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        variants = {}
        for spec in a.variant:
            name, _, ov = spec.partition("=")
            variants[name] = [o for o in ov.split(",") if o]
        ds = a.datasets
        names = Path(ds[1:]).read_text().split() if ds.startswith("@") else ds.split(",")
        run(variants, names, a.workers, a.budget)
    else:
        report(a.base)


if __name__ == "__main__":
    main()
