"""Does a 5-fold bag beat the single run it copies, on sealed test sets? (dev experiment)

Uses the suite projects' ``v-periodic`` runs (bench/variants.py) on datasets up to --max-rows.
Writes scratch/bagtest.jsonl; prints mean relative test improvement of bag vs single.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path.home() / ".cache" / "trainwright-suite"
OUT = ROOT / "scratch" / "bagtest.jsonl"
LOWER = {"rmse", "mae", "log_loss"}


def tw(args: list[str], cwd: Path) -> dict[str, Any]:
    exe = str(ROOT / ".venv" / "bin" / "trainwright")
    env = dict(os.environ, OMP_NUM_THREADS="2")
    p = subprocess.run([exe, "--json", *args], cwd=cwd, capture_output=True, text=True, env=env)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": (p.stderr or p.stdout)[-400:]}


def one(ds: str) -> dict[str, Any]:
    proj = WORK / ds
    runs = tw(["runs"], proj)["result"]
    metric = tw(["status"], proj)["result"]["metric"].split()[0]
    single = next((r for r in runs if r["id"].endswith("v-periodic")), None)
    gbm = next((r for r in runs if r.get("model") == "baseline:hist_gbm"), None)
    if single is None:
        return {"dataset": ds, "error": "no v-periodic run"}
    b = tw(["bag", single["id"]], proj)
    if not b.get("ok"):
        return {"dataset": ds, "error": str(b.get("error"))[:300]}
    bag_id = b["result"]["run"]
    res: dict[str, Any] = {"dataset": ds, "metric": metric}
    for label, rid in [("single", single["id"]), ("bag", bag_id), ("gbm", gbm and gbm["id"])]:
        if rid is None:
            continue
        ev = tw(["evaluate", rid, "--split", "test", "--reason", "bag dev test", "--force"], proj)
        if ev.get("ok"):
            res[label] = ev["result"]["metrics"].get(metric)
    res["bag_val"] = b["result"]["summary"]["val"].get(metric)
    res["single_val"] = single.get(f"val_{metric}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cells", type=int, default=150_000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    meta = [json.loads(line) for line in (ROOT / "scratch/suite/results.jsonl").open()]
    done = {json.loads(line)["dataset"] for line in OUT.open()} if OUT.exists() else set()
    todo = [
        m["dataset"]
        for m in meta
        if m["status"] == "ok" and m["rows"] * m["cols"] <= a.max_cells and m["dataset"] not in done
    ]
    print(f"{len(todo)} datasets", flush=True)

    def task(ds: str) -> None:
        r = one(ds)
        with OUT.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(r, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(task, todo))
    rows = [json.loads(line) for line in OUT.open()]
    rel = []
    for r in rows:
        if r.get("bag") is None or r.get("single") is None:
            continue
        a_, b_ = r["bag"], r["single"]
        rel.append((b_ - a_) / abs(b_) if r["metric"] in LOWER else (a_ - b_) / abs(b_))
    if rel:
        print(
            f"bag vs single on test over {len(rel)} datasets: mean {np.mean(rel):+.2%}, "
            f"median {np.median(rel):+.2%}, bag better on {sum(x > 0 for x in rel)}"
        )


if __name__ == "__main__":
    main()
