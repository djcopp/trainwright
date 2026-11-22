"""Build the hard agent-trial tasks: problems where process matters and the ceiling is far.

    uv run --with openpyxl python bench/fetch_hard_tasks.py      # -> scratch/tasks/<name>/

  retail_repeat  relational + time  raw transaction log (UCI Online Retail II, CC BY 4.0). Which
                                    customers buy again in the 90 days after the cutoff? The
                                    agent must build per-customer features from the log.
  airbnb_price   messy table        London listings (Inside Airbnb, CC BY 4.0): ~70 columns,
                                    prices as "$1,234.00", amenity lists, free text, dates.
  ledgar         hard text          contract clauses, 100 provision types (LEDGAR / LexGLUE,
                                    CC BY-SA 4.0), subsampled to 30k.

Each task directory gets ``data/`` (what the agent sees), ``hidden_labels.csv`` and
``task.json``, the same layout ``agent_trials.py prepare`` writes.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_tasks import _get

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "scratch" / "tasks"
RAW = ROOT / "scratch" / "datasets" / "hard"
SEED = 0


def _task_dir(name: str) -> Path:
    d = TASKS / name
    if d.exists():
        shutil.rmtree(d)
    (d / "data").mkdir(parents=True)
    return d


def _write(d: Path, spec: dict, hidden: pd.Series) -> None:
    hidden.rename("label").to_frame().to_csv(d / "hidden_labels.csv", index=False)
    (d / "task.json").write_text(json.dumps({"name": d.name, **spec}, indent=2))
    print(f"prepared {d.name}: {len(hidden)} hidden rows")


def _cached(name: str, url: str) -> bytes:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / name
    if not p.exists():
        p.write_bytes(_get(url, timeout=900))
    return p.read_bytes()


def retail_repeat() -> None:
    raw = _cached(
        "online_retail_ii.zip", "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    )
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None)
    tx = pd.concat(sheets.values(), ignore_index=True)
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"])
    cutoff = pd.Timestamp("2011-09-01")
    horizon = cutoff + pd.Timedelta(days=90)
    before = tx[tx["InvoiceDate"] < cutoff]
    bought = before[~before["Invoice"].astype(str).str.startswith("C")]
    customers = sorted(bought["Customer ID"].dropna().astype(int).unique())
    after = tx[(tx["InvoiceDate"] >= cutoff) & (tx["InvoiceDate"] < horizon)]
    after = after[~after["Invoice"].astype(str).str.startswith("C")]
    repeat = set(after["Customer ID"].dropna().astype(int))
    lab = pd.DataFrame(
        {
            "customer_id": customers,
            "buys_next_90d": ["yes" if c in repeat else "no" for c in customers],
        }
    )
    lab = lab.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    hide = (
        lab.groupby("buys_next_90d", group_keys=False)
        .apply(lambda g: g.sample(frac=0.15, random_state=SEED))
        .index
    )
    d = _task_dir("retail_repeat")
    before.to_csv(d / "data" / "transactions.csv", index=False)
    lab.drop(index=hide).to_csv(d / "data" / "customers.csv", index=False)
    lab.loc[hide, ["customer_id"]].to_csv(d / "data" / "holdout.csv", index=False)
    spec = {
        "kind": "tabular",
        "file": "customers.csv",
        "target": "buys_next_90d",
        "task": "classification",
        "prompt": "We run an online gift shop. transactions.csv is our raw sales log up to 1 Sep 2011 "
        "(one row per invoice line; cancelled invoices start with 'C'). customers.csv says, for "
        "some customers, whether they bought again in the 90 days after that date "
        "(`buys_next_90d`, yes/no). Marketing wants to target customers who are about to lapse. "
        "Build a model that predicts it and tell me how good it is. Finally write predictions for "
        "the customers in holdout.csv to predictions.csv: one row per holdout row, same order, "
        "column `prediction` (yes/no).",
    }
    _write(d, spec, lab.loc[hide, "buys_next_90d"].reset_index(drop=True))


def airbnb_price() -> None:
    raw = _cached(
        "airbnb_london_listings.csv.gz",
        "https://data.insideairbnb.com/united-kingdom/england/london/2026-09-17/data/listings.csv.gz",
    )
    df = pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False)
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[$£,]", "", regex=True), errors="coerce"
    )
    df = df[df["price"].between(10, 1000)]
    # derived from the price itself, or scraping metadata a new host wouldn't have
    drop = [
        "estimated_revenue_l365d", "listing_url", "scrape_id", "last_scraped", "source",
        "picture_url", "host_url", "host_thumbnail_url", "host_picture_url",
        "calendar_last_scraped",
    ]  # fmt: skip
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.sample(n=min(40_000, len(df)), random_state=SEED).reset_index(drop=True)
    hide = df.sample(frac=0.15, random_state=SEED).index
    d = _task_dir("airbnb_price")
    df.drop(index=hide).to_csv(d / "data" / "listings.csv", index=False)
    df.loc[hide].drop(columns=["price"]).to_csv(d / "data" / "holdout.csv", index=False)
    spec = {
        "kind": "tabular",
        "file": "listings.csv",
        "target": "price",
        "task": "regression",
        "prompt": "listings.csv has London Airbnb listings with their nightly price (`price`, GBP). "
        "We want to suggest a price to new hosts. Train a model that predicts the nightly price "
        "and tell me how accurate it is. Finally write predictions for holdout.csv (same columns, "
        "no price) to predictions.csv: one row per holdout row, same order, column `prediction`.",
    }
    _write(d, spec, df.loc[hide, "price"].reset_index(drop=True))


def ledgar() -> None:
    info = json.loads(
        _get(
            "https://datasets-server.huggingface.co/info?dataset=coastalcph/lex_glue&config=ledgar"
        )
    )
    names = info["dataset_info"]["features"]["label"]["names"]
    parts = []
    for split in ("train", "validation", "test"):
        urls = json.loads(
            _get(f"https://huggingface.co/api/datasets/coastalcph/lex_glue/parquet/ledgar/{split}")
        )
        parts += [
            pd.read_parquet(io.BytesIO(_cached(f"ledgar_{split}_{i}.parquet", u)))
            for i, u in enumerate(urls)
        ]
    df = pd.concat(parts, ignore_index=True)
    df["label"] = [names[int(i)] for i in df["label"]]
    rng = np.random.default_rng(SEED)
    df = df.iloc[rng.permutation(len(df))[:30_000]].reset_index(drop=True)
    hide = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(frac=0.15, random_state=SEED))
        .index
    )
    d = _task_dir("ledgar")
    df.drop(index=hide).to_csv(d / "data" / "ledgar.csv", index=False)
    df.loc[hide, ["text"]].to_csv(d / "data" / "holdout.csv", index=False)
    spec = {
        "kind": "text",
        "file": "ledgar.csv",
        "target": "label",
        "task": "classification",
        "prompt": "We sort contract clauses into 100 provision types with an LLM call today. "
        "ledgar.csv has clauses (`text`) and their type (`label`). Train a small model that could "
        "replace the LLM call, and tell me how good it is and whether it's worth switching. "
        "Finally write predictions for holdout.csv (column `text`) to predictions.csv: one row "
        "per holdout row, same order, column `prediction`.",
    }
    _write(d, spec, df.loc[hide, "label"].reset_index(drop=True))


if __name__ == "__main__":
    for fn in (retail_repeat, airbnb_price, ledgar):
        if len(sys.argv) < 2 or fn.__name__ in sys.argv[1:]:
            fn()
