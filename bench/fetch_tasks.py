"""Download the round-8+ agent-trial datasets (public; OpenML + Hugging Face parquet).

    uv run python bench/fetch_tasks.py            # -> scratch/datasets/<name>/

Chosen to span modality, size and difficulty:
  beans          image  ~1.3k   3 classes, leaf disease photos (small, real-world)
  eurosat        image  ~27k   10 classes, 64px satellite tiles (medium, not natural photos)
  tweet_irony    text   ~4.6k  binary, hard and noisy
  app_reviews    text   20k    star rating 1-5 as *regression* from review text
  mercedes       table  4.2k   regression, 377 columns, high-cardinality categoricals, noisy
  bike_sharing   table  17k    regression, hourly demand (temporal structure)
  miniboone      table  130k   binary physics signal vs background (large)
  telco_churn    table  7k     binary, imbalanced, mixed types, dirty numeric column
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "datasets"


def _get(url: str, timeout: float = 300) -> bytes:
    last: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(3)
    raise RuntimeError(f"{url}: {last}")


def _hf(ds: str, config: str, split: str) -> pd.DataFrame:
    urls = json.loads(_get(f"https://huggingface.co/api/datasets/{ds}/parquet/{config}/{split}"))
    return pd.concat([pd.read_parquet(io.BytesIO(_get(u))) for u in urls], ignore_index=True)


def _openml(did: int) -> tuple[pd.DataFrame, str]:
    desc = json.loads(_get(f"https://www.openml.org/api/v1/json/data/{did}"))[
        "data_set_description"
    ]
    df = pd.read_parquet(io.BytesIO(_get(desc["parquet_url"])))
    return df, desc.get("default_target_attribute")


def _images(df: pd.DataFrame, img_col: str, label_col: str, names: list[str], dest: Path) -> None:
    for i, row in df.iterrows():
        img = row[img_col]
        cls = names[int(row[label_col])]
        d = dest / "images" / cls
        d.mkdir(parents=True, exist_ok=True)
        ext = ".png" if img["bytes"][:4] == b"\x89PNG" else ".jpg"
        (d / f"{i:05d}{ext}").write_bytes(img["bytes"])


def fetch(name: str) -> None:
    dest = OUT / name
    if (dest / ".done").exists():
        print(f"{name}: cached")
        return
    dest.mkdir(parents=True, exist_ok=True)
    if name == "beans":
        df = pd.concat(
            [_hf("AI-Lab-Makerere/beans", "default", s) for s in ("train", "validation", "test")],
            ignore_index=True,
        )
        _images(df, "image", "labels", ["angular_leaf_spot", "bean_rust", "healthy"], dest)
    elif name == "eurosat":
        df = pd.concat(
            [_hf("blanchon/EuroSAT_RGB", "default", s) for s in ("train", "validation", "test")],
            ignore_index=True,
        )
        names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture",
                 "PermanentCrop", "Residential", "River", "SeaLake"]  # fmt: skip
        _images(df, "image", "label", names, dest)
    elif name == "tweet_irony":
        df = pd.concat(
            [_hf("cardiffnlp/tweet_eval", "irony", s) for s in ("train", "validation", "test")],
            ignore_index=True,
        )
        df = df.rename(columns={"text": "tweet"})
        df["irony"] = df.pop("label").map({0: "not_ironic", 1: "ironic"})
        df.to_csv(dest / "tweets.csv", index=False)
    elif name == "app_reviews":
        df = _hf("sealuzh/app_reviews", "default", "train")
        df = df[df["review"].str.len() > 3].sample(n=20000, random_state=0)
        df[["package_name", "review", "star"]].to_csv(dest / "reviews.csv", index=False)
    elif name in ("mercedes", "bike_sharing", "miniboone", "telco_churn"):
        did = {"mercedes": 42570, "bike_sharing": 42712, "miniboone": 41150, "telco_churn": 42178}[
            name
        ]
        df, target = _openml(did)
        df.columns = [str(c).replace(" ", "_") for c in df.columns]
        (dest / "target.txt").write_text(str(target).replace(" ", "_"))
        df.to_csv(dest / f"{name}.csv", index=False)
    else:
        raise KeyError(name)
    (dest / ".done").write_text("ok")
    print(f"{name}: ok")


if __name__ == "__main__":
    names = sys.argv[1:] or ["beans", "eurosat", "tweet_irony", "app_reviews", "mercedes", "bike_sharing", "miniboone", "telco_churn"]  # fmt: skip
    for n in names:
        try:
            fetch(n)
        except Exception as e:
            print(f"{n}: FAILED {e}")
