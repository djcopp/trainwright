"""Download PolyAI/banking77 (CC-BY-4.0) as one CSV with `message` and `intent` columns.

uv run python examples/banking77/prepare.py data/banking77.csv
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default"


def main(out: Path) -> None:
    info = json.load(
        urllib.request.urlopen(
            "https://datasets-server.huggingface.co/info?dataset=PolyAI/banking77"
        )
    )
    names = info["dataset_info"]["default"]["features"]["label"]["names"]
    frames = [pd.read_parquet(f"{BASE}/{split}/0000.parquet") for split in ("train", "test")]
    df = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame({"message": df["text"], "intent": [names[i] for i in df["label"]]})
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} messages, {df.intent.nunique()} intents -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/banking77.csv"))
