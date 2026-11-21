"""Download the Toyota Corolla engine-bay car-parts photos (Apache-2.0) from Hugging Face.

    uv run python examples/car_parts/download.py data/car_parts

Creates <out>/<Part Name>/<photo>.jpg (8 classes, ~1,350 photos, ~6 GB of full-size JPEGs).
Source: https://huggingface.co/datasets/stevenalbert10/Toyota-Corolla-Car-Parts
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "stevenalbert10/Toyota-Corolla-Car-Parts"


def main(out: Path) -> None:
    meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}"))
    files = [
        s["rfilename"]
        for s in meta["siblings"]
        if s["rfilename"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    def get(rel: str) -> None:
        dest = out / rel
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{urllib.parse.quote(rel)}"
        urllib.request.urlretrieve(url, dest)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(get, files))
    print(f"{len(files)} images in {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/car_parts"))
