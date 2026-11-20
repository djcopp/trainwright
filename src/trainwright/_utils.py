"""Small shared helpers: JSON IO, hashing, seeding, devices, environment capture."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:max_len].strip("-") or "run"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def clean_floats(obj: Any) -> Any:
    """Replace NaN/inf with None so output is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {k: clean_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_floats(v) for v in obj]
    return obj


def to_json(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(clean_floats(obj), indent=indent, default=_json_default)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(to_json(obj))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(to_json(obj, indent=None) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_strings(items: Iterable[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode())
        h.update(b"\0")
    return h.hexdigest()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (all devices)."""
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_info(cwd: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
            )
            return out.stdout.strip()
        except Exception:
            return None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None}
    dirty = run("status", "--porcelain")
    return {"sha": sha, "dirty": bool(dirty)}


def environment_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    try:
        import sklearn

        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    return info


def human_count(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


def parse_duration(text: str | float | int | None) -> float | None:
    """Parse '90s', '10m', '1.5h' or a number of minutes into seconds."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text) * 60
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([smh]?)\s*", str(text))
    if not m:
        raise ValueError(f"Cannot parse duration {text!r}; use e.g. '90s', '10m', '1h'")
    value, unit = float(m.group(1)), m.group(2) or "m"
    return value * {"s": 1, "m": 60, "h": 3600}[unit]
