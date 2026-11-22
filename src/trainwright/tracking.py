"""File-based experiment tracking. No server, no account: every run is a directory.

::

    runs/007-mlp-wider/
        meta.json        # id, name, hypothesis, kind, status, git, env, data/split fingerprints
        config.yaml      # fully-resolved ExperimentConfig (reproduce with --from-run)
        metrics.jsonl    # one JSON object per epoch
        summary.json     # best epoch + metrics, stop reason, timings, param count
        diagnosis.json   # automatic curve diagnosis
        curves.png       # loss + primary-metric curves (agents can open this image)
        checkpoints/best.pt
        eval/<split>.json
        log.txt
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._utils import (
    append_jsonl,
    environment_info,
    git_info,
    now_iso,
    read_json,
    read_jsonl,
    slugify,
    write_json,
)

_ID_RE = re.compile(r"^(\d{3,})-")


@dataclass
class Run:
    dir: Path

    # ----------------------------------------------------------------- creation
    @classmethod
    def create(
        cls,
        runs_dir: Path,
        name: str,
        *,
        kind: str = "neural",
        hypothesis: str | None = None,
        tags: list[str] | None = None,
        parent: str | None = None,
        extra_meta: dict[str, Any] | None = None,
        project_root: Path | None = None,
    ) -> Run:
        runs_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(50):  # another process may claim the same number: take the next one
            seq = next_sequence(runs_dir)
            run_id = f"{seq:03d}-{slugify(name)}"
            d = runs_dir / run_id
            try:
                d.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:  # pragma: no cover
            raise RuntimeError(f"could not allocate a run directory in {runs_dir}")
        run = cls(d)
        meta: dict[str, Any] = {
            "id": run_id,
            "name": name,
            "kind": kind,
            "hypothesis": hypothesis,
            "tags": tags or [],
            "parent": parent,
            "status": "created",
            "created_at": now_iso(),
            "git": git_info(project_root or runs_dir.parent),
            "env": environment_info(),
        }
        if extra_meta:
            meta.update(extra_meta)
        write_json(run.meta_path, meta)
        return run

    @classmethod
    def open(cls, runs_dir: Path, ref: str) -> Run:
        return cls(resolve_run_dir(runs_dir, ref))

    # ----------------------------------------------------------------- paths
    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def config_path(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def metrics_path(self) -> Path:
        return self.dir / "metrics.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.dir / "summary.json"

    @property
    def diagnosis_path(self) -> Path:
        return self.dir / "diagnosis.json"

    @property
    def curves_path(self) -> Path:
        return self.dir / "curves.png"

    @property
    def checkpoint_dir(self) -> Path:
        return self.dir / "checkpoints"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def eval_dir(self) -> Path:
        return self.dir / "eval"

    @property
    def log_path(self) -> Path:
        return self.dir / "log.txt"

    # ----------------------------------------------------------------- IO
    @property
    def meta(self) -> dict[str, Any]:
        return read_json(self.meta_path)

    def update_meta(self, **kv: Any) -> None:
        m = self.meta
        m.update(kv)
        write_json(self.meta_path, m)

    def set_status(self, status: str, **kv: Any) -> None:
        self.update_meta(status=status, status_at=now_iso(), **kv)

    @property
    def summary(self) -> dict[str, Any]:
        return read_json(self.summary_path) if self.summary_path.exists() else {}

    def write_summary(self, summary: dict[str, Any]) -> None:
        write_json(self.summary_path, summary)

    def log_metrics(self, row: dict[str, Any]) -> None:
        append_jsonl(self.metrics_path, row)

    @property
    def history(self) -> list[dict[str, Any]]:
        return read_jsonl(self.metrics_path)

    def log(self, msg: str) -> None:
        with self.log_path.open("a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    def eval_result(self, split: str) -> dict[str, Any] | None:
        p = self.eval_dir / f"{split}.json"
        return read_json(p) if p.exists() else None

    @contextmanager
    def lifecycle(self) -> Iterator[Run]:
        """Mark the run running/completed/failed/interrupted around a block."""
        self.set_status("running", started_at=now_iso())
        try:
            yield self
        except KeyboardInterrupt:
            self.set_status("interrupted")
            raise
        except Exception as e:
            self.set_status("failed", error=f"{type(e).__name__}: {e}")
            self.log(f"FAILED: {type(e).__name__}: {e}")
            raise
        else:
            self.set_status("completed", finished_at=now_iso())


def next_sequence(runs_dir: Path) -> int:
    best = 0
    if runs_dir.exists():
        for d in runs_dir.iterdir():
            m = _ID_RE.match(d.name)
            if m:
                best = max(best, int(m.group(1)))
    return best + 1


def resolve_run_dir(runs_dir: Path, ref: str) -> Path:
    """Accept full id ('007-mlp'), sequence ('7', '007'), 'latest' or 'best' is resolved elsewhere."""
    if not runs_dir.exists():
        raise FileNotFoundError("No runs yet.")
    dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and _ID_RE.match(d.name)])
    if ref == "latest":
        if not dirs:
            raise FileNotFoundError("No runs yet.")
        return dirs[-1]
    exact = runs_dir / ref
    if exact.is_dir():
        return exact
    if ref.isdigit():
        for d in dirs:
            m = _ID_RE.match(d.name)
            if m and int(m.group(1)) == int(ref):
                return d
    matches = [d for d in dirs if ref in d.name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(f"Run ref {ref!r} is ambiguous: {[d.name for d in matches]}")
    raise FileNotFoundError(f"No run matching {ref!r}. See `trainwright runs`.")


def list_runs(runs_dir: Path) -> list[Run]:
    if not runs_dir.exists():
        return []
    return [Run(d) for d in sorted(runs_dir.iterdir()) if d.is_dir() and (d / "meta.json").exists()]
