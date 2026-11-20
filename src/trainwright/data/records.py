"""Load the raw dataset described by a project into a modality-agnostic ``Records`` object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._utils import sha256_file, sha256_strings
from ..config import ProjectConfig
from ..preprocessing import IMAGE_EXTENSIONS


@dataclass
class Records:
    modality: str
    task: str
    ids: list[str]
    y: np.ndarray  # str labels (classification) or float (regression)
    fingerprint: str
    table: pd.DataFrame | None = None  # tabular features (target removed)
    texts: list[str] | None = None
    paths: list[Path] | None = None
    groups: np.ndarray | None = None
    times: np.ndarray | None = None
    dropped_missing_target: int = 0
    _pos: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._pos = {i: k for k, i in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def positions(self, ids: list[str]) -> np.ndarray:
        try:
            return np.fromiter((self._pos[i] for i in ids), dtype=np.int64, count=len(ids))
        except KeyError as e:
            raise KeyError(
                f"Split references id {e} that is not in the data. The data changed after the split "
                "was frozen; re-run `trainwright split --force` (this invalidates earlier comparisons)."
            ) from None

    def subset(self, ids: list[str]) -> Records:
        pos = self.positions(ids)
        return Records(
            modality=self.modality,
            task=self.task,
            ids=[self.ids[p] for p in pos],
            y=self.y[pos],
            fingerprint=self.fingerprint,
            table=self.table.iloc[pos].reset_index(drop=True) if self.table is not None else None,
            texts=[self.texts[p] for p in pos] if self.texts is not None else None,
            paths=[self.paths[p] for p in pos] if self.paths is not None else None,
            groups=self.groups[pos] if self.groups is not None else None,
            times=self.times[pos] if self.times is not None else None,
        )

    def table_dict(self) -> dict[str, Any]:
        """Column -> numpy array, the input format of TabularPreprocessor."""
        assert self.table is not None
        out: dict[str, Any] = {}
        for c in self.table.columns:
            s = self.table[c]
            if pd.api.types.is_bool_dtype(s):
                out[c] = s.astype("float64").to_numpy()
            elif pd.api.types.is_numeric_dtype(s):
                out[c] = s.to_numpy(dtype=np.float64, na_value=np.nan)
            elif pd.api.types.is_datetime64_any_dtype(s):
                vals = (
                    s.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
                    if s.dt.tz
                    else s.to_numpy(dtype="datetime64[ns]")
                )
                secs = vals.astype("int64").astype(np.float64) / 1e9
                secs[np.isnat(vals)] = np.nan
                out[c] = secs
            else:
                out[c] = s.astype(object).where(s.notna(), None).to_numpy()
        return out


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _clean_target(y: pd.Series, task: str) -> tuple[pd.Series, np.ndarray]:
    mask = y.notna().to_numpy()
    y = y[mask]
    if task == "classification":
        # 1.0 / 1 -> "1" consistently
        if pd.api.types.is_float_dtype(y) and np.all(np.mod(y.to_numpy(), 1) == 0):
            y = y.astype("int64")
        return y.astype(str), mask
    return pd.to_numeric(y, errors="coerce").astype(float), mask


def load_records(project_root: Path, cfg: ProjectConfig) -> Records:
    p = Path(cfg.data.path)
    path = p if p.is_absolute() else (project_root / p).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data path {path} does not exist (data.path in trainwright.yaml)")
    if cfg.modality == "image":
        return _load_images(path, cfg)
    return _load_table_like(path, cfg)


def _load_table_like(path: Path, cfg: ProjectConfig) -> Records:
    df = read_table(path)
    target = cfg.data.target
    if not target or target not in df.columns:
        raise ValueError(
            f"Target column {target!r} not in {path.name}. Columns: {list(df.columns)[:30]}"
        )
    ids_all = np.array([str(i) for i in range(len(df))])
    y, mask = _clean_target(df[target], cfg.task)
    dropped = int((~mask).sum())
    df = df[mask].reset_index(drop=True)
    ids = ids_all[mask].tolist()
    groups = df[cfg.data.group_column].astype(str).to_numpy() if cfg.data.group_column else None
    times = None
    if cfg.data.time_column:
        times = pd.to_datetime(df[cfg.data.time_column], errors="coerce").to_numpy()
    fingerprint = sha256_file(path)
    if cfg.modality == "text":
        col = cfg.data.text_column
        if not col or col not in df.columns:
            raise ValueError(
                f"data.text_column {col!r} not found. Columns: {list(df.columns)[:30]}"
            )
        texts = df[col].fillna("").astype(str).tolist()
        return Records(
            "text",
            cfg.task,
            ids,
            y.to_numpy(),
            fingerprint,
            texts=texts,
            groups=groups,
            times=times,
            dropped_missing_target=dropped,
        )
    feature_cols = [c for c in df.columns if c != target]
    table = df[feature_cols]
    return Records(
        "tabular",
        cfg.task,
        ids,
        y.to_numpy(),
        fingerprint,
        table=table,
        groups=groups,
        times=times,
        dropped_missing_target=dropped,
    )


def _load_images(path: Path, cfg: ProjectConfig) -> Records:
    if path.is_dir():
        classes = sorted([d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")])
        if not classes:
            raise ValueError(
                f"{path} has no class sub-folders. Expected <data>/<class_name>/<image files> "
                "or a CSV manifest with data.image_column and data.target."
            )
        paths, labels = [], []
        for cdir in classes:
            for f in sorted(cdir.rglob("*")):
                if (
                    f.is_file()
                    and f.suffix.lower() in IMAGE_EXTENSIONS
                    and not f.name.startswith(".")
                ):
                    paths.append(f)
                    labels.append(cdir.name)
        rel = [str(p.relative_to(path)) for p in paths]
        fingerprint = sha256_strings(f"{r}:{p.stat().st_size}" for r, p in zip(rel, paths))
        groups = None
        return Records(
            "image",
            cfg.task,
            rel,
            np.array(labels, dtype=object).astype(str),
            fingerprint,
            paths=paths,
            groups=groups,
        )
    # CSV manifest
    df = read_table(path)
    col = cfg.data.image_column
    if not col or col not in df.columns:
        raise ValueError(
            f"data.image_column {col!r} not found in manifest. Columns: {list(df.columns)}"
        )
    if not cfg.data.target or cfg.data.target not in df.columns:
        raise ValueError(f"Target column {cfg.data.target!r} not in manifest")
    y, mask = _clean_target(df[cfg.data.target], cfg.task)
    df = df[mask].reset_index(drop=True)
    base = path.parent
    paths = [
        (base / str(p)).resolve() if not Path(str(p)).is_absolute() else Path(str(p))
        for p in df[col]
    ]
    rel = [str(p) for p in df[col].astype(str)]
    groups = df[cfg.data.group_column].astype(str).to_numpy() if cfg.data.group_column else None
    fingerprint = sha256_file(path)
    return Records(
        "image",
        cfg.task,
        rel,
        y.to_numpy(),
        fingerprint,
        paths=paths,
        groups=groups,
        dropped_missing_target=int((~mask).sum()),
    )
