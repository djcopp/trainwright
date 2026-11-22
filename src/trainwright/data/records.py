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
    corrections: dict[str, Any] | None = None
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
        return table_to_dict(self.table)


def table_to_dict(df: pd.DataFrame) -> dict[str, Any]:
    """DataFrame -> column -> numpy array (numerics as float64 with NaN, others as objects/None)."""
    out: dict[str, Any] = {}
    for c in df.columns:
        s = df[c]
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
    if cfg.task == "segmentation":
        if not cfg.data.mask_path:
            raise ValueError("Segmentation needs data.mask_path: a folder of PNG masks.")
        m = Path(cfg.data.mask_path)
        recs = _load_segmentation(path, m if m.is_absolute() else (project_root / m).resolve())
    elif cfg.modality == "image":
        recs = _load_images(path, cfg)
    else:
        recs = _load_table_like(path, cfg, project_root / ".trainwright")
    if cfg.data.corrections:
        recs = _apply_corrections(recs, project_root, cfg.data.corrections)
    return recs


def _apply_corrections(recs: Records, root: Path, rel: str) -> Records:
    from .._utils import sha256_file
    from ..label import load_corrections

    cpath = Path(rel) if Path(rel).is_absolute() else root / rel
    fixes = load_corrections(cpath)
    y = recs.y.copy().astype(object)
    applied = 0
    for rid, lab in fixes.items():
        k = recs._pos.get(rid)
        if k is not None and str(y[k]) != lab:
            y[k] = lab if recs.task == "classification" else float(lab)
            applied += 1
    recs.y = y.astype(str) if recs.task == "classification" else y.astype(float)
    recs.corrections = {
        "file": str(rel),
        "sha256": sha256_file(cpath)[:16],
        "applied": applied,
        "unknown_ids": sum(1 for r in fixes if r not in recs._pos),
    }
    return recs


def text_feature_states(
    state_dir: Path, cfg: ProjectConfig, df: pd.DataFrame | None, data_sha: str
) -> dict[str, Any]:
    """Fitted TF-IDF+SVD states for data.text_features, cached per data fingerprint."""
    import json

    from ..preprocessing import fit_text_svd

    cols = list(cfg.data.text_features)
    if not cols:
        return {}
    cache = state_dir / "text_features.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("data_sha") == data_sha and sorted(blob.get("states", {})) == sorted(cols):
            return blob["states"]
    if df is None:
        raise FileNotFoundError("text feature state missing; reload the data to refit it")
    states = {
        c: fit_text_svd(df[c].fillna("").astype(str).tolist()) for c in cols if c in df.columns
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"data_sha": data_sha, "states": states}))
    return states


def _load_table_like(path: Path, cfg: ProjectConfig, state_dir: Path | None = None) -> Records:
    df = read_table(path)
    if cfg.data.text_features and cfg.modality == "tabular" and state_dir is not None:
        from ..preprocessing import expand_text_features

        states = text_feature_states(state_dir, cfg, df, sha256_file(path))
        cols = {c: df[c].tolist() for c in states}
        derived = {k: v for k, v in expand_text_features(cols, states).items() if k not in cols}
        df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
    if cfg.data.date_columns and cfg.modality == "tabular":
        from ..preprocessing import expand_dates

        cols = {c: df[c].tolist() for c in cfg.data.date_columns if c in df.columns}
        derived = {
            k: v for k, v in expand_dates(cols, cfg.data.date_columns).items() if k not in cols
        }
        df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
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


def _load_segmentation(path: Path, mask_dir: Path) -> Records:
    """Images under ``path`` (recursive) paired with ``mask_dir``/<same stem>.png."""
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask folder {mask_dir} does not exist (data.mask_path)")
    masks = {
        f.relative_to(mask_dir).with_suffix("").as_posix(): f
        for f in mask_dir.rglob("*.png")
        if not f.name.startswith(".")
    }
    paths, ys, missing = [], [], []
    for f in sorted(path.rglob("*")):
        if not (f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS) or f.name.startswith("."):
            continue
        key = f.relative_to(path).with_suffix("").as_posix()
        m = masks.get(key) or masks.get(f.stem)
        if m is None:
            missing.append(key)
            continue
        paths.append(f)
        ys.append(str(m))
    if not paths:
        raise ValueError(f"No image in {path} has a mask with the same stem in {mask_dir}.")
    rel = [str(p.relative_to(path)) for p in paths]
    fingerprint = sha256_strings(
        f"{r}:{p.stat().st_size}:{Path(m).stat().st_size}" for r, p, m in zip(rel, paths, ys)
    )
    recs = Records(
        "image", "segmentation", rel, np.array(ys, dtype=object), fingerprint, paths=paths
    )
    recs.dropped_missing_target = len(missing)
    return recs


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
