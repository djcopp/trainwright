"""Error analysis and fast feature trials: the hypothesis loop a model search can't do.

``error_report`` scores val with a fast proxy (early-stopped HistGBM on the current
``features.py``) or with any run, then reports where the loss concentrates: slices of the data,
systematic bias, confused class pairs and the worst rows. It also writes candidate
``features.py`` files for structure it can test on its own (group statistics, a log target)
and measures each one.

``try_features`` compares a candidate ``features.py`` with the current one using the same proxy
and a paired bootstrap on val, logs the trial and can promote the candidate.

The proxy takes seconds where a neural run takes minutes, so an agent can test several ideas
while ``auto`` runs.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from ._utils import write_json
from .featurehook import FILENAME, FeatureHook
from .metrics import _single_metric, top_confusions
from .pipeline import LifecycleError, get_records, require_split
from .preprocessing import LabelEncoder, TabularPreprocessor
from .project import Project

MAX_TRAIN_ROWS = 50_000
MIN_SLICE = 20


# --------------------------------------------------------------------------- proxy model


class _Data:
    """The frozen train/val split as DataFrames, labels encoded."""

    def __init__(self, project: Project, max_train: int = MAX_TRAIN_ROWS, seed: int = 0):
        cfg = project.config
        if cfg.modality != "tabular":
            raise LifecycleError("The proxy model is tabular-only; pass a run id instead.")
        records = get_records(project)
        split = require_split(project, records)
        tr_ids = list(split["train"])
        if len(tr_ids) > max_train:
            rng = np.random.default_rng(seed)
            tr_ids = [tr_ids[i] for i in np.sort(rng.choice(len(tr_ids), max_train, False))]
        self.train, self.val = records.subset(tr_ids), records.subset(split["val"])
        self.classes = split.get("classes")
        self.k = len(self.classes or [])
        le = LabelEncoder(self.classes) if self.classes else None
        self.y_tr = le.encode(self.train.y.tolist()) if le else self.train.y.astype(float)
        self.y_va = le.encode(self.val.y.tolist()) if le else self.val.y.astype(float)
        self.task, self.metric, self.mode = cfg.task, cfg.metric, cfg.metric_mode
        self.drop = list(cfg.data.drop)
        self.seed = cfg.split.seed


def _proxy(
    d: _Data, source: str | None, seeds: int = 2, log_target: bool = False
) -> dict[str, Any]:
    """Fit HistGBM on train with ``source`` as features.py; -> val outputs + score."""
    from .baselines import _make_tabular, _proba_full, _tabular_frame, _to_str_cats
    from .data.records import table_to_dict

    t0 = time.time()
    hook = FeatureHook(source) if source else None
    assert d.train.table is not None and d.val.table is not None
    if hook is not None:
        hook.fit(d.train.table)
    tr_df = hook.transform(d.train.table) if hook is not None else d.train.table
    pre = TabularPreprocessor(max_cat_cardinality=1000).fit(table_to_dict(tr_df), drop=d.drop)
    cols = pre.numeric_columns + pre.categorical_columns
    Xtr = _to_str_cats(_tabular_frame(d.train, cols, hook), pre.categorical_columns)
    Xva = _to_str_cats(_tabular_frame(d.val, cols, hook), pre.categorical_columns)
    ytr = np.log1p(d.y_tr) if log_target else d.y_tr
    outs = []
    for s in range(seeds):
        model = _make_tabular("hist_gbm", d.task, pre, d.seed + s)
        model.fit(Xtr, ytr)
        if d.task == "classification":
            outs.append(_proba_full(model, Xva, d.k))
        else:
            p = model.predict(Xva)
            outs.append(np.expm1(p) if log_target else p)
    out = np.mean(outs, axis=0)
    return {
        "out": out,
        "score": _single_metric(d.task, d.metric, d.y_va, out, d.k),
        "frame": hook.transform(d.val.table) if hook is not None else d.val.table,
        "n_features": len(cols),
        "seconds": round(time.time() - t0, 1),
    }


def _paired(d: _Data, a: np.ndarray, b: np.ndarray, n_boot: int = 200) -> tuple[float, float]:
    """Improvement of b over a on the primary metric (positive = better) and its bootstrap SE."""
    sign = 1.0 if d.mode == "max" else -1.0
    y = d.y_va
    point = sign * (
        _single_metric(d.task, d.metric, y, b, d.k) - _single_metric(d.task, d.metric, y, a, d.k)
    )
    rng = np.random.default_rng(0)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=min(n, 20_000))
        try:
            da = _single_metric(d.task, d.metric, y[idx], a[idx], d.k)
            db = _single_metric(d.task, d.metric, y[idx], b[idx], d.k)
        except ValueError:
            continue
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(sign * (db - da))
    se = float(np.std(deltas, ddof=1)) if len(deltas) > 10 else float("nan")
    return float(point), se


def _verdict(delta: float, se: float) -> str:
    if not np.isfinite(se) or se == 0:
        return "better" if delta > 0 else "worse" if delta < 0 else "tied"
    z = delta / se
    return "better" if z > 2 else "likely better" if z > 1 else "worse" if z < -2 else "tied"


# --------------------------------------------------------------------------- error analysis


def _row_loss(task: str, y: np.ndarray, out: np.ndarray) -> np.ndarray:
    if task == "classification":
        p = out[np.arange(len(y)), y.astype(int)]
        return -np.log(np.clip(p, 1e-12, 1.0))
    return (out - y) ** 2


def _slices(frame: Any, loss: np.ndarray, resid: np.ndarray | None, task: str, err: np.ndarray):
    """Per-column bins whose mean loss (or bias) differs from the rest beyond noise."""
    import pandas as pd

    n = len(loss)
    mu, sd = float(loss.mean()), float(loss.std()) or 1e-12
    min_n = max(MIN_SLICE, int(0.02 * n))
    hot, bias = [], []
    for col in frame.columns:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
            try:
                b = pd.qcut(s, 5, duplicates="drop")
            except ValueError:
                continue
            labels = b.astype(str).where(s.notna(), "missing").to_numpy()
        else:
            v = s.astype(str).where(s.notna(), "missing")
            top = v.value_counts().index[:8]
            labels = v.where(v.isin(top), "other").to_numpy()
        for lab in pd.unique(labels):
            m = labels == lab
            k = int(m.sum())
            if k < min_n or k > 0.8 * n:
                continue
            ml = float(loss[m].mean())
            z = (ml - mu) / (sd / np.sqrt(k))
            row = {
                "column": str(col),
                "slice": str(lab),
                "rows": k,
                "share_of_rows": round(k / n, 3),
                "share_of_loss": round(float(loss[m].sum() / max(loss.sum(), 1e-12)), 3),
                "loss_ratio": round(ml / max(mu, 1e-12), 2),
                "z": round(float(z), 1),
            }
            if task == "classification":
                row["error_rate"] = round(float(err[m].mean()), 4)
                row["error_rate_elsewhere"] = round(float(err[~m].mean()), 4)
            else:
                row["rmse"] = round(float(np.sqrt(loss[m].mean())), 4)
                row["rmse_elsewhere"] = round(float(np.sqrt(loss[~m].mean())), 4)
            if z > 3:
                hot.append(row)
            if resid is not None:
                r = resid[m]
                bz = float(r.mean() / (resid.std() / np.sqrt(k) + 1e-12))
                if abs(bz) > 4:
                    bias.append({**row, "mean_error": round(float(r.mean()), 4), "z": round(bz, 1)})

    # one entry per column: its strongest slice
    def best(rows: list[dict[str, Any]], key: Any) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for r in sorted(rows, key=key):
            seen.setdefault(r["column"], r)
        return list(seen.values())[:6]

    return (
        best(hot, lambda r: -(r["share_of_loss"] - r["share_of_rows"])),
        best(bias, lambda r: -abs(r["z"])),
    )


def _worst_rows(
    ids: list[str],
    y: np.ndarray,
    out: np.ndarray,
    loss: np.ndarray,
    task: str,
    classes: Any,
    raw: Any,
    texts: Any,
    paths: Any,
    k: int = 10,
) -> list[dict[str, Any]]:
    rows = []
    for i in np.argsort(-loss)[:k]:
        r: dict[str, Any] = {"id": ids[i]}
        if task == "classification":
            pred = int(out[i].argmax())
            r.update(
                label=classes[int(y[i])],
                predicted=classes[pred],
                confidence=round(float(out[i, pred]), 3),
            )
        else:
            r.update(target=round(float(y[i]), 4), predicted=round(float(out[i]), 4))
        if raw is not None:
            r["row"] = {str(c): _jsonable(v) for c, v in raw.iloc[int(i)].items()}
        if texts is not None:
            r["text"] = str(texts[i])[:300]
        if paths is not None:
            r["path"] = str(paths[i])
        rows.append(r)
    return rows


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 6)
    return v if isinstance(v, (int, str, bool)) or v is None else str(v)


# --------------------------------------------------------------------------- candidates

_GROUP_TEMPLATE = '''"""Candidate from `trainwright errors`: per-{key} statistics of the numeric columns.

Rows sharing a value of {key} are alike ({similarity:.2f} chance-adjusted similarity). Adds the
group's row count, the train mean of each numeric column within the group, and each row's
deviation and z-score relative to it. Statistics come from train only, so this helps only when
predictions are for {key} values seen in training.
"""
import numpy as np
import pandas as pd

{base}
KEY = {key!r}
COLS = {cols!r}


def fit(train: pd.DataFrame) -> dict:
    state = {{"base": _base_fit(train)}}
    df = _base_transform(train, state["base"])
    g = df.groupby(df[KEY].astype(str))[COLS]
    state["count"] = df[KEY].astype(str).value_counts().to_dict()
    state["mean"] = {{str(k): v for k, v in g.mean().to_dict(orient="index").items()}}
    state["std"] = {{str(k): v for k, v in g.std().to_dict(orient="index").items()}}
    return state


def transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    df = _base_transform(df, state["base"])
    key = df[KEY].astype(str)
    df[f"{{KEY}}_count"] = key.map(state["count"]).astype(float)
    mean = pd.DataFrame.from_dict(state["mean"], orient="index")
    std = pd.DataFrame.from_dict(state["std"], orient="index")
    for c in COLS:
        m = key.map(mean[c]).astype(float)
        s = key.map(std[c]).astype(float)
        df[f"{{c}}_{{KEY}}_mean"] = m
        df[f"{{c}}_minus_{{KEY}}_mean"] = df[c] - m
        df[f"{{c}}_{{KEY}}_z"] = (df[c] - m) / (s + 1e-9)
    return df
'''

_BASE_NONE = """
def _base_fit(train):
    return {}


def _base_transform(df, state):
    return df.copy()
"""

_BASE_WRAP = """# the current features.py, applied first
_BASE_SOURCE = {source!r}
_ns: dict = {{}}
exec(compile(_BASE_SOURCE, "features.py", "exec"), _ns)


def _base_fit(train):
    return _ns["fit"](train.copy()) if "fit" in _ns else {{}}


def _base_transform(df, state):
    return _ns["transform"](df.copy(), state)
"""


def _base_block(source: str | None) -> str:
    return _BASE_WRAP.format(source=source) if source else _BASE_NONE


def _group_candidates(project: Project, d: _Data, source: str | None) -> list[dict[str, Any]]:
    from .data.profile import _group_scan

    assert d.train.table is not None and d.val.table is not None
    hook = FeatureHook(source).fit(d.train.table) if source else None
    tr = hook.transform(d.train.table) if hook else d.train.table
    scan = _group_scan(d.train.table, d.train.y, d.task)  # keys are raw columns
    out = []
    for g in scan:
        key = g["column"]
        if key == project.config.data.group_column:
            continue  # predictions are for unseen groups: train statistics won't exist for them
        if any(str(c).startswith(f"{key}_") or f"_{key}_" in str(c) for c in tr.columns):
            continue  # features.py already has statistics for this key
        seen = d.val.table[key].astype(str).isin(set(tr[key].astype(str))).mean()
        if seen < 0.5:
            continue
        num = [
            c
            for c in tr.select_dtypes("number").columns
            if c != key and c not in d.drop and tr[c].nunique() > 2 and str(key) not in str(c)
        ][:20]
        if not num:
            continue
        src = _GROUP_TEMPLATE.format(
            key=key, cols=num, similarity=g["feature_similarity"], base=_base_block(source)
        )
        out.append(
            {
                "name": f"group_{_safe(key)}",
                "hypothesis": f"rows are best judged relative to their {key} ({g['n_groups']} "
                f"groups, {g['rows_per_group']} rows each, {seen:.0%} of val groups seen in train)",
                "source": src,
            }
        )
    return out


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))[:40]


def _skew(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    s = y.std()
    return float(((y - y.mean()) ** 3).mean() / s**3) if s > 0 else 0.0


# --------------------------------------------------------------------------- public API


def error_report(
    project: Project, ref: str | None = None, auto_candidates: bool = True, seeds: int = 2
) -> dict[str, Any]:
    """Where the val loss concentrates, plus measured candidate features (tabular)."""
    cfg = project.config
    if cfg.task == "segmentation":
        return _segmentation_report(project, ref or "best")
    src_path = project.root / FILENAME
    source = src_path.read_text() if src_path.exists() else None
    records = get_records(project)
    split = require_split(project, records)
    val = records.subset(split["val"])
    classes: list[str] = list(split.get("classes") or [])
    d: _Data | None = None
    if ref in (None, "proxy"):
        d = _Data(project)
        base = _proxy(d, source, seeds)
        out, y, frame = base["out"], d.y_va, base["frame"]
        source_desc = (
            f"proxy HistGBM on {'features.py' if source else 'raw columns'} ({base['seconds']}s)"
        )
        score = base["score"]
    else:
        from .ensemble import member_outputs
        from .leaderboard import resolve_ref

        run = resolve_ref(project, ref)
        out, y = member_outputs(project, run, "val")
        if cfg.task == "classification" and y.dtype.kind not in "iu":
            y = LabelEncoder(classes).encode(list(y))
        frame = val.table
        if frame is not None and source and run.meta.get("features_sha"):
            hook = FeatureHook(source).fit(records.subset(split["train"]).table)
            frame = hook.transform(frame)
        source_desc = f"run {run.id}"
        k = len(classes or [])
        score = _single_metric(cfg.task, cfg.metric, y, out, k)
    y = np.asarray(y)
    loss = _row_loss(cfg.task, y, out)
    err = (out.argmax(1) != y).astype(float) if cfg.task == "classification" else np.sqrt(loss)
    resid = (out - y) if cfg.task == "regression" else None

    res: dict[str, Any] = {
        "source": source_desc,
        "metric": cfg.metric,
        "score": round(float(score), 6),
        "n_val": len(y),
    }
    if frame is not None:
        hot, bias = _slices(frame.reset_index(drop=True), loss, resid, cfg.task, err)
        res["hot_slices"] = hot
        res["biased_slices"] = bias
    if cfg.task == "classification":
        from .metrics import confusion

        cm = confusion(y.astype(int), out.argmax(1), len(classes))
        res["confusions"] = top_confusions(cm, list(classes), k=6)
    res["worst_rows"] = _worst_rows(
        val.ids, y, out, loss, cfg.task, classes, frame, val.texts, val.paths
    )

    # per-row table for the agent's own pandas analysis
    import pandas as pd

    tab = pd.DataFrame({"id": val.ids, "loss": loss})
    if cfg.task == "classification":
        tab["label"] = [classes[int(i)] for i in y]
        tab["predicted"] = [classes[int(i)] for i in out.argmax(1)]
        tab["confidence"] = out.max(1)
    else:
        tab["target"], tab["predicted"], tab["error"] = y, out, out - y
    if frame is not None:
        tab = pd.concat([tab, frame.reset_index(drop=True)], axis=1)
    elif val.texts is not None:
        tab["text"] = val.texts
    elif val.paths is not None:
        tab["path"] = [str(p) for p in val.paths]
    rows_path = project.state_dir / "val_errors.csv"
    tab.sort_values("loss", ascending=False).to_csv(rows_path, index=False)
    res["rows_csv"] = str(rows_path)

    # candidates the tool can test by itself
    cands: list[dict[str, Any]] = []
    if auto_candidates and cfg.modality == "tabular":
        d = d or _Data(project)
        base_out = _proxy(d, source, seeds) if ref not in (None, "proxy") else base
        cdir = project.state_dir / "candidates"
        cdir.mkdir(parents=True, exist_ok=True)
        for c in _group_candidates(project, d, source):
            path = cdir / f"{c['name']}.py"
            path.write_text(c["source"])
            try:
                r = _proxy(d, c["source"], seeds)
            except Exception as e:  # a candidate must never break the report
                cands.append({"name": c["name"], "error": f"{type(e).__name__}: {e}"})
                continue
            delta, se = _paired(d, base_out["out"], r["out"])
            cands.append(
                {
                    "name": c["name"],
                    "hypothesis": c["hypothesis"],
                    "file": str(path),
                    "delta": round(delta, 6),
                    "se": round(se, 6),
                    "verdict": _verdict(delta, se),
                    "command": f"trainwright try {_rel(project, path)} --keep",
                }
            )
        if cfg.task == "regression" and float(np.min(d.y_tr)) >= 0 and _skew(d.y_tr) > 1.0:
            r = _proxy(d, source, seeds, log_target=True)
            delta, se = _paired(d, base_out["out"], r["out"])
            cands.append(
                {
                    "name": "log_target",
                    "hypothesis": f"the target is right-skewed (skew {_skew(d.y_tr):.1f}); fit log1p(y)",
                    "delta": round(delta, 6),
                    "se": round(se, 6),
                    "verdict": _verdict(delta, se),
                    "command": "trainwright auto -o preprocess.target_transform=log1p",
                }
            )
        cands.sort(key=lambda c: -c.get("delta", -np.inf))
    res["candidates"] = cands
    write_json(project.state_dir / "errors.json", res)
    return res


def try_features(
    project: Project,
    candidate: Path | None = None,
    drop: list[str] | None = None,
    hypothesis: str = "",
    keep: bool = False,
    seeds: int = 2,
) -> dict[str, Any]:
    """Proxy val score of the current features.py vs ``candidate`` (or with ``drop`` removed)."""
    cur_path = project.root / FILENAME
    current = cur_path.read_text() if cur_path.exists() else None
    if candidate is not None:
        cand_src = Path(candidate).read_text()
        label = str(candidate)
    elif drop:
        cand_src = (
            _base_block(current)
            + f"\nDROP = {drop!r}\n\n\ndef fit(train):\n    return _base_fit(train)\n\n\n"
            "def transform(df, state):\n"
            "    return _base_transform(df, state).drop(columns=DROP, errors='ignore')\n"
        )
        label = f"drop {', '.join(drop)}"
    else:
        raise LifecycleError("Pass a candidate features file or --drop COLUMNS.")
    d = _Data(project)
    try:
        cand = _proxy(d, cand_src, seeds)
    except Exception as e:  # the candidate is user code: report, don't crash
        raise LifecycleError(f"{label} failed: {type(e).__name__}: {e}") from None
    base = _proxy(d, current, seeds)
    delta, se = _paired(d, base["out"], cand["out"])
    verdict = _verdict(delta, se)
    res: dict[str, Any] = {
        "candidate": label,
        "hypothesis": hypothesis,
        "metric": d.metric,
        "current": round(base["score"], 6),
        "candidate_score": round(cand["score"], 6),
        "delta": round(delta, 6),
        "se": round(se, 6),
        "verdict": verdict,
        "n_features": [base["n_features"], cand["n_features"]],
        "seconds": round(base["seconds"] + cand["seconds"], 1),
        "kept": False,
    }
    if keep:
        if verdict in ("better", "likely better"):
            hist = project.state_dir / "features-history"
            hist.mkdir(parents=True, exist_ok=True)
            if current is not None:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                shutil.copy(cur_path, hist / f"{stamp}-{FeatureHook(current).sha}.py")
            cur_path.write_text(cand_src)
            res["kept"] = True
        else:
            res["not_kept_reason"] = f"{verdict} on the proxy; features.py unchanged"
    with open(project.state_dir / "feature_trials.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **res}) + "\n")
    return res


def feature_trials(project: Project) -> list[dict[str, Any]]:
    p = project.state_dir / "feature_trials.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _rel(project: Project, p: Path) -> str:
    try:
        return str(p.relative_to(project.root))
    except ValueError:
        return str(p)


def _segmentation_report(project: Project, ref: str) -> dict[str, Any]:
    """Per-class IoU, the most confused class pairs and the worst images (per-image mIoU)."""
    import pandas as pd

    from .ensemble import member_outputs
    from .leaderboard import resolve_ref
    from .metrics import per_class_iou, segmentation_metrics
    from .segmentation import image_miou

    cfg = project.config
    run = resolve_ref(project, ref)
    rows, _ = member_outputs(project, run, "val")
    records = get_records(project)
    split = require_split(project, records)
    val = records.subset(split["val"])
    classes = list(split["classes"])
    k = len(classes)
    cm = np.asarray(rows).reshape(-1, k, k).sum(0)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for f in np.argsort(off, axis=None)[::-1][:8]:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        pairs.append(
            {
                "true": classes[i],
                "predicted": classes[j],
                "pixels": int(off[i, j]),
                "share_of_true": round(float(off[i, j] / max(cm[i].sum(), 1)), 4),
            }
        )
    per_img = image_miou(rows, k)
    paths = [_rel(project, Path(p)) for p in (val.paths or [])]
    order = np.argsort(per_img)
    worst = [
        {
            "id": val.ids[i],
            "image_miou": round(float(per_img[i]), 4),
            "path": paths[i] if paths else None,
        }
        for i in order[:10]
    ]
    tab = pd.DataFrame({"id": val.ids, "image_miou": per_img, "path": paths or None})
    rows_path = project.state_dir / "val_errors.csv"
    tab.sort_values("image_miou").to_csv(rows_path, index=False)
    res: dict[str, Any] = {
        "source": f"run {run.id}",
        "metric": cfg.metric,
        "score": round(float(segmentation_metrics(rows, k)[cfg.metric]), 6),
        "n_val": len(per_img),
        "per_class_iou": per_class_iou(rows, classes)[:12],
        "confusions": [
            {"true": c["true"], "predicted": c["predicted"], "count": c["pixels"]} for c in pairs
        ],
        "pixel_confusions": pairs,
        "worst_rows": worst,
        "rows_csv": str(rows_path),
        "candidates": [],
    }
    write_json(project.state_dir / "errors.json", res)
    return res
