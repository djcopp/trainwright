"""Data profiling: the "look at your data first" step, as structured output.

Produces facts (sizes, types, missingness, class balance) *and* recommendations an agent can
act on (task type, primary metric, columns to drop, leakage suspects, split strategy, model
starting point). Images additionally get a contact sheet PNG an agent can open and look at.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from .records import Records


def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_float_dtype(y):
        vals = y.dropna()
        if len(vals) and (np.mod(vals.to_numpy(), 1) != 0).any():
            return "regression"
        return "classification" if vals.nunique() <= 50 else "regression"
    if pd.api.types.is_integer_dtype(y):
        return "classification" if y.nunique() <= 50 else "regression"
    return "classification"


def recommend_metric(task: str, class_counts: dict[str, int] | None) -> tuple[str, str, str]:
    """-> (metric, mode, why)"""
    if task == "regression":
        return "rmse", "min", "RMSE on the original target scale; MAE and R2 are also reported."
    counts = list((class_counts or {}).values())
    k = len(counts)
    ratio = (max(counts) / max(1, min(counts))) if counts else 1.0
    if k == 2:
        return (
            "roc_auc",
            "max",
            f"Binary task (imbalance {ratio:.1f}:1): ROC-AUC is threshold-free and "
            "robust to imbalance; accuracy/F1 also reported.",
        )
    if ratio > 3:
        return (
            "macro_f1",
            "max",
            f"{k} classes with imbalance {ratio:.1f}:1: macro-F1 weights every class "
            "equally so the majority class can't dominate.",
        )
    return (
        "accuracy",
        "max",
        f"{k} roughly balanced classes: accuracy is interpretable; macro-F1 also reported.",
    )


# --------------------------------------------------------------------------- tabular


def _column_profile(s: pd.Series, n: int) -> dict[str, Any]:
    from ..preprocessing import infer_column_kind

    missing = float(s.isna().mean())
    n_unique = int(s.nunique(dropna=True))
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        kind = "numeric" if n_unique > 1 else "constant"
    elif pd.api.types.is_datetime64_any_dtype(s):
        kind = "datetime"
    else:
        sample = s.dropna().astype(str).head(5000).tolist()
        kind = infer_column_kind(sample) if sample else "empty"
        if kind == "categorical":
            parsed = pd.to_datetime(pd.Series(sample[:200]), errors="coerce", format="mixed")
            if (
                len(sample)
                and parsed.notna().mean() > 0.95
                and not all(x.isdigit() for x in sample[:50])
            ):
                kind = "datetime-like string"
    prof: dict[str, Any] = {
        "dtype": str(s.dtype),
        "kind": kind,
        "missing": round(missing, 4),
        "n_unique": n_unique,
    }
    if kind == "numeric" and not pd.api.types.is_bool_dtype(s):
        v = pd.to_numeric(s, errors="coerce").dropna()
        if len(v):
            prof.update(
                mean=float(v.mean()),
                std=float(v.std()),
                min=float(v.min()),
                max=float(v.max()),
                skew=float(v.skew()) if len(v) > 2 else 0.0,
            )
    if kind in ("categorical", "text", "datetime-like string"):
        prof["top"] = {str(k): int(c) for k, c in s.astype(str).value_counts().head(5).items()}
    flags = []
    if n_unique == n and n > 20 and kind in ("categorical", "numeric", "text"):
        if kind != "numeric" or (pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing):
            flags.append("id-like (unique per row): drop")
    if kind == "constant":
        flags.append("constant: carries no signal")
    if missing > 0.5:
        flags.append(f"{missing:.0%} missing")
    if kind == "categorical" and n_unique > 1000:
        flags.append("very high cardinality: will be dropped as an id unless you encode it")
    if kind == "text":
        flags.append("free text: consider the text modality or drop")
    if kind in ("datetime", "datetime-like string"):
        flags.append("datetime: consider data.time_column for a chronological split")
    if flags:
        prof["flags"] = flags
    return prof


def _leakage_scan(
    df: pd.DataFrame, y: np.ndarray, task: str, max_cols: int = 100
) -> list[dict[str, Any]]:
    """Single-feature shallow trees that predict the target almost perfectly are suspicious."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    n = len(df)
    if n < 50:
        return []
    rng = np.random.default_rng(0)
    take = rng.choice(n, size=min(n, 5000), replace=False)
    sub, ys = df.iloc[take], y[take]
    out = []
    if task == "classification":
        majority = Counter(ys.tolist()).most_common(1)[0][1] / len(ys)
    for col in list(df.columns)[:max_cols]:
        s = sub[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            x = pd.to_numeric(s, errors="coerce").astype(float).fillna(-1e9).to_numpy()[:, None]
        else:
            if s.nunique() > 0.5 * len(s):
                continue  # ids handled by profile flags
            x = s.astype(str).astype("category").cat.codes.to_numpy()[:, None]
        half = len(x) // 2
        if task == "classification":
            m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(x[:half], ys[:half])
            score = float((m.predict(x[half:]) == ys[half:]).mean())
            if score > 0.99 and majority < 0.97:
                out.append(
                    {
                        "column": col,
                        "single_feature_accuracy": round(score, 4),
                        "why": "one feature alone predicts the target almost perfectly",
                    }
                )
        else:
            m2 = DecisionTreeRegressor(max_depth=4, random_state=0).fit(
                x[:half], ys[:half].astype(float)
            )
            pred = m2.predict(x[half:])
            yt = ys[half:].astype(float)
            ss = float(((yt - yt.mean()) ** 2).sum()) or 1.0
            r2 = 1 - float(((yt - pred) ** 2).sum()) / ss
            if r2 > 0.98:
                out.append(
                    {
                        "column": col,
                        "single_feature_r2": round(r2, 4),
                        "why": "one feature alone explains the target almost perfectly",
                    }
                )
    return out


def profile_tabular(records: Records, cfg: ProjectConfig) -> dict[str, Any]:
    assert records.table is not None
    df = records.table
    n = len(df)
    cols = {c: _column_profile(df[c], n) for c in df.columns}
    configured_drop = set(cfg.data.drop)
    usable = [
        c
        for c, p in cols.items()
        if c not in configured_drop
        and not any(f.startswith(("id-like", "constant")) for f in p.get("flags", []))
    ]
    kinds = Counter(p["kind"] for c, p in cols.items() if c in usable)
    leaks = _leakage_scan(df[[c for c in usable]], records.y, cfg.task) if usable else []
    suggested_drop = sorted(
        {
            c
            for c, p in cols.items()
            if any(f.startswith(("id-like", "constant")) for f in p.get("flags", []))
        }
        | {x["column"] for x in leaks}
    )
    dup = int(df.duplicated().sum())
    return {
        "n_rows": n,
        "n_features": len(df.columns),
        "feature_kinds": dict(kinds),
        "duplicate_rows": dup,
        "columns": cols,
        "leakage_suspects": leaks,
        "suggested_drop": [c for c in suggested_drop if c not in configured_drop],
    }


# --------------------------------------------------------------------------- text


def profile_text(records: Records, cfg: ProjectConfig) -> dict[str, Any]:
    assert records.texts is not None
    lens = np.array([len(t.split()) for t in records.texts])
    empty = int((lens == 0).sum())
    dup = len(records.texts) - len(set(records.texts))
    return {
        "n_docs": len(records.texts),
        "words_per_doc": {
            "mean": float(lens.mean()),
            "p50": float(np.median(lens)),
            "p95": float(np.percentile(lens, 95)),
            "max": int(lens.max()),
        },
        "empty_docs": empty,
        "duplicate_docs": dup,
        "examples": [t[:200] for t in records.texts[:3]],
    }


# --------------------------------------------------------------------------- image


def profile_images(
    records: Records, cfg: ProjectConfig, sheet_path: Path | None = None
) -> dict[str, Any]:
    from PIL import Image

    assert records.paths is not None
    rng = np.random.default_rng(0)
    n = len(records.paths)
    sample_idx: Any = rng.choice(n, size=min(n, 400), replace=False) if n else []
    sizes: list[tuple[int, int]] = []
    modes: Counter[str] = Counter()
    corrupt: list[dict[str, str]] = []
    for i in sample_idx:
        p = records.paths[int(i)]
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
                modes[im.mode] += 1
                im.verify()
        except Exception as e:
            corrupt.append({"path": str(p), "error": f"{type(e).__name__}: {e}"[:200]})
    widths = np.array([s[0] for s in sizes]) if sizes else np.array([0])
    heights = np.array([s[1] for s in sizes]) if sizes else np.array([0])
    out: dict[str, Any] = {
        "n_images": n,
        "sampled": len(sample_idx),
        "width": {
            "p5": float(np.percentile(widths, 5)),
            "p50": float(np.median(widths)),
            "p95": float(np.percentile(widths, 95)),
        },
        "height": {
            "p5": float(np.percentile(heights, 5)),
            "p50": float(np.median(heights)),
            "p95": float(np.percentile(heights, 95)),
        },
        "modes": dict(modes),
        "corrupt_in_sample": corrupt[:20],
    }
    if sheet_path is not None and n:
        try:
            out["contact_sheet"] = str(_contact_sheet(records, sheet_path))
        except Exception as e:  # pragma: no cover - cosmetic
            out["contact_sheet_error"] = str(e)
    return out


def _contact_sheet(
    records: Records, path: Path, per_class: int = 6, thumb: int = 96, max_classes: int = 20
) -> Path:
    from PIL import Image, ImageDraw

    assert records.paths is not None
    labels = records.y.astype(str)
    classes = sorted(set(labels.tolist()))[:max_classes]
    label_w = 160
    sheet = Image.new("RGB", (label_w + per_class * thumb, len(classes) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    rng = np.random.default_rng(0)
    for r, c in enumerate(classes):
        members = np.flatnonzero(labels == c)
        pick = rng.choice(members, size=min(per_class, len(members)), replace=False)
        draw.text((6, r * thumb + thumb // 2 - 6), f"{c[:22]} ({len(members)})", fill="black")
        for k, i in enumerate(pick):
            try:
                with Image.open(records.paths[int(i)]) as src:
                    im = src.convert("RGB")
                    im.thumbnail((thumb, thumb))
                    sheet.paste(im, (label_w + k * thumb, r * thumb))
            except Exception:
                pass
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


# --------------------------------------------------------------------------- entry


def profile(records: Records, cfg: ProjectConfig, state_dir: Path | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "modality": cfg.modality,
        "task": cfg.task,
        "n": len(records),
        "dropped_missing_target": records.dropped_missing_target,
    }
    recs: list[str] = []
    if cfg.task == "classification":
        counts = Counter(records.y.astype(str).tolist())
        ratio = max(counts.values()) / max(1, min(counts.values()))
        out["target"] = {
            "n_classes": len(counts),
            "counts": dict(counts.most_common()),
            "imbalance_ratio": round(ratio, 2),
        }
        if ratio > 5:
            recs.append(
                f"Classes are imbalanced ({ratio:.1f}:1). Watch macro-F1, and try "
                "`-o train.class_weights=balanced` if minority classes underperform."
            )
        small = [c for c, k in counts.items() if k < 10]
        if small:
            recs.append(
                f"{len(small)} class(es) have <10 examples; their metrics will be noisy. "
                "Consider collecting more data or merging rare classes."
            )
        metric, mode, why = recommend_metric("classification", dict(counts))
    else:
        yv = records.y.astype(float)
        skew = float(pd.Series(yv).skew())
        out["target"] = {
            "mean": float(yv.mean()),
            "std": float(yv.std()),
            "min": float(yv.min()),
            "max": float(yv.max()),
            "skew": skew,
        }
        if skew > 2 and yv.min() >= 0:
            recs.append(
                f"Target is right-skewed (skew {skew:.1f}). Try `-o preprocess.target_transform=log1p`."
            )
        metric, mode, why = recommend_metric("regression", None)
    out["recommended_metric"] = {"metric": metric, "mode": mode, "why": why}

    if cfg.modality == "tabular":
        out["tabular"] = profile_tabular(records, cfg)
        t = out["tabular"]
        if t["suggested_drop"]:
            recs.append(
                f"Add to data.drop in trainwright.yaml: {t['suggested_drop']} "
                "(ids/constants/leakage suspects)."
            )
        if t["leakage_suspects"]:
            recs.append(
                "LEAKAGE SUSPECTED: a single feature predicts the target almost perfectly. "
                "Confirm it is available at prediction time before training on it."
            )
        if t["duplicate_rows"] > 0:
            recs.append(
                f"{t['duplicate_rows']} duplicate rows: duplicates across splits inflate validation scores."
            )
        n = t["n_rows"]
        if n < 1000:
            recs.append(
                "Small dataset (<1k rows): expect gradient-boosting baselines to be strong; "
                "use a smaller MLP (hidden=64-128) with more dropout/weight decay."
            )
        else:
            recs.append(
                "Start with the default tabular_mlp (quantile-normalised numerics, categorical "
                "embeddings) and compare against the HistGradientBoosting baseline."
            )
    elif cfg.modality == "text":
        out["text"] = profile_text(records, cfg)
        p95 = out["text"]["words_per_doc"]["p95"]
        recs.append(
            f"95% of docs have <= {p95:.0f} words; preprocess.max_tokens covers unigrams+ngrams, "
            f"set it to ~{int(min(1024, max(64, p95 * 2)))}."
        )
    elif cfg.modality == "image":
        sheet = (state_dir / "samples.png") if state_dir else None
        out["image"] = profile_images(records, cfg, sheet)
        im = out["image"]
        if im.get("contact_sheet"):
            recs.append(
                f"Open {im['contact_sheet']} and look at the images per class before training."
            )
        if im["corrupt_in_sample"]:
            recs.append(
                f"{len(im['corrupt_in_sample'])} unreadable image(s) in the sample; remove them."
            )
        per_class = (
            len(records) / max(1, out["target"].get("n_classes", 1))
            if cfg.task == "classification"
            else len(records)
        )
        if per_class < 100:
            recs.append(
                "Few images per class: fine-tune a pretrained backbone (default image_pretrained "
                "resnet18) and consider train.freeze_backbone_epochs=2 + preprocess.augment=medium."
            )
        recs.append(
            "If left/right orientation matters for your labels (e.g. left vs right parts), "
            "set preprocess.hflip=false."
        )
    out["recommendations"] = recs
    return out
