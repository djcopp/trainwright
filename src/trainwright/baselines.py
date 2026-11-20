"""Reference baselines, recorded as runs on the *same* split with the *same* metrics.

Baselines answer "is the neural model actually earning its keep?". They are deliberately
standard (scikit-learn) and untuned: a floor to beat, not a competitor to optimise. If a
baseline wins, that is a real finding — report it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ._utils import write_json
from .data.records import Records
from .metrics import compute_metrics
from .pipeline import get_records, monitor_key, require_split
from .preprocessing import LabelEncoder, TabularPreprocessor, load_image
from .project import Project
from .tracking import Run

BASELINES = {
    "tabular": ["dummy", "linear", "hist_gbm"],
    "text": ["dummy", "tfidf_linear"],
    "image": ["dummy", "pixels_linear"],
}

DESCRIPTIONS = {
    "dummy": "predicts class priors / the training mean: the absolute floor",
    "linear": "logistic/ridge regression on standardised numerics + one-hot categoricals",
    "hist_gbm": "scikit-learn HistGradientBoosting with native categorical support (strong for tables)",
    "tfidf_linear": "TF-IDF word 1-2 grams + logistic/ridge regression (strong for text)",
    "pixels_linear": "logistic/ridge regression on 16x16 downsampled pixels",
}


def _tabular_frame(recs: Records, cols: list[str]) -> Any:
    import pandas as pd

    d = recs.table_dict()
    return pd.DataFrame({c: d[c] for c in cols})


def _make_tabular(name: str, task: str, pre: TabularPreprocessor, seed: int) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    num, cat = pre.numeric_columns, pre.categorical_columns
    if name == "linear":
        ct = ColumnTransformer(
            [
                (
                    "num",
                    make_pipeline(
                        SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                    ),
                    num,
                ),
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(strategy="constant", fill_value="<NA>"),
                        OneHotEncoder(
                            handle_unknown="infrequent_if_exist",
                            min_frequency=5,
                            max_categories=100,
                        ),
                    ),
                    cat,
                ),
            ]
        )
        est = LogisticRegression(max_iter=2000) if task == "classification" else Ridge(alpha=1.0)
        return make_pipeline(ct, est)
    if name == "hist_gbm":
        ct = ColumnTransformer(
            [
                ("num", "passthrough", num),
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(strategy="constant", fill_value="<NA>"),
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1,
                        ),
                    ),
                    cat,
                ),
            ]
        )
        cat_mask = [False] * len(num) + [True] * len(cat)
        kw = dict(
            categorical_features=cat_mask if cat else None, random_state=seed, early_stopping=True
        )
        est = (
            HistGradientBoostingClassifier(**kw)
            if task == "classification"
            else HistGradientBoostingRegressor(**kw)
        )
        return make_pipeline(ct, est)
    raise KeyError(name)


def _to_str_cats(df: Any, cat: list[str]) -> Any:
    df = df.copy()
    for c in cat:
        df[c] = df[c].astype(object).where(df[c].notna(), "<NA>").astype(str)
    return df


def _pixels(recs: Records, size: int = 16) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor

    assert recs.paths is not None
    with ThreadPoolExecutor(8) as ex:
        arrs = list(ex.map(lambda p: load_image(p, size), recs.paths))
    return np.stack(arrs).reshape(len(arrs), -1).astype(np.float32) / 255.0


def _proba_full(model: Any, X: Any, n_classes: int) -> np.ndarray:
    p = model.predict_proba(X)
    full = np.zeros((p.shape[0], n_classes))
    full[:, np.asarray(model.classes_, dtype=int)] = p
    return full


def run_baselines(
    project: Project, which: list[str] | None = None, progress: Callable[[str], None] | None = print
) -> list[Run]:
    import joblib
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = project.config
    project.ensure_dirs()
    records = get_records(project)
    split = require_split(project, records)
    names = which or BASELINES[cfg.modality]
    unknown = [n for n in names if n not in BASELINES[cfg.modality]]
    if unknown:
        raise ValueError(
            f"Unknown baseline(s) {unknown} for {cfg.modality}. Options: {BASELINES[cfg.modality]}"
        )
    train, val = records.subset(split["train"]), records.subset(split["val"])
    task = cfg.task
    classes = split.get("classes")
    le = LabelEncoder(classes) if classes else None
    y_tr = le.encode(train.y.tolist()) if le else train.y.astype(float)
    y_va = le.encode(val.y.tolist()) if le else val.y.astype(float)
    _, mode = monitor_key(cfg)

    # features per modality
    feats: dict[str, Any] = {}
    if cfg.modality == "tabular":
        pre = TabularPreprocessor(max_cat_cardinality=1000).fit(
            train.table_dict(), drop=cfg.data.drop
        )
        cols = pre.numeric_columns + pre.categorical_columns
        feats["train"] = _to_str_cats(_tabular_frame(train, cols), pre.categorical_columns)
        feats["val"] = _to_str_cats(_tabular_frame(val, cols), pre.categorical_columns)
    elif cfg.modality == "text":
        feats["train"], feats["val"] = train.texts, val.texts
    else:
        if progress:
            progress("decoding 16x16 thumbnails for the pixel baseline...")
        feats["train"], feats["val"] = _pixels(train), _pixels(val)

    runs = []
    for name in names:
        t0 = time.time()
        if name == "dummy":
            model: Any = (
                DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor()
            )
        elif cfg.modality == "tabular":
            model = _make_tabular(name, task, pre, cfg.split.seed)
        elif name == "tfidf_linear":
            vec = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True
            )
            est = (
                LogisticRegression(max_iter=2000, C=4.0)
                if task == "classification"
                else Ridge(alpha=1.0)
            )
            model = make_pipeline(vec, est)
        elif name == "pixels_linear":
            est = (
                LogisticRegression(max_iter=1000) if task == "classification" else Ridge(alpha=1.0)
            )
            model = make_pipeline(StandardScaler(), est)
        else:
            raise KeyError(name)
        Xtr = feats["train"] if name != "dummy" else np.zeros((len(y_tr), 1))
        Xva = feats["val"] if name != "dummy" else np.zeros((len(y_va), 1))
        model.fit(Xtr, y_tr)
        if task == "classification":
            out = _proba_full(model, Xva, len(classes or []))
            metrics = compute_metrics(
                "classification", y_va, out, len(classes or []), outputs_are_probs=True
            )
            tr_out = _proba_full(model, Xtr, len(classes or []))
            tr_metrics = compute_metrics(
                "classification", y_tr, tr_out, len(classes or []), outputs_are_probs=True
            )
        else:
            metrics = compute_metrics("regression", y_va, model.predict(Xva))
            tr_metrics = compute_metrics("regression", y_tr, model.predict(Xtr))
        elapsed = time.time() - t0
        run = Run.create(
            project.runs_dir,
            f"baseline-{name}",
            kind="baseline",
            hypothesis=f"Reference: {DESCRIPTIONS[name]}",
            project_root=project.root,
            extra_meta={
                "model": f"baseline:{name}",
                "split_hash": split["split_hash"],
                "data_fingerprint": records.fingerprint,
                "status": "completed",
            },
        )
        run.set_status("completed")
        joblib.dump(
            {
                "model": model,
                "modality": cfg.modality,
                "task": task,
                "classes": classes,
                "columns": (pre.numeric_columns, pre.categorical_columns)
                if cfg.modality == "tabular"
                else None,
            },
            run.dir / "model.joblib",
        )
        run.write_summary(
            {
                "kind": "baseline",
                "model": f"baseline:{name}",
                "description": DESCRIPTIONS[name],
                "primary_metric": cfg.metric,
                "metric_mode": mode,
                "train_time_s": round(elapsed, 2),
                "val": metrics,
                "train": tr_metrics,
                "split_hash": split["split_hash"],
                "n_train": len(y_tr),
                "n_val": len(y_va),
            }
        )
        write_json(run.diagnosis_path, {"verdict": "baseline", "findings": []})
        score = metrics.get(cfg.metric)
        if progress:
            progress(
                f"baseline {name:<14} val_{cfg.metric}={score:.4f}  ({elapsed:.1f}s)  -> {run.id}"
            )
        runs.append(run)
    return runs
