"""Frozen train/val/test splits.

A split is created once per project and stored in ``.trainwright/split.json`` together with a
fingerprint of the data. Every run uses the same split, so results are comparable, and the
test split is *sealed*: only ``trainwright evaluate --split test`` touches it, and every such
access is recorded in a ledger.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .._utils import now_iso, read_json, sha256_strings, write_json
from ..config import ProjectConfig
from ..project import Project
from .records import Records


def resolve_strategy(cfg: ProjectConfig) -> str:
    s = cfg.split.strategy
    if s != "auto":
        return s
    if cfg.data.time_column:
        return "time"
    if cfg.data.group_column:
        return "group"
    return "stratified" if cfg.task == "classification" else "random"


def make_split(records: Records, cfg: ProjectConfig) -> dict[str, Any]:
    strategy = resolve_strategy(cfg)
    rng = np.random.default_rng(cfg.split.seed)
    n = len(records)
    val_f, test_f = cfg.split.val, cfg.split.test
    idx = np.arange(n)
    warnings: list[str] = []

    if strategy == "time":
        if records.times is None:
            raise ValueError("split.strategy=time requires data.time_column")
        order = np.argsort(records.times, kind="stable")
        n_test, n_val = int(round(n * test_f)), int(round(n * val_f))
        test_idx = order[n - n_test :]
        val_idx = order[n - n_test - n_val : n - n_test]
        train_idx = order[: n - n_test - n_val]
    elif strategy == "group":
        if records.groups is None:
            raise ValueError("split.strategy=group requires data.group_column")
        uniq = np.unique(records.groups)
        rng.shuffle(uniq)
        sizes = Counter(records.groups.tolist())
        assign: dict[str, str] = {}
        n_test_target, n_val_target = n * test_f, n * val_f
        n_test = n_val = 0
        for g in uniq:
            if n_test < n_test_target:
                assign[g] = "test"
                n_test += sizes[g]
            elif n_val < n_val_target:
                assign[g] = "val"
                n_val += sizes[g]
            else:
                assign[g] = "train"
        parts = np.array([assign[g] for g in records.groups])
        train_idx, val_idx, test_idx = (
            idx[parts == "train"],
            idx[parts == "val"],
            idx[parts == "test"],
        )
    elif strategy == "stratified":
        train_l, val_l, test_l = [], [], []
        labels = records.y.astype(str)
        for cls in sorted(set(labels.tolist())):
            members = idx[labels == cls]
            rng.shuffle(members)
            m = len(members)
            if m < 3:
                warnings.append(f"class {cls!r} has only {m} example(s); all placed in train")
                train_l.append(members)
                continue
            n_test = max(1, int(round(m * test_f)))
            n_val = max(1, int(round(m * val_f)))
            test_l.append(members[:n_test])
            val_l.append(members[n_test : n_test + n_val])
            train_l.append(members[n_test + n_val :])
        train_idx, val_idx, test_idx = (
            np.sort(np.concatenate(x)) if x else np.array([], int) for x in (train_l, val_l, test_l)
        )
    elif strategy == "random":
        perm = rng.permutation(n)
        n_test, n_val = int(round(n * test_f)), int(round(n * val_f))
        test_idx, val_idx, train_idx = (
            perm[:n_test],
            perm[n_test : n_test + n_val],
            perm[n_test + n_val :],
        )
    else:
        raise ValueError(f"Unknown split strategy {strategy}")

    ids = np.array(records.ids, dtype=object)
    out: dict[str, Any] = {
        "strategy": strategy,
        "seed": cfg.split.seed,
        "fractions": {"val": val_f, "test": test_f},
        "data_fingerprint": records.fingerprint,
        "created_at": now_iso(),
        "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "warnings": warnings,
        "train": ids[np.sort(train_idx)].tolist(),
        "val": ids[np.sort(val_idx)].tolist(),
        "test": ids[np.sort(test_idx)].tolist(),
    }
    if cfg.task == "classification":
        out["classes"] = sorted(set(records.y.astype(str).tolist()))
        out["class_counts"] = {
            part: dict(Counter(records.y[records.positions(out[part])].astype(str).tolist()))
            for part in ("train", "val", "test")
        }
        missing_val = [c for c in out["classes"] if c not in out["class_counts"]["val"]]
        if missing_val:
            warnings.append(f"{len(missing_val)} class(es) absent from val: {missing_val[:10]}")
    out["split_hash"] = sha256_strings(
        ["train", *out["train"], "val", *out["val"], "test", *out["test"]]
    )[:16]
    return out


def save_split(project: Project, split: dict[str, Any]) -> None:
    write_json(project.split_path, split)


def load_split(project: Project) -> dict[str, Any] | None:
    return read_json(project.split_path) if project.split_path.exists() else None


def split_summary(split: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "strategy",
        "seed",
        "fractions",
        "counts",
        "warnings",
        "split_hash",
        "created_at",
        "data_fingerprint",
    ]
    out = {k: split.get(k) for k in keep}
    if "classes" in split:
        out["n_classes"] = len(split["classes"])
    return out
