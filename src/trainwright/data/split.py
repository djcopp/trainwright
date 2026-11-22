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


def segmentation_classes(records: Records, cfg: ProjectConfig, sample: int = 400) -> list[str]:
    """Class names in mask-value order: data.classes, else 0..max value seen (255 = ignore)."""
    if cfg.data.classes:
        return list(cfg.data.classes)
    from PIL import Image

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(records))[:sample]
    top = 0
    for i in idx:
        a = np.asarray(Image.open(records.y[i]))
        a = a[a != 255]
        if a.size:
            top = max(top, int(a.max()))
    return [str(v) for v in range(top + 1)]


def make_split(
    records: Records,
    cfg: ProjectConfig,
    dup_groups: np.ndarray | None = None,
    dup_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    elif strategy in ("stratified", "random"):
        # Allocate *units* (groups of duplicates/near-duplicates, singletons otherwise) so that
        # no duplicate group straddles splits. Without duplicates this is a plain item split.
        units = dup_groups if dup_groups is not None else np.array([str(i) for i in idx])
        unit_members: dict[str, list[int]] = {}
        for i, u in enumerate(units.tolist()):
            unit_members.setdefault(u, []).append(i)
        if strategy == "stratified":
            labels = records.y.astype(str)
            by_class: dict[str, list[str]] = {}
            for u, mem in unit_members.items():
                maj = Counter(labels[mem].tolist()).most_common(1)[0][0]
                by_class.setdefault(maj, []).append(u)
            pools = [(c, by_class[c]) for c in sorted(by_class)]
        else:
            pools = [("all", list(unit_members))]
        train_l, val_l, test_l = [], [], []
        for cls, us in pools:
            order = rng.permutation(len(us))
            us = [us[k] for k in order]
            m = sum(len(unit_members[u]) for u in us)
            if strategy == "stratified" and len(us) < 3:
                warnings.append(
                    f"class {cls!r} has only {len(us)} distinct example group(s); all placed in train"
                )
                train_l += [i for u in us for i in unit_members[u]]
                continue
            t_test = max(1, int(round(m * test_f)))
            t_val = max(1, int(round(m * val_f)))
            n_test = n_val = 0
            for u in us:
                mem = unit_members[u]
                if n_test < t_test and (n_test + len(mem) <= t_test * 1.5 or n_test == 0):
                    test_l += mem
                    n_test += len(mem)
                elif n_val < t_val and (n_val + len(mem) <= t_val * 1.5 or n_val == 0):
                    val_l += mem
                    n_val += len(mem)
                else:
                    train_l += mem
        train_idx, val_idx, test_idx = (
            np.array(sorted(x), dtype=int) for x in (train_l, val_l, test_l)
        )
        if dup_groups is not None and (len(val_idx) == 0 or len(test_idx) == 0):
            # grouping too coarse (e.g. near-identical synthetic images): fall back loudly
            fallback = make_split(records, cfg, None, {**(dup_stats or {}), "fallback": True})
            fallback["warnings"].insert(
                0,
                f"near-duplicate grouping produced only {len(unit_members)} groups, too few to fill "
                "val/test; fell back to an item-level split. Scores may be inflated by near-copies.",
            )
            return fallback
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
        "dedup": dup_stats if strategy in ("stratified", "random") else None,
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
    elif cfg.task == "segmentation":
        out["classes"] = segmentation_classes(records, cfg)
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
        "dedup",
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
