"""Duplicate and near-duplicate detection, so they never straddle train/val/test.

Real datasets are full of duplicates: burst photos taken seconds apart, the same support
message sent twice, repeated rows. A random split puts one copy in train and another in test,
and every metric becomes optimistic. trainwright groups duplicates and splits by group.

* images: 256-bit difference hash (dHash) per image; images within ``threshold`` Hamming
  distance are linked; connected components (union-find) form groups.
* text: exact match after normalisation (lower-case, alphanumerics only).
* tabular: exact match of all feature values.
"""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .records import Records

HASH_SIDE = 16  # 16x16 = 256-bit dHash


def _dhash(path: Path) -> np.ndarray:
    from PIL import Image

    try:
        src = Image.open(path)
        if getattr(src, "format", None) == "JPEG":
            src.draft("L", (256, 256))
        im = src.convert("L").resize((HASH_SIDE + 1, HASH_SIDE), Image.Resampling.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
    except Exception:
        bits = np.zeros(HASH_SIDE * HASH_SIDE, dtype=bool)
    return np.packbits(bits)


def image_hashes(records: Records, cache_dir: Path | None = None) -> np.ndarray:
    """-> uint8 [N, 32] packed 256-bit hashes, cached per data fingerprint."""
    assert records.paths is not None
    cache = cache_dir / f"dhash-{records.fingerprint[:16]}.npy" if cache_dir else None
    if cache is not None and cache.exists():
        arr = np.load(cache)
        if len(arr) == len(records):
            return arr
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(_dhash, records.paths)))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
    return arr


class _UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, i: int) -> int:
        p = self.p
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            p[i], i = root, p[i]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def near_duplicate_pairs(hashes: np.ndarray, max_bits: int, block: int = 1024) -> np.ndarray:
    """All pairs (i < j) with Hamming distance <= max_bits, computed blockwise."""
    words = hashes.view(np.uint64) if hashes.shape[1] % 8 == 0 else hashes
    n = len(words)
    out = []
    for s in range(0, n, block):
        a = words[s : s + block]
        d = np.bitwise_count(a[:, None, :] ^ words[None, :, :]).sum(-1)
        ii, jj = np.nonzero(d <= max_bits)
        ii = ii + s
        keep = ii < jj
        out.append(np.stack([ii[keep], jj[keep]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)


def _norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def duplicate_groups(
    records: Records,
    threshold: float = 0.16,
    cache_dir: Path | None = None,
    max_pairwise: int = 60000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """-> (group label per record, stats). Singletons get their own group."""
    n = len(records)
    uf = _UnionFind(n)
    method = ""
    if records.modality == "image" and records.paths is not None:
        hashes = image_hashes(records, cache_dir)
        if n <= max_pairwise:
            max_bits = int(round(threshold * HASH_SIDE * HASH_SIDE))
            for i, j in near_duplicate_pairs(hashes, max_bits):
                uf.union(int(i), int(j))
            method = f"image dHash-256, hamming <= {max_bits}"
        else:
            method = "image dHash-256, exact match (dataset too large for pairwise)"
        keys = [h.tobytes().hex() for h in hashes]
    elif records.modality == "text" and records.texts is not None:
        keys = [_norm_text(t) for t in records.texts]
        method = "normalised text exact match"
    elif records.table is not None:
        import pandas as pd

        keys = [str(h) for h in pd.util.hash_pandas_object(records.table, index=False).to_numpy()]
        method = "exact feature-row match"
    else:
        keys = [str(i) for i in range(n)]
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k in first:
            uf.union(i, first[k])
        else:
            first[k] = i
    roots = np.array([uf.find(i) for i in range(n)])
    sizes = Counter(roots.tolist())
    in_dups = int(sum(s for s in sizes.values() if s > 1))
    conflicting = 0
    if records.task == "classification":
        labels = records.y.astype(str)
        by_group: dict[int, set[str]] = {}
        for r, lab in zip(roots.tolist(), labels.tolist()):
            by_group.setdefault(r, set()).add(lab)
        conflicting = sum(1 for s in by_group.values() if len(s) > 1)
    stats = {
        "method": method,
        "n_groups": len(sizes),
        "items_with_duplicates": in_dups,
        "share_with_duplicates": round(in_dups / max(1, n), 4),
        "largest_group": max(sizes.values()) if sizes else 0,
        "groups_with_conflicting_labels": conflicting,
    }
    return np.array([f"dup{r}" for r in roots]), stats
