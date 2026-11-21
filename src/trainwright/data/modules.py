"""DataModules turn ``Records`` + a frozen split into batches of tensors on the device.

Batches are ``dict[str, Tensor]``; the target is always under ``"y"`` and every other key is
passed to the model's ``forward`` as a keyword argument. That single convention is what lets
the Trainer, export and serving code stay modality-agnostic.

* tabular/text: the whole (small) dataset lives on the device; batches are index slices.
* image: images are decoded + resized once into a disk-backed uint8 cache (reused across runs),
  then augmented on the device per batch.
"""

from __future__ import annotations

import json
import math
import queue
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ExperimentConfig, ProjectConfig
from ..preprocessing import (
    HFTokenizer,
    LabelEncoder,
    TabularPreprocessor,
    TargetScaler,
    TextTokenizer,
    load_image,
    text_preprocessor_from_state,
)
from .records import Records

Batch = dict[str, torch.Tensor]


class TensorLoader:
    """Minimal, fast in-memory loader over a dict of equally-sized tensors."""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last_singleton: bool = False,
        transform: Callable[[Batch], Batch] | None = None,
    ):
        self.tensors = tensors
        self.n = len(next(iter(tensors.values())))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last_singleton = drop_last_singleton
        self.transform = transform
        self._gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        n = math.ceil(self.n / self.batch_size)
        if self.drop_last_singleton and self.n % self.batch_size == 1 and n > 1:
            n -= 1
        return n

    def __iter__(self) -> Iterator[Batch]:
        device = next(iter(self.tensors.values())).device
        if self.shuffle:
            order = torch.randperm(self.n, generator=self._gen).to(device)
        else:
            order = torch.arange(self.n, device=device)
        for b in range(len(self)):
            idx = order[b * self.batch_size : (b + 1) * self.batch_size]
            batch = {k: v[idx] for k, v in self.tensors.items()}
            yield self.transform(batch) if self.transform else batch


class DataModule:
    """Base class. Subclass for custom data; the Trainer only needs the two loaders."""

    task: str = "classification"
    classes: list[str] | None = None

    def train_dataloader(self) -> Any:
        raise NotImplementedError

    def val_dataloader(self) -> Any:
        raise NotImplementedError

    def dataloader(self, split: str) -> Any:
        return self.val_dataloader() if split == "val" else self.train_dataloader()

    @property
    def spec(self) -> dict[str, Any]:
        return {}


class ProjectDataModule(DataModule):
    """Builds a DataModule for a trainwright project (tabular / text / image)."""

    def __init__(
        self,
        project: ProjectConfig,
        exp: ExperimentConfig,
        records: Records,
        split: dict[str, Any],
        device: str,
        cache_dir: Path | None = None,
        splits: tuple[str, ...] = ("train", "val"),
        preprocessor_state: dict[str, Any] | None = None,
        log: Callable[[str], None] = lambda s: None,
    ):
        self.project = project
        self.exp = exp
        self.task = project.task
        self.modality = project.modality
        self.device = device
        self.log = log
        self.split_ids = {s: split[s] for s in ("train", "val", "test")}
        self.records = {s: records.subset(split[s]) for s in splits}
        if "train" not in self.records:
            self.records["train"] = records.subset(split["train"])
        self.classes = split.get("classes") if self.task == "classification" else None
        self.label_encoder = LabelEncoder(self.classes) if self.classes else None
        self.target_scaler: TargetScaler | None = None
        self._tensors: dict[str, dict[str, torch.Tensor]] = {}
        self._image_source: dict[str, ImageSource] = {}
        self._spec: dict[str, Any] = {}
        self._fit_or_load(preprocessor_state, cache_dir, full_records=records)

    # ------------------------------------------------------------------ setup
    def _targets(self, recs: Records) -> np.ndarray:
        if self.task == "classification":
            assert self.label_encoder is not None
            return self.label_encoder.encode(recs.y.tolist())
        assert self.target_scaler is not None
        return self.target_scaler.encode(recs.y.astype(float))

    def _fit_or_load(
        self, state: dict[str, Any] | None, cache_dir: Path | None, full_records: Records
    ) -> None:
        pp = self.exp.preprocess
        train = self.records["train"]
        if self.task == "regression":
            if state and "target" in state:
                self.target_scaler = TargetScaler.from_state(state["target"])
            else:
                self.target_scaler = TargetScaler(pp.target_transform).fit(train.y.astype(float))

        if self.modality == "tabular":
            if state:
                self.pre: Any = TabularPreprocessor.from_state(state["features"])
            else:
                self.pre = TabularPreprocessor(
                    pp.numeric, pp.cat_min_freq, pp.max_cat_cardinality
                ).fit(train.table_dict(), drop=self.project.data.drop)
            for s, recs in self.records.items():
                feats = self.pre.transform(recs.table_dict())
                self._tensors[s] = {
                    "x_num": torch.from_numpy(feats["x_num"]).to(self.device),
                    "x_cat": torch.from_numpy(feats["x_cat"]).to(self.device),
                    "y": torch.from_numpy(self._targets(recs)).to(self.device),
                }
            self._spec = dict(self.pre.spec)
        elif self.modality == "text":
            if state:
                self.pre = text_preprocessor_from_state(state["features"])
            elif self.exp.model.name == "text_pretrained":
                params = self.exp.model.params
                self.pre = HFTokenizer(
                    params.get("encoder", "sentence-transformers/all-MiniLM-L6-v2"),
                    int(params.get("max_length", 128)),
                ).fit([])
            else:
                self.pre = TextTokenizer(
                    pp.max_vocab, pp.ngrams, pp.max_tokens, pp.lowercase, pp.min_token_freq
                )
                self.pre.fit(train.texts or [])
            for s, recs in self.records.items():
                self._tensors[s] = {
                    "tokens": torch.from_numpy(self.pre.encode(recs.texts or [])).to(self.device),
                    "y": torch.from_numpy(self._targets(recs)).to(self.device),
                }
            self._spec = {
                "vocab_size": self.pre.vocab_size,
                "max_tokens": self.pre.max_tokens,
                "pad_id": getattr(self.pre, "pad_id", 0),
            }
        elif self.modality == "image":
            size = pp.image_size
            self.pre = None
            cache = None
            if pp.cache_images != "never" and cache_dir is not None:
                est = len(full_records) * 3 * size * size
                if pp.cache_images == "always" or est < 20e9:
                    cache = ImageCache.build(cache_dir, full_records, size, self.log)
            for s, recs in self.records.items():
                self._image_source[s] = ImageSource(recs, size, cache)
                self._tensors[s] = {"y": torch.from_numpy(self._targets(recs))}
            self._spec = {"image_size": size}
        else:
            raise ValueError(self.modality)
        self._spec["n_outputs"] = len(self.classes) if self.classes else 1
        self._spec["task"] = self.task
        self._spec["modality"] = self.modality

    # ------------------------------------------------------------------ public
    @property
    def spec(self) -> dict[str, Any]:
        return self._spec

    def preprocessor_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"modality": self.modality, "task": self.task}
        if self.modality in ("tabular", "text"):
            state["features"] = self.pre.state_dict()
        else:
            state["features"] = {"type": "image", "image_size": self.exp.preprocess.image_size}
        if self.target_scaler is not None:
            state["target"] = self.target_scaler.state_dict()
        if self.classes:
            state["classes"] = self.classes
        return state

    def class_weights(self) -> torch.Tensor | None:
        if self.task != "classification" or self.exp.train.class_weights == "none":
            return None
        y = self._tensors["train"]["y"].cpu().numpy()
        k = len(self.classes or [])
        counts = np.bincount(y, minlength=k).astype(np.float64)
        w = np.where(counts > 0, len(y) / (k * np.maximum(counts, 1)), 0.0)
        return torch.tensor(w, dtype=torch.float32)

    def n(self, split: str) -> int:
        return len(self.records[split])

    def ids(self, split: str) -> list[str]:
        return self.records[split].ids

    def y_raw(self, split: str) -> np.ndarray:
        return self.records[split].y

    def y_encoded(self, split: str) -> np.ndarray:
        return self._tensors[split]["y"].cpu().numpy()

    def train_dataloader(self) -> Any:
        return self.dataloader("train", train=True)

    def val_dataloader(self) -> Any:
        return self.dataloader("val", train=False)

    def dataloader(self, split: str, train: bool = False) -> Any:
        bs = self.exp.train.batch_size
        seed = self.exp.train.seed
        if self.modality == "image":
            aug = self.exp.preprocess.augment if train else "none"
            return ImageLoader(
                self._image_source[split],
                self._tensors[split]["y"],
                bs,
                shuffle=train,
                seed=seed,
                device=self.device,
                augment=aug,
                hflip=self.exp.preprocess.hflip,
                drop_last_singleton=train,
            )
        return TensorLoader(
            self._tensors[split],
            bs if train else max(bs, 1024),
            shuffle=train,
            seed=seed,
            drop_last_singleton=train,
        )


# --------------------------------------------------------------------------- images


class ImageCache:
    """Disk-backed uint8 array [N, 3, S, S] of decoded, resized images in records order."""

    def __init__(self, array: np.ndarray, index: dict[str, int]):
        self.array = array
        self.index = index

    @classmethod
    def build(
        cls, cache_dir: Path, records: Records, size: int, log: Callable[[str], None]
    ) -> ImageCache:
        assert records.paths is not None
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = cache_dir / f"images-{size}-{records.fingerprint[:16]}"
        arr_path, meta_path = stem.with_suffix(".npy"), stem.with_suffix(".json")
        n = len(records)
        if arr_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("complete") and meta.get("n") == n:
                arr = np.load(arr_path, mmap_mode="r")
                return cls(arr, {i: k for k, i in enumerate(meta["ids"])})
        log(
            f"building image cache for {n} images at {size}px -> {arr_path.name} (one-off, reused by later runs)"
        )
        arr = np.lib.format.open_memmap(
            arr_path, mode="w+", dtype=np.uint8, shape=(n, 3, size, size)
        )
        bad: list[str] = []

        def work(i: int) -> None:
            try:
                arr[i] = load_image(records.paths[i], size)  # type: ignore[index]
            except Exception:
                arr[i] = 0
                bad.append(records.ids[i])

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(work, range(n), chunksize=64))
        arr.flush()
        meta_path.write_text(
            json.dumps(
                {"n": n, "size": size, "ids": records.ids, "complete": True, "unreadable": bad}
            )
        )
        if bad:
            log(f"WARNING: {len(bad)} unreadable images were replaced by black images: {bad[:5]}")
        return cls(np.load(arr_path, mmap_mode="r"), {i: k for k, i in enumerate(records.ids)})


class ImageSource:
    """Fetch uint8 image batches for one split, from the cache or by decoding on the fly."""

    def __init__(self, records: Records, size: int, cache: ImageCache | None):
        self.records = records
        self.size = size
        self.cache = cache
        self.rows = (
            np.array([cache.index[i] for i in records.ids], dtype=np.int64) if cache else None
        )
        self._pool = ThreadPoolExecutor(max_workers=8) if cache is None else None

    def __len__(self) -> int:
        return len(self.records)

    def get(self, idx: np.ndarray) -> np.ndarray:
        if self.cache is not None and self.rows is not None:
            rows = self.rows[idx]
            order = np.argsort(rows)
            out = np.empty((len(rows), 3, self.size, self.size), dtype=np.uint8)
            out[order] = self.cache.array[rows[order]]
            return out
        assert self._pool is not None and self.records.paths is not None
        paths = [self.records.paths[i] for i in idx]
        return np.stack(list(self._pool.map(lambda p: load_image(p, self.size), paths)))


class ImageLoader:
    def __init__(
        self,
        source: ImageSource,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        seed: int,
        device: str,
        augment: str = "none",
        hflip: bool = True,
        drop_last_singleton: bool = False,
    ):
        self.source = source
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.augment = augment
        self.hflip = hflip
        self.drop_last_singleton = drop_last_singleton
        self._rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self) -> int:
        n = math.ceil(len(self.source) / self.batch_size)
        if self.drop_last_singleton and len(self.source) % self.batch_size == 1 and n > 1:
            n -= 1
        return n

    def _batches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(self.source)
        order = self._rng.permutation(n) if self.shuffle else np.arange(n)
        for b in range(len(self)):
            idx = order[b * self.batch_size : (b + 1) * self.batch_size]
            yield self.source.get(idx), idx

    def __iter__(self) -> Iterator[Batch]:
        q: queue.Queue[Any] = queue.Queue(maxsize=3)
        sentinel = object()

        def producer() -> None:
            try:
                for item in self._batches():
                    q.put(item)
            except Exception as e:  # surface loader errors in the main thread
                q.put(e)
            q.put(sentinel)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            imgs, idx = item
            x = torch.from_numpy(imgs).to(self.device, non_blocking=True).float().div_(255.0)
            if self.augment != "none":
                x = augment_images(x, self.augment, self.hflip, self._gen)
            yield {"image": x, "y": self.y[torch.from_numpy(idx)].to(self.device)}
        t.join()


_AUG = {
    #          scale range      max rot  translate  jitter  erase
    "light": ((0.80, 1.0), 0.0, 0.08, 0.10, 0.0),
    "medium": ((0.65, 1.0), 15.0, 0.12, 0.25, 0.0),
    "strong": ((0.45, 1.0), 25.0, 0.15, 0.40, 0.25),
}


def augment_images(x: torch.Tensor, level: str, hflip: bool, gen: torch.Generator) -> torch.Tensor:
    """Vectorised on-device augmentation: random scale/translate/rotate/flip + colour jitter + erase."""
    (smin, smax), rot, trans, jitter, erase = _AUG[level]
    b = x.shape[0]

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=gen).to(x.device)

    zoom = 1.0 / (smin + (smax - smin) * rand(b)).sqrt()  # zoom in == random-resized-crop
    ang = (rand(b) * 2 - 1) * math.radians(rot)
    tx, ty = (rand(b) * 2 - 1) * trans, (rand(b) * 2 - 1) * trans
    flip = torch.where(rand(b) < 0.5, -1.0, 1.0) if hflip else torch.ones(b, device=x.device)
    cos, sin = torch.cos(ang) / zoom, torch.sin(ang) / zoom
    theta = torch.stack(
        [torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if jitter > 0:
        bright = 1 + (rand(b, 1, 1, 1) * 2 - 1) * jitter
        contrast = 1 + (rand(b, 1, 1, 1) * 2 - 1) * jitter
        sat = 1 + (rand(b, 1, 1, 1) * 2 - 1) * jitter
        x = x * bright
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * contrast + mean
        gray = x.mean(dim=1, keepdim=True)
        x = (x - gray) * sat + gray
    if erase > 0:
        h, w = x.shape[-2:]
        mask = rand(b) < erase
        eh = (rand(b) * 0.3 + 0.1) * h
        ew = (rand(b) * 0.3 + 0.1) * w
        cy, cx = rand(b) * h, rand(b) * w
        ys = torch.arange(h, device=x.device).view(1, h, 1)
        xs = torch.arange(w, device=x.device).view(1, 1, w)
        box = ((ys - cy.view(b, 1, 1)).abs() < eh.view(b, 1, 1) / 2) & (
            (xs - cx.view(b, 1, 1)).abs() < ew.view(b, 1, 1) / 2
        )
        box = box & mask.view(b, 1, 1)
        x = x.masked_fill(box.unsqueeze(1), 0.5)
    return x.clamp_(0, 1)
