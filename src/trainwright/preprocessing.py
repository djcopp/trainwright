"""Numpy-only preprocessing with JSON-serialisable state.

Everything here is fit on the *training split only* and is shipped inside exported bundles,
so inference does not need torch, pandas, sklearn or scipy. Keep this module dependency-free
(numpy + stdlib; Pillow imported lazily for images).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import NormalDist
from typing import Any

import numpy as np

MISSING_CAT = "<NA>"


# --------------------------------------------------------------------------- numeric


class NumericTransformer:
    """Median impute (+ missing indicators) then quantile->normal or standard scaling."""

    def __init__(self, method: str = "quantile", n_quantiles: int = 1000, clip: float = 5.0):
        self.method = method
        self.n_quantiles = n_quantiles
        self.clip = clip
        self.columns: list[str] = []
        self.medians: list[float] = []
        self.missing_indicator: list[bool] = []
        self.quantiles: list[list[float]] = []
        self.references: list[float] = []
        self.means: list[float] = []
        self.stds: list[float] = []

    def fit(self, X: np.ndarray, columns: list[str]) -> NumericTransformer:
        X = np.asarray(X, dtype=np.float64)
        self.columns = list(columns)
        n = X.shape[0]
        self.medians, self.missing_indicator = [], []
        self.quantiles, self.means, self.stds = [], [], []
        nq = int(max(2, min(self.n_quantiles, n)))
        probs = np.linspace(0.0, 1.0, nq)
        nd = NormalDist()
        eps = 1e-7
        self.references = [nd.inv_cdf(min(max(p, eps), 1 - eps)) for p in probs]
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = col[np.isfinite(col)]
            med = float(np.median(finite)) if finite.size else 0.0
            self.medians.append(med)
            self.missing_indicator.append(bool(finite.size < col.size))
            filled = np.where(np.isfinite(col), col, med)
            if self.method == "quantile":
                self.quantiles.append(np.quantile(filled, probs).tolist())
            mu, sd = float(filled.mean()), float(filled.std())
            self.means.append(mu)
            self.stds.append(sd if sd > 1e-12 else 1.0)
        return self

    @property
    def n_out(self) -> int:
        return len(self.columns) + sum(self.missing_indicator)

    @property
    def output_names(self) -> list[str]:
        names = list(self.columns)
        names += [f"{c}__missing" for c, m in zip(self.columns, self.missing_indicator) if m]
        return names

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        out = np.empty((X.shape[0], len(self.columns)), dtype=np.float64)
        indicators = []
        ref = np.asarray(self.references)
        for j in range(len(self.columns)):
            col = X[:, j]
            miss = ~np.isfinite(col)
            col = np.where(miss, self.medians[j], col)
            if self.method == "quantile":
                q = np.asarray(self.quantiles[j])
                # average of forward/backward interpolation handles repeated quantiles (ties)
                fwd = np.interp(col, q, ref)
                bwd = -np.interp(-col, -q[::-1], -ref[::-1])
                col = 0.5 * (fwd + bwd)
            elif self.method == "standard":
                col = (col - self.means[j]) / self.stds[j]
            out[:, j] = col
            if self.missing_indicator[j]:
                indicators.append(miss.astype(np.float64))
        if indicators:
            out = np.concatenate([out, np.stack(indicators, axis=1)], axis=1)
        if self.clip:
            out = np.clip(out, -self.clip, self.clip)
        return out.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> NumericTransformer:
        obj = cls()
        obj.__dict__.update(state)
        return obj


# --------------------------------------------------------------------------- categorical


class CategoricalEncoder:
    """Map each categorical column to integer codes. 0 = unknown / rare value."""

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.columns: list[str] = []
        self.vocabs: list[dict[str, int]] = []

    @staticmethod
    def _norm(v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return MISSING_CAT
        return str(v)

    def fit(self, columns_values: list[list[Any]], columns: list[str]) -> CategoricalEncoder:
        self.columns = list(columns)
        self.vocabs = []
        for values in columns_values:
            counts = Counter(self._norm(v) for v in values)
            kept = sorted([k for k, c in counts.items() if c >= self.min_freq])
            self.vocabs.append({k: i + 1 for i, k in enumerate(kept)})
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(v) + 1 for v in self.vocabs]

    def transform(self, columns_values: list[list[Any]]) -> np.ndarray:
        if not self.columns:
            n = len(columns_values[0]) if columns_values else 0
            return np.zeros((n, 0), dtype=np.int64)
        cols = []
        for vocab, values in zip(self.vocabs, columns_values):
            cols.append(np.fromiter((vocab.get(self._norm(v), 0) for v in values), dtype=np.int64))
        return np.stack(cols, axis=1)

    def state_dict(self) -> dict[str, Any]:
        return {"min_freq": self.min_freq, "columns": self.columns, "vocabs": self.vocabs}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> CategoricalEncoder:
        obj = cls(state["min_freq"])
        obj.columns = state["columns"]
        obj.vocabs = state["vocabs"]
        return obj


class TabularPreprocessor:
    """Column-role inference + numeric/categorical transforms for a table of records.

    Works on ``dict[str, list]`` (column -> values) so it needs no pandas at inference.
    """

    def __init__(
        self, numeric: str = "quantile", cat_min_freq: int = 1, max_cat_cardinality: int = 1000
    ):
        self.numeric_method = numeric
        self.cat_min_freq = cat_min_freq
        self.max_cat_cardinality = max_cat_cardinality
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.dropped: dict[str, str] = {}
        self.num = NumericTransformer(numeric if numeric != "none" else "none")
        self.cat = CategoricalEncoder(cat_min_freq)

    def fit(
        self, table: dict[str, list[Any]], drop: list[str] | None = None
    ) -> TabularPreprocessor:
        drop_set = set(drop or [])
        n = len(next(iter(table.values()))) if table else 0
        self.numeric_columns, self.categorical_columns, self.dropped = [], [], {}
        for col, values in table.items():
            if col in drop_set:
                self.dropped[col] = "configured drop"
                continue
            kind = infer_column_kind(values)
            if kind == "numeric":
                self.numeric_columns.append(col)
            elif kind == "categorical":
                n_unique = len({CategoricalEncoder._norm(v) for v in values})
                if n_unique > self.max_cat_cardinality or (n > 50 and n_unique > 0.9 * n):
                    self.dropped[col] = (
                        f"high-cardinality categorical ({n_unique} unique) treated as id"
                    )
                elif n_unique <= 1:
                    self.dropped[col] = "constant"
                else:
                    self.categorical_columns.append(col)
            else:
                self.dropped[col] = kind
        if self.numeric_columns:
            Xn = _to_float_matrix(table, self.numeric_columns)
            self.num.fit(Xn, self.numeric_columns)
        self.cat.fit([table[c] for c in self.categorical_columns], self.categorical_columns)
        return self

    def transform(self, table: dict[str, list[Any]]) -> dict[str, np.ndarray]:
        n = len(next(iter(table.values()))) if table else 0
        missing = [c for c in self.numeric_columns + self.categorical_columns if c not in table]
        if missing:
            raise KeyError(f"Input is missing feature columns: {missing}")
        if self.numeric_columns:
            x_num = self.num.transform(_to_float_matrix(table, self.numeric_columns))
        else:
            x_num = np.zeros((n, 0), dtype=np.float32)
        x_cat = self.cat.transform([table[c] for c in self.categorical_columns])
        if x_cat.shape[0] != n:
            x_cat = np.zeros((n, 0), dtype=np.int64)
        return {"x_num": x_num, "x_cat": x_cat}

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "n_num": self.num.n_out if self.numeric_columns else 0,
            "cat_cardinalities": self.cat.cardinalities,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "dropped": self.dropped,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "tabular",
            "numeric_method": self.numeric_method,
            "cat_min_freq": self.cat_min_freq,
            "max_cat_cardinality": self.max_cat_cardinality,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "dropped": self.dropped,
            "num": self.num.state_dict(),
            "cat": self.cat.state_dict(),
        }

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> TabularPreprocessor:
        obj = cls(s["numeric_method"], s["cat_min_freq"], s["max_cat_cardinality"])
        obj.numeric_columns = s["numeric_columns"]
        obj.categorical_columns = s["categorical_columns"]
        obj.dropped = s["dropped"]
        obj.num = NumericTransformer.from_state(s["num"])
        obj.cat = CategoricalEncoder.from_state(s["cat"])
        return obj


def _to_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _to_float_array(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.dtype.kind in "biuf":
        return values.astype(np.float64)
    return np.fromiter((_to_float(v) for v in values), dtype=np.float64)


def _to_float_matrix(table: dict[str, Any], cols: list[str]) -> np.ndarray:
    return np.stack([_to_float_array(table[c]) for c in cols], axis=1)


def infer_column_kind(values: list[Any]) -> str:
    """numeric | categorical | text | constant | empty"""
    if isinstance(values, np.ndarray) and values.dtype.kind in "biuf":
        arr = values.astype(np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return "empty"
        return "constant" if finite.min() == finite.max() else "numeric"
    non_null = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not non_null:
        return "empty"
    sample = non_null[:5000]
    numeric_ok = 0
    for v in sample:
        if isinstance(v, (bool, int, float, np.integer, np.floating)):
            numeric_ok += 1
        else:
            try:
                float(v)
                numeric_ok += 1
            except (TypeError, ValueError):
                pass
    if numeric_ok == len(sample):
        if len(set(sample)) <= 1 and len(set(non_null)) <= 1:
            return "constant"
        return "numeric"
    strs = [str(v) for v in sample]
    avg_len = sum(len(s) for s in strs) / len(strs)
    n_unique = len(set(strs))
    if avg_len > 40 and n_unique > 0.5 * len(strs):
        return "text"
    return "categorical"


# --------------------------------------------------------------------------- text

_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)


class TextTokenizer:
    """Word unigram (+ optional bi/tri-gram) vocabulary tokenizer. 0 = pad, 1 = unknown."""

    def __init__(
        self,
        max_vocab: int = 50000,
        ngrams: int = 2,
        max_tokens: int = 256,
        lowercase: bool = True,
        min_freq: int = 2,
    ):
        self.max_vocab = max_vocab
        self.ngrams = ngrams
        self.max_tokens = max_tokens
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.vocab: dict[str, int] = {}

    def _tokens(self, text: str) -> list[str]:
        text = text if isinstance(text, str) else ("" if text is None else str(text))
        if self.lowercase:
            text = text.lower()
        words = _TOKEN_RE.findall(text)
        toks = list(words)
        for n in range(2, self.ngrams + 1):
            toks += [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        return toks

    def fit(self, texts: list[str]) -> TextTokenizer:
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(set(self._tokens(t)))
        kept = [w for w, c in counts.most_common() if c >= self.min_freq][: self.max_vocab - 2]
        self.vocab = {w: i + 2 for i, w in enumerate(kept)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 2

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.max_tokens), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.vocab.get(tok, 1) for tok in self._tokens(t)][: self.max_tokens]
            if ids:
                out[i, : len(ids)] = ids
        return out

    def state_dict(self) -> dict[str, Any]:
        return dict(self.__dict__, type="text")

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> TextTokenizer:
        obj = cls()
        obj.__dict__.update({k: v for k, v in s.items() if k != "type"})
        return obj


class HFTokenizer:
    """Wraps a Hugging Face ``tokenizers`` tokenizer (state = its JSON, so bundles are self-contained)."""

    def __init__(self, name: str = "", max_tokens: int = 128, tokenizer_json: str | None = None):
        self.name = name
        self.max_tokens = max_tokens
        self.tokenizer_json = tokenizer_json
        self._tok: Any = None
        self.pad_id = 0

    def _load(self) -> Any:
        if self._tok is None:
            from tokenizers import Tokenizer

            tok = (
                Tokenizer.from_str(self.tokenizer_json)
                if self.tokenizer_json
                else Tokenizer.from_pretrained(self.name)
            )
            self.tokenizer_json = tok.to_str()
            pad = tok.token_to_id("[PAD]")
            if pad is None:
                pad = tok.token_to_id("<pad>")
            self.pad_id = int(pad or 0)
            tok.enable_truncation(self.max_tokens)
            tok.enable_padding(pad_id=self.pad_id, length=self.max_tokens)
            self._tok = tok
        return self._tok

    def fit(self, texts: list[str]) -> HFTokenizer:
        self._load()
        return self

    @property
    def vocab_size(self) -> int:
        return int(self._load().get_vocab_size())

    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self._load().encode_batch([t if isinstance(t, str) else str(t) for t in texts])
        return np.asarray([e.ids for e in enc], dtype=np.int64)

    def state_dict(self) -> dict[str, Any]:
        self._load()
        return {
            "type": "hf_text",
            "name": self.name,
            "max_tokens": self.max_tokens,
            "tokenizer_json": self.tokenizer_json,
            "pad_id": self.pad_id,
        }

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> HFTokenizer:
        return cls(s["name"], s["max_tokens"], s["tokenizer_json"])


def text_preprocessor_from_state(s: dict[str, Any]) -> Any:
    return HFTokenizer.from_state(s) if s.get("type") == "hf_text" else TextTokenizer.from_state(s)


# --------------------------------------------------------------------------- image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def load_image(source: Any, size: int) -> np.ndarray:
    """Load an image (path, bytes or PIL image) -> uint8 [3, size, size].

    Resize the short side to ``size`` then centre-crop. Applies EXIF orientation.
    """
    import io

    from PIL import Image, ImageOps

    img: Image.Image
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)
    if getattr(img, "format", None) == "JPEG":
        # decode at reduced resolution (1/2, 1/4, 1/8): 10x+ faster for large phone photos
        img.draft("RGB", (size * 2, size * 2))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


class ImagePreprocessor:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def transform(self, sources: list[Any]) -> dict[str, np.ndarray]:
        arr = np.stack([load_image(s, self.image_size) for s in sources]).astype(np.float32) / 255.0
        return {"image": arr}

    def state_dict(self) -> dict[str, Any]:
        return {"type": "image", "image_size": self.image_size}

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> ImagePreprocessor:
        return cls(s["image_size"])


# --------------------------------------------------------------------------- targets


class LabelEncoder:
    def __init__(self, classes: list[str] | None = None):
        self.classes = list(classes or [])
        self._index = {c: i for i, c in enumerate(self.classes)}

    def encode(self, labels: list[Any]) -> np.ndarray:
        return np.fromiter((self._index[str(v)] for v in labels), dtype=np.int64)

    def decode(self, idx: np.ndarray) -> list[str]:
        return [self.classes[int(i)] for i in idx]

    def state_dict(self) -> dict[str, Any]:
        return {"classes": self.classes}


class TargetScaler:
    """Regression target transform: optional log1p, then standardise."""

    def __init__(self, transform: str = "standard"):
        self.transform_name = transform
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y: np.ndarray) -> TargetScaler:
        z = self._forward_raw(np.asarray(y, dtype=np.float64))
        if self.transform_name in ("standard", "log1p"):
            self.mean = float(z.mean())
            sd = float(z.std())
            self.std = sd if sd > 1e-12 else 1.0
        return self

    def _forward_raw(self, y: np.ndarray) -> np.ndarray:
        if self.transform_name == "log1p":
            if (y < 0).any():
                raise ValueError("target_transform=log1p requires non-negative targets")
            return np.log1p(y)
        return y

    def encode(self, y: np.ndarray) -> np.ndarray:
        return ((self._forward_raw(np.asarray(y, dtype=np.float64)) - self.mean) / self.std).astype(
            np.float32
        )

    def decode(self, z: np.ndarray) -> np.ndarray:
        y = np.asarray(z, dtype=np.float64) * self.std + self.mean
        if self.transform_name == "log1p":
            y = np.expm1(y)
        return y

    def state_dict(self) -> dict[str, Any]:
        return {"transform": self.transform_name, "mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> TargetScaler:
        obj = cls(s["transform"])
        obj.mean, obj.std = s["mean"], s["std"]
        return obj
