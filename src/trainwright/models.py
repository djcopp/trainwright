"""Model registry and the built-in small-model zoo.

Every model takes ``spec`` (from the DataModule: input sizes, number of outputs) plus keyword
params, and its ``forward`` takes the batch keys as keyword arguments:

* tabular: ``forward(x_num, x_cat)``
* image:   ``forward(image)`` with ``image`` float in [0, 1], shape [B, 3, H, W]
* text:    ``forward(tokens)`` int64 [B, L], 0 = padding

Register your own architecture in the project's ``models.py``::

    import trainwright as tw

    @tw.register_model("my_net", modality="tabular", description="...")
    class MyNet(torch.nn.Module):
        def __init__(self, spec, hidden=128):
            ...
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class ModelInfo:
    name: str
    modality: str
    builder: Callable[..., nn.Module]
    description: str = ""
    when_to_use: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)
    recipe: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modality": self.modality,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "params": self.defaults,
            "recipe": self.recipe,
        }


_REGISTRY: dict[str, ModelInfo] = {}


def register_model(
    name: str,
    modality: str,
    description: str = "",
    when_to_use: str = "",
    recipe: dict[str, Any] | None = None,
) -> Callable[[Any], Any]:
    """Register an architecture. ``recipe`` holds dotted overrides (e.g. ``{"optim.lr": 3e-5}``)
    applied automatically whenever an experiment switches to this model."""

    def deco(obj: Any) -> Any:
        sig = inspect.signature(obj)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if k != "spec" and v.default is not inspect.Parameter.empty
        }
        _REGISTRY[name] = ModelInfo(
            name, modality, obj, description, when_to_use, defaults, dict(recipe or {})
        )
        return obj

    return deco


def get_model_info(name: str) -> ModelInfo:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def build_model(name: str, spec: dict[str, Any], params: dict[str, Any] | None = None) -> nn.Module:
    info = get_model_info(name)
    params = dict(params or {})
    unknown = set(params) - set(info.defaults)
    if unknown:
        raise ValueError(
            f"Unknown params for model {name!r}: {sorted(unknown)}. Valid params: {info.defaults}"
        )
    if spec.get("modality") and info.modality != spec["modality"]:
        raise ValueError(
            f"Model {name!r} is for {info.modality} data but the project is {spec['modality']}"
        )
    return info.builder(spec, **params)


def list_models(modality: str | None = None) -> list[dict[str, Any]]:
    return [m.as_dict() for m in _REGISTRY.values() if modality in (None, m.modality)]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[name]


def _norm(name: str, dim: int) -> nn.Module:
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    return nn.Identity()


# --------------------------------------------------------------------------- tabular


class PeriodicEmbedding(nn.Module):
    """Periodic numeric embeddings (Gorishniy et al., 2022, "PLR"): x -> ReLU(W [sin, cos](2 pi c x))."""

    def __init__(
        self, n_features: int, n_frequencies: int = 16, sigma: float = 0.1, d_out: int = 16
    ):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        self.weight = nn.Parameter(torch.empty(n_features, 2 * n_frequencies, d_out))
        self.bias = nn.Parameter(torch.zeros(n_features, d_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, F] -> [B, F * d_out]
        v = 2 * math.pi * self.c[None] * x[..., None]
        v = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # [B, F, 2k]
        out = torch.relu(torch.einsum("bfk,fkd->bfd", v, self.weight) + self.bias)
        return out.flatten(1)


class _Block(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float, norm: str, act: str, residual: bool):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.norm = _norm(norm, d_out)
        self.act = _act(act)
        self.drop = nn.Dropout(dropout)
        self.residual = residual and d_in == d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.norm(self.lin(x))))
        return x + h if self.residual else h


@register_model(
    "tabular_mlp",
    modality="tabular",
    description="MLP with categorical embeddings, optional periodic numeric embeddings, residual blocks.",
    when_to_use="Default for tables. Usually matches or beats boosted trees given quantile-normalised "
    "inputs, tuned weight decay and early stopping, especially with >5k rows. Try num_embedding=periodic "
    "for a strong upgrade on numeric-heavy data.",
)
class TabularMLP(nn.Module):
    def __init__(
        self,
        spec: dict[str, Any],
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        norm: str = "batchnorm",
        activation: str = "relu",
        residual: bool = True,
        num_embedding: str = "none",
        n_frequencies: int = 16,
        frequency_sigma: float = 0.1,
        num_embed_dim: int = 16,
        cat_embed_dim: int = 0,
        k_ensemble: int = 1,
    ):
        super().__init__()
        self.k = int(k_ensemble)
        self.classification = spec.get("task", "classification") == "classification"
        n_num = int(spec.get("n_num", 0))
        cards = list(spec.get("cat_cardinalities", []))
        self.num_embed: nn.Module | None = None
        d_num = n_num
        if num_embedding == "periodic" and n_num > 0:
            self.num_embed = PeriodicEmbedding(n_num, n_frequencies, frequency_sigma, num_embed_dim)
            d_num = n_num * num_embed_dim
        elif num_embedding not in ("none", "periodic"):
            raise ValueError("num_embedding must be 'none' or 'periodic'")
        dims = [cat_embed_dim or min(32, max(2, round(1.6 * c**0.56))) for c in cards]
        self.embeds = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, dims)])
        d_in = d_num + sum(dims)
        if d_in == 0:
            raise ValueError(
                "No usable features (all columns dropped?). Check `trainwright profile`."
            )
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(depth):
            layers.append(_Block(d, hidden, dropout, norm, activation, residual))
            d = hidden
        self.body = nn.Sequential(*layers)
        n_out = int(spec.get("n_outputs", 1))
        if self.k > 1:
            # TabM-mini (Gorishniy et al., 2024): k members share the MLP; each member scales the
            # input by its own random-sign adapter and has its own head. Trained with the mean of the
            # members' losses, predicting with their average: an ensemble for ~1 model's parameters.
            g = torch.Generator().manual_seed(0)
            signs = torch.randint(0, 2, (self.k, d_in), generator=g).float() * 2 - 1
            self.adapter = nn.Parameter(signs)
            self.head_w = nn.Parameter(torch.empty(self.k, d, n_out))
            self.head_b = nn.Parameter(torch.zeros(self.k, n_out))
            nn.init.uniform_(self.head_w, -1 / math.sqrt(d), 1 / math.sqrt(d))  # nn.Linear's init
        else:
            self.head = nn.Linear(d, n_out)

    def aggregate(self, out: torch.Tensor) -> torch.Tensor:
        """[B, k, out] member outputs -> [B, out] ensemble output (log of mean probs / mean)."""
        if self.classification and out.shape[-1] > 1:
            return torch.logsumexp(torch.log_softmax(out, dim=-1), dim=1) - math.log(out.shape[1])
        if self.classification:  # single-logit binary: average probabilities, back to a logit
            p = torch.sigmoid(out).mean(dim=1).clamp(1e-7, 1 - 1e-7)
            return torch.log(p) - torch.log1p(-p)
        return out.mean(dim=1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = []
        if x_num.shape[1] > 0:
            parts.append(self.num_embed(x_num) if self.num_embed is not None else x_num)
        for j, emb in enumerate(self.embeds):
            parts.append(emb(x_cat[:, j]))
        x = torch.cat(parts, dim=1)
        if self.k == 1:
            return self.head(self.body(x))
        b = x.shape[0]
        h = self.body((x[:, None, :] * self.adapter[None]).reshape(b * self.k, -1))
        out = torch.einsum("bkd,kdo->bko", h.view(b, self.k, -1), self.head_w) + self.head_b
        return out if self.training else self.aggregate(out)


# --------------------------------------------------------------------------- image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std  # type: ignore[operator]


_BACKBONES = {
    # name: (torchvision builder, weights enum name, feature attr to strip)
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "efficientnet_b2": ("efficientnet_b2", "EfficientNet_B2_Weights"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    "regnet_y_400mf": ("regnet_y_400mf", "RegNet_Y_400MF_Weights"),
}


def _strip_head(m: nn.Module) -> tuple[nn.Module, int]:
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):  # resnet / regnet
        d = m.fc.in_features
        m.fc = nn.Identity()
        return m, d
    if hasattr(m, "classifier"):
        cls = m.classifier
        if isinstance(cls, nn.Sequential):
            linears = [i for i, layer in enumerate(cls) if isinstance(layer, nn.Linear)]
            first = linears[0]
            d = int(cls[first].in_features)  # type: ignore[arg-type]
            # keep everything before the first Linear (e.g. convnext's LayerNorm2d + Flatten)
            m.classifier = (
                nn.Sequential(*list(cls.children())[:first]) if first > 0 else nn.Identity()
            )
            if (
                isinstance(m.classifier, nn.Sequential)
                and len(m.classifier)
                and not any(isinstance(layer, nn.Flatten) for layer in m.classifier)
            ):
                m.classifier.append(nn.Flatten(1))
            return m, d
    raise ValueError(f"Don't know how to strip the head of {type(m).__name__}")


class _HFVisionBackbone(nn.Module):
    """A Hugging Face ViT-style encoder (e.g. DINOv2) -> [CLS, mean(patch tokens)] features."""

    def __init__(self, repo: str, pretrained: bool):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.net = (
            AutoModel.from_pretrained(repo)
            if pretrained
            else AutoModel.from_config(AutoConfig.from_pretrained(repo))
        )
        self.out_dim = 2 * int(self.net.config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)


_HF_BACKBONES = {"dinov2_small": "facebook/dinov2-small", "dinov2_base": "facebook/dinov2-base"}


@register_model(
    "image_pretrained",
    modality="image",
    description="ImageNet-pretrained torchvision backbone + new linear head, fine-tuned end-to-end "
    "with a lower backbone LR (optim.backbone_lr_mult).",
    when_to_use="Default for photos. resnet18 is a robust start; dinov2_small (self-supervised ViT, "
    "excellent features; try freeze_backbone=true first on small datasets); mobilenet_v3_large/"
    "efficientnet_b0 for cheap CPU serving; convnext_tiny when accuracy matters more than latency.",
)
class ImagePretrained(nn.Module):
    def __init__(
        self,
        spec: dict[str, Any],
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if backbone in _HF_BACKBONES:
            try:
                hf = _HFVisionBackbone(_HF_BACKBONES[backbone], pretrained)
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    f"{backbone} needs transformers: `uv add 'trainwright[text]'`"
                ) from e
            self.backbone: nn.Module = hf
            d = hf.out_dim
        else:
            try:
                import torchvision.models as tvm
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "image models need torchvision: `uv add 'trainwright[vision]'`"
                ) from e
            if backbone not in _BACKBONES:
                raise ValueError(
                    f"backbone must be one of {sorted(_BACKBONES) + sorted(_HF_BACKBONES)}"
                )
            fn_name, weights_name = _BACKBONES[backbone]
            weights = getattr(tvm, weights_name).DEFAULT if pretrained else None
            net = getattr(tvm, fn_name)(weights=weights)
            self.backbone, d = _strip_head(net)
        self.pretrained_backbone = pretrained
        self.frozen = freeze_backbone
        if freeze_backbone:  # linear probe: only the head trains (fast, strong on small data)
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, int(spec.get("n_outputs", 1))))
        # frozen backbone + unaugmented inputs -> each image's features never change, so compute
        # them once (keyed by a random projection of the pixels) and train the head on the cache
        self.cache_features = freeze_backbone
        self._cache: dict[tuple[int, int], torch.Tensor] = {}
        self._proj: torch.Tensor | None = None

    def train(self, mode: bool = True) -> ImagePretrained:
        super().train(mode)
        if self.frozen:
            self.backbone.eval()  # frozen means frozen: no BatchNorm/dropout updates either
        return self

    def _keys(self, x: torch.Tensor) -> list[tuple[int, int]]:
        flat = x.flatten(1).float()
        if (
            self._proj is None
            or self._proj.shape[0] != flat.shape[1]
            or self._proj.device != x.device
        ):
            g = torch.Generator().manual_seed(1234)
            self._proj = torch.randn(flat.shape[1], 2, generator=g).to(x.device)
        with torch.autocast(x.device.type, enabled=False):
            k = (flat @ self._proj).mul(1e4).round().long().cpu().tolist()
        return [(a, b) for a, b in k]

    def features(self, image: torch.Tensor) -> torch.Tensor:
        x = self.normalize(image)
        if not self.frozen:
            return self.backbone(x)
        tracing = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        if not self.cache_features or tracing:
            return self.backbone(x)
        keys = self._keys(image)
        miss = [i for i, k in enumerate(keys) if k not in self._cache]
        if miss:
            with torch.no_grad():
                f = self.backbone(x[miss]).float()
            for i, row in zip(miss, f):
                self._cache[keys[i]] = row.detach()
        return torch.stack([self._cache[k] for k in keys]).to(x.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image))


@register_model(
    "image_cnn",
    modality="image",
    description="Small from-scratch CNN (conv-bn-relu stages + global pooling).",
    when_to_use="Tiny or non-photographic images (icons, scans, spectrograms, <=64px) where ImageNet "
    "features don't transfer, or when the model must be a few hundred KB.",
    recipe={
        "optim.lr": 3e-3,
        "optim.weight_decay": 1e-4,
        "train.max_epochs": 40,
        "train.patience": 10,
        "train.label_smoothing": 0.0,
    },
)
class ImageCNN(nn.Module):
    def __init__(
        self, spec: dict[str, Any], width: int = 32, stages: int = 3, dropout: float = 0.2
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 3
        for s in range(stages):
            c = width * (2**s)
            layers += [
                nn.Conv2d(c_in, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        self.backbone = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(c_in, int(spec.get("n_outputs", 1)))
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(self.normalize(image)))


# --------------------------------------------------------------------------- text


@register_model(
    "text_bag",
    modality="text",
    description="fastText-style model: embeddings of word uni/bi-grams, masked mean+max pooling, MLP head.",
    when_to_use="Default for short-to-medium text classification/regression (intents, tickets, reviews). "
    "Trains in seconds on CPU, serves in microseconds.",
)
class TextBag(nn.Module):
    def __init__(
        self,
        spec: dict[str, Any],
        embed_dim: int = 128,
        hidden: int = 256,
        dropout: float = 0.3,
        pooling: str = "meanmax",
    ):
        super().__init__()
        self.embed = nn.Embedding(int(spec["vocab_size"]), embed_dim, padding_idx=0)
        self.pooling = pooling
        d = embed_dim * (2 if pooling == "meanmax" else 1)
        self.body = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(hidden, int(spec.get("n_outputs", 1)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != 0).unsqueeze(-1).float()
        e = self.embed(tokens)
        denom = mask.sum(1).clamp(min=1.0)
        mean = (e * mask).sum(1) / denom
        if self.pooling == "meanmax":
            mx = (e + (mask - 1) * 1e4).amax(1)
            mx = torch.where(mask.sum(1) > 0, mx, torch.zeros_like(mx))
            h = torch.cat([mean, mx], dim=1)
        else:
            h = mean
        return self.head(self.body(h))


@register_model(
    "text_cnn",
    modality="text",
    description="Embeddings + parallel 1D convolutions (kernel 2/3/5) + max pooling (Kim, 2014).",
    when_to_use="When word order/local phrases matter more than bag-of-ngrams captures.",
)
class TextCNN(nn.Module):
    def __init__(
        self, spec: dict[str, Any], embed_dim: int = 128, channels: int = 128, dropout: float = 0.3
    ):
        super().__init__()
        self.embed = nn.Embedding(int(spec["vocab_size"]), embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, channels, k, padding=k // 2) for k in (2, 3, 5)]
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(3 * channels, int(spec.get("n_outputs", 1)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != 0).unsqueeze(1)
        x = self.embed(tokens).transpose(1, 2)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))[..., : tokens.shape[1]]
            h = h.masked_fill(~mask, -1e4).amax(-1)
            feats.append(h)
        return self.head(self.drop(torch.cat(feats, dim=1)))


# --------------------------------------------------------------------------- pretrained text


@register_model(
    "text_pretrained",
    modality="text",
    description="Small pretrained transformer encoder (default bge-small-en-v1.5, 33M params) + mean pooling + "
    "linear head, fine-tuned end-to-end. Needs `trainwright[text]`.",
    when_to_use="Best accuracy for text classification when wording varies (paraphrases, few examples "
    "per class). ~5-20 ms per message on CPU; still orders of magnitude cheaper than an LLM call.",
    recipe={
        "optim.lr": 1e-3,
        "optim.backbone_lr_mult": 0.05,
        "optim.weight_decay": 0.01,
        "train.max_epochs": 8,
        "train.batch_size": 32,
        "train.patience": 3,
        "train.label_smoothing": 0.0,
        "train.grad_clip": 1.0,
    },
)
class TextPretrained(nn.Module):
    def __init__(
        self,
        spec: dict[str, Any],
        encoder: str = "BAAI/bge-small-en-v1.5",
        dropout: float = 0.1,
        max_length: int = 0,
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "text_pretrained needs transformers: `uv add 'trainwright[text]'`"
            ) from e
        if pretrained:
            self.backbone = AutoModel.from_pretrained(encoder)
        else:
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(encoder))
        self.pretrained_backbone = pretrained
        self.pad_id = int(spec.get("pad_id", 0))
        d = int(self.backbone.config.hidden_size)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, int(spec.get("n_outputs", 1))))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != self.pad_id).long()
        h = self.backbone(input_ids=tokens, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)


@register_model(
    "image_probe",
    modality="image",
    description="Linear probe: frozen self-supervised backbone (DINOv2-small by default) + linear head. "
    "Only the head trains, so runs take seconds per epoch.",
    when_to_use="First model to try on small image datasets (up to a few thousand images); fine-tune with "
    "image_pretrained afterwards if it plateaus.",
    recipe={
        "optim.lr": 1e-2,
        "optim.weight_decay": 1e-4,
        "train.label_smoothing": 0.0,
        "train.max_epochs": 60,
        "train.patience": 15,
        "preprocess.augment": "none",  # lets the frozen features be computed once and cached
    },
)
class ImageProbe(ImagePretrained):
    def __init__(self, spec: dict[str, Any], backbone: str = "dinov2_small", dropout: float = 0.0):
        super().__init__(
            spec, backbone=backbone, pretrained=True, dropout=dropout, freeze_backbone=True
        )
