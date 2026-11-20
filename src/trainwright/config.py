"""Typed, validated configuration.

Two layers:

* ``ProjectConfig`` (``trainwright.yaml``): what the problem *is* — data location, target,
  task type, modality, split policy and the single primary metric. Changes rarely.
* ``ExperimentConfig``: how one run trains — model, optimiser, schedule, preprocessing.
  Every run stores its fully-resolved ``ExperimentConfig`` so it can be reproduced exactly.

All models forbid unknown keys, so a typo in an override fails loudly instead of being
silently ignored (a common failure mode for agents).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

TaskType = Literal["classification", "regression"]
Modality = Literal["tabular", "image", "text"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------- project


class DataConfig(_Strict):
    path: str = Field(
        description="CSV/Parquet file, image folder (one sub-folder per class), or CSV manifest."
    )
    target: str | None = Field(
        "label", description="Target column. For class-per-folder images this is ignored."
    )
    text_column: str | None = Field(None, description="Text column (text modality).")
    image_column: str | None = Field(
        None, description="Image path column for CSV manifests (image modality)."
    )
    drop: list[str] = Field(
        default_factory=list, description="Columns never used as features (ids, leaks)."
    )
    group_column: str | None = Field(
        None, description="Rows sharing a group never straddle splits."
    )
    time_column: str | None = Field(None, description="Split chronologically on this column.")


class SplitConfig(_Strict):
    strategy: Literal["auto", "random", "stratified", "group", "time"] = "auto"
    val: float = Field(0.15, gt=0, lt=0.5)
    test: float = Field(0.15, gt=0, lt=0.5)
    seed: int = 42


class CostConfig(_Strict):
    """Assumptions used by ``trainwright benchmark`` to compare serving costs.

    These are *assumptions*, not facts: edit them to match your provider and contract.
    """

    instance_name: str = "2 vCPU / 4 GB general-purpose cloud VM"
    instance_usd_per_hour: float = 0.05
    instance_utilisation: float = Field(
        0.5, gt=0, le=1, description="Fraction of peak throughput you can sustain."
    )
    cpu_slowdown_factor: float = Field(
        2.0,
        ge=1,
        description="Benchmark machine vs cloud vCPU speed ratio (laptops are usually faster).",
    )
    llm_name: str = "claude-sonnet-5-5"
    llm_usd_per_mtok_input: float = 2.0
    llm_usd_per_mtok_output: float = 10.0
    llm_input_tokens_per_call: int = Field(
        1800,
        description="Prompt + input. Images cost about width*height/750 tokens (~1600 for 1092x1092).",
    )
    llm_output_tokens_per_call: int = 30
    llm_latency_s: float = 2.0
    pricing_note: str = "LLM list prices as of 2026-11; edit to match your contract."


class ProjectConfig(_Strict):
    name: str
    task: TaskType
    modality: Modality
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    metric: str = Field(description="Primary metric used for model selection.")
    metric_mode: Literal["max", "min"] = "max"
    description: str = ""
    cost: CostConfig = Field(default_factory=CostConfig)


# --------------------------------------------------------------------------- experiment


class ModelConfig(_Strict):
    name: str = Field(description="Registered model name, see `trainwright models`.")
    params: dict[str, Any] = Field(default_factory=dict)


class OptimConfig(_Strict):
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    momentum: float = 0.9
    scheduler: Literal["none", "cosine", "onecycle", "plateau"] = "cosine"
    warmup_epochs: float = Field(1.0, ge=0)
    backbone_lr_mult: float = Field(
        0.1, gt=0, description="LR multiplier for pretrained backbones."
    )


class TrainConfig(_Strict):
    max_epochs: int = Field(50, ge=1)
    batch_size: int = Field(256, ge=1)
    patience: int | None = Field(
        10, description="Early-stopping patience in epochs (None disables)."
    )
    grad_clip: float | None = 1.0
    accumulate: int = Field(1, ge=1)
    precision: Literal["auto", "32", "bf16", "16"] = "auto"
    device: str = "auto"
    seed: int = 0
    deterministic: bool = False
    time_budget: str | None = Field(None, description="Stop gracefully after e.g. '10m', '1h'.")
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    class_weights: Literal["none", "balanced"] = "none"
    num_workers: int = 0
    freeze_backbone_epochs: int = Field(
        0, ge=0, description="Train only the head for the first N epochs."
    )
    eval_train_metrics: bool = True


class PreprocessConfig(_Strict):
    # tabular
    numeric: Literal["quantile", "standard", "none"] = "quantile"
    cat_min_freq: int = Field(1, ge=1)
    max_cat_cardinality: int = Field(
        1000, description="Categoricals above this are treated as ids and dropped."
    )
    target_transform: Literal["standard", "log1p", "none"] = Field(
        "standard", description="Regression only. log1p helps heavy-tailed positive targets."
    )
    # image
    image_size: int = 224
    augment: Literal["none", "light", "medium", "strong"] = "light"
    hflip: bool = Field(
        True, description="Disable when left/right matters (e.g. left vs right headlight, text)."
    )
    cache_images: Literal["auto", "always", "never"] = Field(
        "auto", description="Decode+resize once into a disk-backed cache reused across runs."
    )
    # text
    max_vocab: int = 50000
    ngrams: int = Field(2, ge=1, le=3)
    max_tokens: int = 256
    lowercase: bool = True
    min_token_freq: int = 2


class ExperimentConfig(_Strict):
    model: ModelConfig
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)


# --------------------------------------------------------------------------- defaults


def default_experiment(project: ProjectConfig) -> ExperimentConfig:
    """Strong, boring defaults per modality. These are where every project starts."""
    if project.modality == "tabular":
        return ExperimentConfig(
            model=ModelConfig(
                name="tabular_mlp", params={"hidden": 256, "depth": 3, "dropout": 0.1}
            ),
            optim=OptimConfig(lr=1e-3, weight_decay=1e-5, scheduler="cosine", warmup_epochs=1),
            train=TrainConfig(max_epochs=100, batch_size=256, patience=15),
            preprocess=PreprocessConfig(numeric="quantile"),
        )
    if project.modality == "image":
        return ExperimentConfig(
            model=ModelConfig(
                name="image_pretrained", params={"backbone": "resnet18", "dropout": 0.2}
            ),
            optim=OptimConfig(
                lr=1e-3,
                weight_decay=0.05,
                scheduler="cosine",
                warmup_epochs=1,
                backbone_lr_mult=0.1,
            ),
            train=TrainConfig(max_epochs=15, batch_size=64, patience=5, label_smoothing=0.1),
            preprocess=PreprocessConfig(image_size=224, augment="light"),
        )
    if project.modality == "text":
        return ExperimentConfig(
            model=ModelConfig(
                name="text_bag", params={"embed_dim": 128, "hidden": 256, "dropout": 0.3}
            ),
            optim=OptimConfig(lr=3e-3, weight_decay=1e-4, scheduler="cosine", warmup_epochs=1),
            train=TrainConfig(max_epochs=30, batch_size=128, patience=6),
            preprocess=PreprocessConfig(ngrams=2, max_vocab=50000),
        )
    raise ValueError(f"Unknown modality {project.modality}")


# --------------------------------------------------------------------------- overrides


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``a.b.c=value``; value is YAML so ``1e-3``, ``true``, ``[64,128]`` all work."""
    if "=" not in text:
        raise ValueError(f"Override {text!r} must look like key=value, e.g. optim.lr=3e-4")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override {text!r} has an empty key")
    value = yaml.safe_load(raw) if raw.strip() != "" else None
    # YAML parses '1e-3' as a string; coerce scientific notation to float.
    if isinstance(value, str):
        try:
            value = float(value) if any(c in value.lower() for c in "e.") else int(value)
        except ValueError:
            pass
    return key, value


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(data)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = out
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return out


def deep_merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in top.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "params":
            out[k] = deep_merge(out[k], v)
        elif isinstance(v, dict) and k == "params" and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_experiment(
    project: ProjectConfig,
    config_file: Path | None = None,
    overrides: list[str] | dict[str, Any] | None = None,
    base: ExperimentConfig | None = None,
) -> ExperimentConfig:
    """defaults (or ``base``) <- config file <- overrides, validated."""
    data = (base or default_experiment(project)).model_dump()
    if config_file is not None:
        file_data = yaml.safe_load(Path(config_file).read_text()) or {}
        # switching model family resets params rather than merging foreign keys
        if "model" in file_data and file_data["model"].get("name") not in (
            None,
            data["model"]["name"],
        ):
            data["model"]["params"] = {}
        data = deep_merge(data, file_data)
    if overrides:
        ov = (
            dict(parse_override(o) for o in overrides) if isinstance(overrides, list) else overrides
        )
        if "model.name" in ov and ov["model.name"] != data["model"]["name"]:
            data["model"]["params"] = {}
        data = apply_overrides(data, ov)
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(format_validation_error(e)) from e


class ConfigError(ValueError):
    pass


def format_validation_error(e: ValidationError) -> str:
    lines = ["Invalid configuration:"]
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def dump_yaml(obj: BaseModel | dict[str, Any]) -> str:
    data = obj.model_dump() if isinstance(obj, BaseModel) else obj
    return yaml.safe_dump(data, sort_keys=False)
