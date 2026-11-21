"""Export a run into a self-contained, deployable bundle.

::

    artifacts/<run>/
        bundle.json          # preprocessing state, classes, metrics, provenance
        model.onnx           # (neural runs)  or model.joblib (baseline runs)
        tw_runtime/          # numpy-only preprocessing + onnxruntime Predictor
        server.py            # FastAPI app: /health /metadata /predict
        requirements.txt
        Dockerfile
        MODEL_CARD.md
        example_request.json

The bundle runs without torch or trainwright: ``pip install -r requirements.txt && uvicorn server:app``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ._utils import now_iso, read_json, write_json
from .pipeline import LifecycleError, get_records, load_task_and_data, require_split
from .project import Project
from .tracking import Run

RUNTIME_FILES = ["runtime.py", "preprocessing.py", "featurehook.py"]


class _ExportWrapper(nn.Module):
    """Positional-arg wrapper: absent (zero-width) inputs are synthesised inside the graph,
    classification outputs are softmaxed so the runtime gets probabilities."""

    def __init__(
        self,
        model: nn.Module,
        input_names: list[str],
        absent: dict[str, tuple[int, torch.dtype]],
        classification: bool,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.absent = absent
        self.classification = classification
        self.temperature = float(temperature)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        kwargs = dict(zip(self.input_names, args))
        b = args[0].shape[0]
        for name, (width, dtype) in self.absent.items():
            kwargs[name] = torch.zeros(b, width, dtype=dtype, device=args[0].device)
        out = self.model(**kwargs)
        if self.classification:
            return torch.softmax(out.float() / self.temperature, dim=-1)
        return out.float()


def _example_inputs(dm: Any, n: int = 4) -> dict[str, torch.Tensor]:
    loader = dm.dataloader("val", train=False)
    batch = next(iter(loader))
    return {k: v[:n].cpu() for k, v in batch.items() if k != "y"}


def export_onnx(task: Any, dm: Any, path: Path) -> dict[str, Any]:
    model = task.model.cpu().eval()
    example = _example_inputs(dm)
    present = [k for k, v in example.items() if v.ndim < 2 or v.shape[1] > 0]
    absent = {k: (0, v.dtype) for k, v in example.items() if k not in present}
    wrapper = _ExportWrapper(
        model,
        present,
        absent,
        dm.task == "classification",
        float(getattr(task, "temperature", 1.0)),
    ).eval()
    args = tuple(example[k] for k in present)
    with torch.no_grad():
        ref = wrapper(*args).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    import contextlib
    import io
    import warnings

    errors = []
    exported = False
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        try:
            batch = torch.export.Dim("batch", min=1, max=65536)
            torch.onnx.export(
                wrapper,
                args,
                str(path),
                input_names=present,
                output_names=["output"],
                dynamic_shapes={"args": tuple({0: batch} for _ in present)},
                external_data=sum(p.numel() for p in model.parameters()) * 4 > 1.8e9,
                dynamo=True,
                verbose=False,
            )
            exported = True
        except Exception as e:  # fall back to the TorchScript-based exporter
            errors.append(f"dynamo export failed: {type(e).__name__}: {str(e)[:300]}")
        if not exported:
            torch.onnx.export(
                wrapper,
                args,
                str(path),
                input_names=present,
                output_names=["output"],
                dynamic_axes={k: {0: "batch"} for k in [*present, "output"]},
                opset_version=17,
                dynamo=False,
            )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: example[k].numpy() for k in present})[0]
    max_diff = float(np.abs(got - ref).max())
    return {
        "inputs": present,
        "parity_max_abs_diff": max_diff,
        "parity_ok": max_diff < 1e-3,
        "export_warnings": errors,
    }


def export_run(
    project: Project, run: Run, out_dir: Path | None = None, force: bool = False
) -> dict[str, Any]:
    cfg = project.config
    meta, summary = run.meta, run.summary
    if meta.get("status") != "completed":
        raise LifecycleError(
            f"Run {run.id} is {meta.get('status')}; only completed runs can be exported."
        )
    out = out_dir or (project.artifacts_dir / run.id)
    if out.exists():
        if not force:
            raise LifecycleError(f"{out} exists. Pass --force to overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    split = require_split(project, get_records(project))
    test_eval = run.eval_result("test")
    bundle: dict[str, Any] = {
        "format": "trainwright-bundle/1",
        "run": run.id,
        "project": cfg.name,
        "modality": cfg.modality,
        "task": cfg.task,
        "classes": split.get("classes"),
        "primary_metric": cfg.metric,
        "val_metrics": summary.get("val"),
        "test_metrics": test_eval["metrics"] if test_eval else None,
        "exported_at": now_iso(),
        "split_hash": split["split_hash"],
        "data_fingerprint": split["data_fingerprint"],
        "git": meta.get("git"),
        "hypothesis": meta.get("hypothesis"),
    }
    if meta.get("kind") == "baseline" or (run.dir / "model.joblib").exists():
        if cfg.modality == "image":
            raise LifecycleError(
                "Image baselines are reference points only and cannot be exported."
            )
        import joblib

        blob = joblib.load(run.dir / "model.joblib")
        shutil.copy(run.dir / "model.joblib", out / "model.joblib")
        bundle.update(
            kind="sklearn",
            model=meta.get("model"),
            columns=blob.get("columns"),
            baseline=meta.get("model", "").split(":")[-1],
            preprocessor={
                "features": {"type": cfg.modality},
                "feature_hook": blob.get("feature_hook"),
            },
        )
        reqs = ["numpy", "scikit-learn", "joblib", "pandas", "fastapi", "uvicorn"]
    else:
        task, dm, ckpt = load_task_and_data(project, run, splits=("train", "val"), device="cpu")
        info = export_onnx(task, dm, out / "model.onnx")
        bundle.update(
            kind="onnx",
            model=ckpt["model_name"],
            model_params=ckpt["model_params"],
            n_params=summary.get("n_params"),
            preprocessor=ckpt["preprocessor"],
            spec=ckpt["spec"],
            **info,
        )
        reqs = (
            ["numpy", "onnxruntime", "fastapi", "uvicorn"]
            + (["pillow", "python-multipart"] if cfg.modality == "image" else [])
            + (["tokenizers"] if ckpt["preprocessor"]["features"].get("type") == "hf_text" else [])
            + (["pandas"] if ckpt["preprocessor"].get("feature_hook") else [])
        )
    rt = out / "tw_runtime"
    rt.mkdir()
    src = Path(__file__).parent
    for f in RUNTIME_FILES:
        shutil.copy(src / f, rt / f)
    (rt / "__init__.py").write_text('"""Vendored trainwright inference runtime."""\n')
    write_json(out / "bundle.json", bundle)
    (out / "server.py").write_text(SERVER_TEMPLATE)
    (out / "requirements.txt").write_text("\n".join(reqs) + "\n")
    (out / "Dockerfile").write_text(DOCKERFILE)
    (out / "MODEL_CARD.md").write_text(model_card(project, run, bundle))
    write_json(out / "example_request.json", {"instances": _example_instances(project, split)})
    # smoke-test the bundle exactly as a deployment would use it
    from .runtime import Predictor

    pred = Predictor(out)
    example = read_json(out / "example_request.json")["instances"]
    bundle["smoke_test"] = {"ok": True, "example_prediction": pred.predict(example[:1])[0]}
    write_json(out / "bundle.json", bundle)
    return {
        "dir": str(out),
        **{
            k: bundle.get(k)
            for k in ("kind", "model", "parity_ok", "parity_max_abs_diff", "smoke_test")
        },
    }


def _example_instances(project: Project, split: dict[str, Any], n: int = 2) -> list[Any]:
    records = get_records(project)
    recs = records.subset(split["val"][:n])
    if project.config.modality == "tabular":
        assert recs.table is not None
        rows = recs.table.astype(object).where(recs.table.notna(), None).to_dict(orient="records")
        return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in rows]
    if project.config.modality == "text":
        return list(recs.texts or [])
    return [str(p) for p in (recs.paths or [])]


def model_card(project: Project, run: Run, bundle: dict[str, Any]) -> str:
    cfg = project.config
    m = cfg.metric
    val = bundle.get("val_metrics") or {}
    test = bundle.get("test_metrics")
    split = require_split(project, get_records(project))
    lines = [
        f"# Model card: {cfg.name} / {run.id}",
        "",
        f"- **Task**: {cfg.task} on {cfg.modality} data (`{cfg.data.target}`)",
        f"- **Model**: `{bundle.get('model')}` {('(' + format(bundle['n_params'], ',') + ' parameters)') if bundle.get('n_params') else ''}",
        f"- **Primary metric**: {m}",
        f"- **Validation {m}**: {val.get(m, float('nan')):.4f}",
        f"- **Test {m}**: {test[m]:.4f}"
        if test
        else "- **Test**: not evaluated (run `trainwright evaluate <run> --split test`)",
        f"- **Data**: {sum(split['counts'].values())} examples, split {split['counts']} ({split['strategy']}, seed {split['seed']})",
        f"- **Data fingerprint**: `{split['data_fingerprint'][:16]}`, split hash `{split['split_hash']}`",
        f"- **Exported**: {bundle['exported_at']}; git `{(bundle.get('git') or {}).get('sha') or 'n/a'}`",
        "",
        "## All validation metrics",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in val.items() if isinstance(v, (int, float))],
        "",
        "## Intended use and limitations",
        "",
        "- Trained only on the data above; inputs outside that distribution (new categories, other cameras, "
        "other languages) may degrade silently. Monitor confidence and accuracy in production.",
        "- For classification, `confidence` is the softmax probability. Check calibration (ECE in the metrics) "
        "before routing on it; `trainwright evaluate` reports accuracy at confidence thresholds.",
        "",
        "## Serving",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "uvicorn server:app --host 0.0.0.0 --port 8000",
        "curl -s localhost:8000/predict -H 'content-type: application/json' -d @example_request.json",
        "```",
        "",
        "Or `docker build -t model . && docker run -p 8000:8000 model`.",
        "",
    ]
    return "\n".join(lines)


SERVER_TEMPLATE = '''"""FastAPI server for a trainwright bundle. Generated; safe to edit."""

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from tw_runtime.runtime import Predictor

BUNDLE = Path(os.environ.get("TW_BUNDLE", Path(__file__).parent))
predictor = Predictor(BUNDLE)
app = FastAPI(title=f"{predictor.meta['project']} ({predictor.meta['run']})")


class PredictRequest(BaseModel):
    instances: list[Any]
    top_k: int = 3


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    keep = ["project", "run", "modality", "task", "classes", "primary_metric", "val_metrics", "test_metrics", "model"]
    return {k: predictor.meta.get(k) for k in keep}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if not req.instances:
        raise HTTPException(400, "instances must be a non-empty list")
    try:
        return {"predictions": predictor.predict(req.instances, top_k=req.top_k)}
    except (KeyError, ValueError) as e:
        raise HTTPException(422, f"bad input: {e}") from e


if predictor.modality == "image":

    @app.post("/predict/image")
    async def predict_image(file: UploadFile = File(...)) -> dict[str, Any]:
        data = await file.read()
        return {"predictions": predictor.predict([data])}
'''

DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
"""
