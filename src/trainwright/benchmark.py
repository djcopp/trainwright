"""Latency / throughput / cost benchmark of an exported bundle, versus an LLM API call.

Measured numbers come from this machine (CPU, onnxruntime). Cost numbers combine those
measurements with the *assumptions* in ``trainwright.yaml -> cost`` (instance price,
utilisation, CPU slowdown vs this machine, LLM token prices). Every assumption is echoed in
the output so nobody mistakes an estimate for a quote.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ._utils import now_iso, read_json, write_json
from .project import Project


def _time_calls(
    fn: Any, min_iters: int = 20, min_seconds: float = 2.0, max_iters: int = 2000
) -> list[float]:
    fn()  # warm-up
    times: list[float] = []
    t_end = time.perf_counter() + min_seconds
    while (len(times) < min_iters or time.perf_counter() < t_end) and len(times) < max_iters:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def benchmark_bundle(
    project: Project, bundle_dir: Path, batch_size: int = 64, threads: int = 2
) -> dict[str, Any]:
    from .runtime import Predictor

    cfg = project.config
    cost = cfg.cost
    pred = Predictor(bundle_dir, threads=threads)
    meta = read_json(bundle_dir / "bundle.json")
    example = read_json(bundle_dir / "example_request.json")["instances"]
    one = example[:1]
    batch = (example * (batch_size // max(1, len(example)) + 1))[:batch_size]
    if cfg.modality == "image":
        # decode once so we time the model + resize, not disk reads
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in example]
        for im in imgs:
            im.load()
        one = imgs[:1]
        batch = (imgs * (batch_size // len(imgs) + 1))[:batch_size]

    lat = np.array(_time_calls(lambda: pred.predict(one))) * 1000
    tput_times = _time_calls(lambda: pred.predict(batch), min_iters=5, min_seconds=3.0)
    throughput = batch_size / float(np.median(tput_times))
    model_only = None
    if meta.get("kind") == "onnx":
        feats = pred.featurise(batch)
        mt = _time_calls(lambda: pred.session.run(None, feats), min_iters=5, min_seconds=2.0)
        model_only = batch_size / float(np.median(mt))

    model_file = bundle_dir / ("model.onnx" if meta.get("kind") == "onnx" else "model.joblib")
    size_mb = sum(f.stat().st_size for f in bundle_dir.glob(model_file.name + "*")) / 1e6

    eff_tput = throughput / cost.cpu_slowdown_factor * cost.instance_utilisation
    usd_per_m_small = cost.instance_usd_per_hour / (eff_tput * 3600) * 1e6
    llm_per_call = (
        cost.llm_input_tokens_per_call * cost.llm_usd_per_mtok_input
        + cost.llm_output_tokens_per_call * cost.llm_usd_per_mtok_output
    ) / 1e6
    usd_per_m_llm = llm_per_call * 1e6
    p50 = float(np.percentile(lat, 50))
    result = {
        "bundle": str(bundle_dir),
        "run": meta.get("run"),
        "modality": meta.get("modality"),
        "model": meta.get("model"),
        "measured_at": now_iso(),
        "measured_on": {
            "cpu_threads": threads,
            "machine": os.uname().machine if hasattr(os, "uname") else None,
        },
        "model_size_mb": round(size_mb, 3),
        "n_params": meta.get("n_params"),
        "latency_ms_batch1": {
            "p50": round(p50, 3),
            "p95": round(float(np.percentile(lat, 95)), 3),
            "p99": round(float(np.percentile(lat, 99)), 3),
            "n": len(lat),
        },
        "throughput_per_s": round(throughput, 1),
        "model_only_throughput_per_s": round(model_only, 1) if model_only else None,
        "preprocessing_share": round(1 - throughput / model_only, 3) if model_only else None,
        "throughput_batch_size": batch_size,
        "val_metrics": meta.get("val_metrics"),
        "test_metrics": meta.get("test_metrics"),
        "cost_estimate": {
            "small_model_usd_per_1m_predictions": round(usd_per_m_small, 4),
            "llm_usd_per_1m_predictions": round(usd_per_m_llm, 2),
            "llm_usd_per_call": round(llm_per_call, 6),
            "ratio_llm_over_small": round(usd_per_m_llm / max(usd_per_m_small, 1e-9), 1),
            "latency_speedup_vs_llm": round(cost.llm_latency_s * 1000 / max(p50, 1e-6), 1),
            "assumptions": cost.model_dump(),
            "method": "small model: instance $/h / (measured throughput / cpu_slowdown_factor * utilisation * 3600); "
            "LLM: tokens per call x list price. Excludes engineering, storage, egress and retraining.",
        },
    }
    write_json(project.benchmark_path(meta.get("run", bundle_dir.name)), result)
    return result


def _short(p: str) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return p


def format_benchmark(b: dict[str, Any]) -> str:
    c = b["cost_estimate"]
    a = c["assumptions"]
    lines = [
        f"bundle {_short(b['bundle'])}  ({b['model']}, {b['model_size_mb']} MB)",
        f"latency (batch 1, {b['measured_on']['cpu_threads']} CPU threads): p50 {b['latency_ms_batch1']['p50']:.2f} ms, "
        f"p95 {b['latency_ms_batch1']['p95']:.2f} ms",
        f"throughput (batch {b['throughput_batch_size']}): {b['throughput_per_s']:,.0f} predictions/s end-to-end"
        + (
            f" ({b['model_only_throughput_per_s']:,.0f}/s model only; preprocessing is "
            f"{b['preprocessing_share']:.0%} of the time"
            + (
                {
                    "image": ": resize images client-side to cut cost)",
                    "text": ": tokenisation dominates)",
                }.get(b.get("modality") or "", ": features.py / row conversion dominates; batch requests)")
                if (b.get("preprocessing_share") or 0) > 0.5
                else ")"
            )
            if b.get("model_only_throughput_per_s")
            else ""
        ),
        "",
        "cost per 1M predictions (ESTIMATE; edit `cost:` in trainwright.yaml):",
        f"  small model on {a['instance_name']} @ ${a['instance_usd_per_hour']}/h: ${c['small_model_usd_per_1m_predictions']:,.4f}",
        f"  {a['llm_name']} @ {a['llm_input_tokens_per_call']} in / {a['llm_output_tokens_per_call']} out tokens: "
        f"${c['llm_usd_per_1m_predictions']:,.2f}",
        f"  -> LLM costs ~{c['ratio_llm_over_small']:,.0f}x more; small model is ~{c['latency_speedup_vs_llm']:,.0f}x faster "
        f"(vs {a['llm_latency_s']}s LLM latency)",
    ]
    return "\n".join(lines)
