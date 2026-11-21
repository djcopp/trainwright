"""Distillation helpers: label data with an LLM, then find and fix the labels that are wrong.

The usual route from "an LLM does this today" to "a small model does this":

1. ``trainwright label``   - label unlabelled items with Claude (or import the labels your
   production LLM already produced: any CSV with a label column works as-is).
2. train / diagnose / iterate as normal (metrics now measure agreement with the LLM).
3. ``trainwright review``  - rank validation items where the trained model confidently
   disagrees with the label: these are mostly *label errors*. Optionally ask a stronger model
   to adjudicate. Write fixes to ``corrections.csv``.
4. set ``data.corrections: corrections.csv`` in trainwright.yaml; corrections override labels
   by id at load time, are hashed into every run, and never change the frozen split.

Spending money: every LLM call is priced up front and capped by ``--max-cost-usd``.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

# List prices (USD per million tokens). Edit if your contract differs.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-2": (8.0, 40.0),
}
DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2"}
IMAGE_MAX_SIDE = 768  # downscale before sending: ~(768*576)/750 ≈ 600 tokens per photo


class BudgetExceeded(RuntimeError):
    pass


def _image_block(path: Path) -> tuple[dict[str, Any], int]:
    from PIL import Image, ImageOps

    src = Image.open(path)
    if getattr(src, "format", None) == "JPEG":
        src.draft("RGB", (IMAGE_MAX_SIDE * 2, IMAGE_MAX_SIDE * 2))
    im = ImageOps.exif_transpose(src).convert("RGB")
    im.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    tokens = int(im.size[0] * im.size[1] / 750)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }, tokens


def _item_content(item: Any, modality: str) -> tuple[list[dict[str, Any]], int]:
    if modality == "image":
        block, toks = _image_block(Path(item))
        return [block, {"type": "text", "text": "Classify this image."}], toks + 10
    text = json.dumps(item, default=str) if isinstance(item, dict) else str(item)
    return [{"type": "text", "text": f"Item to classify:\n<item>\n{text}\n</item>"}], len(
        text
    ) // 4 + 15


def _system_prompt(labels: list[str], instructions: str) -> str:
    lines = "\n".join(f"- {lab}" for lab in labels)
    return (
        "You are a careful data labeller producing training labels for a classifier.\n"
        f"{instructions.strip()}\n\nAllowed labels (choose exactly one):\n{lines}\n\n"
        "Return the single best label, your confidence that it is correct (0 to 1), and a "
        "one-sentence rationale. If the item is ambiguous, still choose the best label and lower "
        "the confidence."
    )


def _schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": labels},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["label", "confidence", "rationale"],
        "additionalProperties": False,
    }


def estimate_cost(
    items: list[Any],
    modality: str,
    labels: list[str],
    instructions: str,
    model: str,
    output_tokens: int = 350,
) -> dict[str, Any]:
    p_in, p_out = PRICES.get(model, PRICES[DEFAULT_MODEL])
    sys_tokens = len(_system_prompt(labels, instructions)) // 4
    sample = items[: min(len(items), 20)]
    per_item = [(_item_content(x, modality)[1] + sys_tokens) for x in sample] or [sys_tokens]
    avg_in = float(np.mean(per_item))
    total = len(items) * (avg_in * p_in + output_tokens * p_out) / 1e6
    return {
        "model": model,
        "n_items": len(items),
        "avg_input_tokens": round(avg_in),
        "assumed_output_tokens": output_tokens,
        "usd_per_mtok": [p_in, p_out],
        "estimated_usd": round(total, 4),
    }


def label_items(
    items: list[Any],
    labels: list[str],
    instructions: str,
    modality: str,
    model: str = DEFAULT_MODEL,
    effort: str = "low",
    max_cost_usd: float = 5.0,
    concurrency: int = 8,
    client: Any = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label items with Claude. Returns (rows, spend). Stops before exceeding ``max_cost_usd``."""
    if client is None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "LLM labelling needs the anthropic SDK: `uv add 'trainwright[label]'`"
            ) from e
        client = anthropic.Anthropic()
    p_in, p_out = PRICES.get(model, PRICES[DEFAULT_MODEL])
    system = [
        {
            "type": "text",
            "text": _system_prompt(labels, instructions),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    lock = threading.Lock()
    spend: dict[str, Any] = {
        "usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "calls": 0,
        "refusals": 0,
        "errors": 0,
    }
    stop = threading.Event()

    def one(i: int) -> dict[str, Any]:
        if stop.is_set():
            return {"index": i, "label": None, "status": "skipped (budget)"}
        content, _ = _item_content(items[i], modality)
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": _schema(labels)},
            },
        )
        try:
            if model in FALLBACK_MODELS:
                resp = client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            else:
                resp = client.messages.create(**kwargs)
        except Exception as e:
            with lock:
                spend["errors"] += 1
            return {
                "index": i,
                "label": None,
                "status": f"error: {type(e).__name__}: {str(e)[:200]}",
            }
        u = resp.usage
        cost = (
            (getattr(u, "input_tokens", 0) or 0)
            + (getattr(u, "cache_creation_input_tokens", 0) or 0) * 1.25
            + (getattr(u, "cache_read_input_tokens", 0) or 0) * 0.1
        ) * p_in / 1e6 + (u.output_tokens or 0) * p_out / 1e6
        with lock:
            spend["usd"] += cost
            spend["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            spend["output_tokens"] += u.output_tokens or 0
            spend["calls"] += 1
            if spend["usd"] >= max_cost_usd:
                stop.set()
        if resp.stop_reason == "refusal":
            with lock:
                spend["refusals"] += 1
            return {"index": i, "label": None, "status": "refused"}
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            return {"index": i, "label": None, "status": f"unparseable ({resp.stop_reason})"}
        return {
            "index": i,
            "label": out["label"],
            "confidence": float(out.get("confidence", 0)),
            "rationale": out.get("rationale", ""),
            "status": "ok",
        }

    t0 = time.time()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        for k, r in enumerate(ex.map(one, range(len(items)))):
            rows.append(r)
            if progress and (k + 1) % 50 == 0:
                progress(f"labelled {k + 1}/{len(items)}  spend ${spend['usd']:.3f}")
    spend["usd"] = round(spend["usd"], 4)
    spend["seconds"] = round(time.time() - t0, 1)
    spend["model"] = model
    spend["budget_hit"] = stop.is_set()
    return rows, spend


# --------------------------------------------------------------------------- review


def review_queue(project: Any, run: Any, top: int = 50) -> dict[str, Any]:
    """Validation items where the model confidently disagrees with the given label.

    Uses only predictions on data the model never trained on (val). High model confidence on a
    different class + low probability on the given label = probable label error.
    """
    from .evaluate import evaluate

    ev = run.eval_result("val") or evaluate(project, run, "val")
    preds_path = run.eval_dir / "val_predictions.csv"
    import pandas as pd

    df = pd.read_csv(preds_path, dtype={"id": str})
    df = df[df["true"].astype(str) != df["predicted"].astype(str)].copy()
    n_errors = len(df)
    df = df.sort_values("confidence", ascending=False).head(top)
    from .pipeline import get_records

    records = get_records(project)
    pos = {i: k for k, i in enumerate(records.ids)}
    items = []
    for _, r in df.iterrows():
        k = pos.get(str(r["id"]))
        entry: dict[str, Any] = {
            "id": str(r["id"]),
            "given_label": str(r["true"]),
            "model_prediction": str(r["predicted"]),
            "model_confidence": float(r["confidence"]),
        }
        if k is not None:
            if records.texts is not None:
                entry["input"] = records.texts[k]
            elif records.paths is not None:
                entry["input"] = str(records.paths[k])
            elif records.table is not None:
                entry["input"] = json.dumps(records.table.iloc[k].to_dict(), default=str)
        items.append(entry)
    return {
        "run": run.id,
        "n_val_disagreements": n_errors,
        "val_metrics": ev.get("metrics"),
        "queue": items,
    }


def write_review(project: Any, queue: dict[str, Any]) -> dict[str, str]:
    import csv

    out_dir = project.root / "review"
    out_dir.mkdir(exist_ok=True)
    qpath = out_dir / "queue.csv"
    with qpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "given_label", "model_prediction", "model_confidence", "llm_verdict", "input"]
        )
        for q in queue["queue"]:
            w.writerow(
                [
                    q["id"],
                    q["given_label"],
                    q["model_prediction"],
                    f"{q['model_confidence']:.3f}",
                    q.get("llm_verdict", ""),
                    str(q.get("input", ""))[:500],
                ]
            )
    paths = {"queue": str(qpath)}
    if project.config.modality == "image" and queue["queue"]:
        paths["sheet"] = str(_review_sheet(queue["queue"], out_dir / "queue.png"))
    corr = project.root / "corrections.csv"
    if not corr.exists():
        corr.write_text("id,label,source,note\n")
    paths["corrections"] = str(corr)
    return paths


def _review_sheet(items: list[dict[str, Any]], path: Path, thumb: int = 160, cols: int = 5) -> Path:
    from PIL import Image, ImageDraw

    items = items[:30]
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 30)), "white")
    d = ImageDraw.Draw(sheet)
    for k, it in enumerate(items):
        r, c = divmod(k, cols)
        x, y = c * thumb, r * (thumb + 30)
        try:
            with Image.open(it["input"]) as src:
                im = src.convert("RGB")
                im.thumbnail((thumb, thumb))
                sheet.paste(im, (x, y))
        except Exception:
            pass
        d.text((x + 2, y + thumb + 1), f"label: {it['given_label'][:20]}", fill="darkblue")
        d.text(
            (x + 2, y + thumb + 15),
            f"model: {it['model_prediction'][:14]} {it['model_confidence']:.2f}",
            fill="red",
        )
    sheet.save(path)
    return path


def adjudicate(
    queue: dict[str, Any],
    labels: list[str],
    instructions: str,
    modality: str,
    model: str,
    max_cost_usd: float,
    client: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask a (stronger) LLM to label each disputed item blind, i.e. without seeing either answer."""
    items = [q.get("input") for q in queue["queue"]]
    rows, spend = label_items(
        items, labels, instructions, modality, model, "medium", max_cost_usd, client=client
    )
    for q, r in zip(queue["queue"], rows):
        q["llm_verdict"] = r.get("label") or ""
        q["llm_confidence"] = r.get("confidence")
    return queue, spend


def load_corrections(path: Path) -> dict[str, str]:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"data.corrections file {path} not found")
    df = pd.read_csv(path, dtype=str).dropna(subset=["id", "label"])
    return dict(zip(df["id"].astype(str), df["label"].astype(str)))
