"""Distillation suite tests with a fake Claude client (no network, no spend)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from trainwright.label import BudgetExceeded, estimate_cost, label_items  # noqa: F401


class FakeClient:
    """Mimics client.messages.create / client.beta.messages.create."""

    def __init__(self, rule, refuse_on: str | None = None):
        self.rule = rule
        self.refuse_on = refuse_on
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.kwargs: list[dict] = []

    def _create(self, **kw):
        self.calls += 1
        self.kwargs.append(kw)
        text = kw["messages"][0]["content"][-1]["text"]
        usage = SimpleNamespace(
            input_tokens=200,
            output_tokens=60,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        if self.refuse_on and self.refuse_on in text:
            return SimpleNamespace(stop_reason="refusal", content=[], usage=usage)
        body = json.dumps({"label": self.rule(text), "confidence": 0.9, "rationale": "fake"})
        return SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text=body)], usage=usage
        )


def _rule(text: str) -> str:
    return "billing" if "invoice" in text or "refund" in text else "tech"


def test_label_items_structured_output_and_refusal():
    items = ["my invoice is wrong", "app crashes", "refund please", "BLOCKED content"]
    client = FakeClient(_rule, refuse_on="BLOCKED")
    rows, spend = label_items(
        items,
        ["billing", "tech"],
        "Route tickets.",
        "text",
        model="claude-opus-5-5",
        client=client,
        max_cost_usd=1.0,
        concurrency=2,
    )
    assert [r["label"] for r in rows] == ["billing", "tech", "billing", None]
    assert rows[3]["status"] == "refused" and spend["refusals"] == 1
    kw = client.kwargs[0]
    schema = kw["output_config"]["format"]["schema"]
    assert schema["properties"]["label"]["enum"] == ["billing", "tech"]
    assert kw["fallbacks"] == "default"  # server-side refusal fallback for Opus 5.5
    assert spend["usd"] > 0


def test_label_budget_cap_stops_spending():
    items = [f"ticket {i} invoice" for i in range(200)]
    client = FakeClient(_rule)
    rows, spend = label_items(
        items,
        ["billing", "tech"],
        "",
        "text",
        model="claude-opus-5-5",
        client=client,
        max_cost_usd=0.01,
        concurrency=1,
    )
    assert spend["budget_hit"] and client.calls < 200
    assert any(r["status"].startswith("skipped") for r in rows)


def test_estimate_cost_scales_with_items():
    small = estimate_cost(["hello"] * 10, "text", ["a", "b"], "", "claude-haiku-5-5")
    big = estimate_cost(["hello"] * 1000, "text", ["a", "b"], "", "claude-haiku-5-5")
    assert big["estimated_usd"] == pytest.approx(small["estimated_usd"] * 100, rel=0.05)


def test_corrections_override_labels_without_breaking_split(text_project):
    tw = text_project
    tw("init", "--data", "tickets.csv", "--target", "intent")
    tw("split")
    run = tw(
        "train",
        "-o",
        "train.device=cpu",
        "-o",
        "train.max_epochs=3",
        "-o",
        "preprocess.min_token_freq=1",
        "-H",
        "base",
    )["result"]["run"]
    rq = tw("review", run, "--top", "5")["result"]
    assert "queue" in rq and (tw.root / "review" / "queue.csv").exists()
    split_before = json.loads((tw.root / ".trainwright" / "split.json").read_text())["split_hash"]
    # flip one label via corrections and make sure it is applied and recorded
    df = pd.read_csv(tw.root / "tickets.csv")
    (tw.root / "corrections.csv").write_text(
        f"id,label,source,note\n0,{'sales' if df.intent[0] != 'sales' else 'tech'},human,test\n"
    )
    cfg = (
        (tw.root / "trainwright.yaml")
        .read_text()
        .replace("corrections: null", "corrections: corrections.csv")
    )
    (tw.root / "trainwright.yaml").write_text(cfg)
    run2 = tw(
        "train", "-o", "train.device=cpu", "-o", "train.max_epochs=2", "-H", "with corrections"
    )["result"]["run"]
    meta = json.loads((tw.root / "runs" / run2 / "meta.json").read_text())
    assert meta["corrections"]["applied"] == 1
    assert (
        json.loads((tw.root / ".trainwright" / "split.json").read_text())["split_hash"]
        == split_before
    )
