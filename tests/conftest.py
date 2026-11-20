from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from trainwright.cli import STATE, app


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class TW:
    """Run trainwright CLI commands in a project directory and parse --json output."""

    def __init__(self, root: Path):
        self.root = root
        self.runner = CliRunner()

    def __call__(self, *args: str, ok: bool = True) -> dict:
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            STATE["json"] = False
            res = self.runner.invoke(app, ["--json", *args], catch_exceptions=not ok)
        finally:
            os.chdir(cwd)
        if ok:
            assert res.exit_code == 0, f"{args} failed:\n{res.output}\n{res.exception}"
        out = res.stdout.strip()
        start = out.rfind("\n{\n") + 1 if "\n{\n" in out else out.find("{")
        return (
            json.loads(out[start:])
            if start >= 0 and out
            else {"raw": res.output, "exception": res.exception}
        )


@pytest.fixture
def tabular_project(tmp_path: Path) -> TW:
    rng = np.random.default_rng(0)
    n = 1200
    X = rng.normal(size=(n, 4))
    cat = rng.choice(["red", "green", "blue"], n)
    y = (
        (X[:, 0] + (cat == "red") * 1.5 - X[:, 1] ** 2 + rng.normal(scale=0.3, size=n)) > 0
    ).astype(int)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df.loc[rng.choice(n, 50, replace=False), "c"] = np.nan
    df["colour"] = cat
    df["row_id"] = np.arange(n)
    df["label"] = np.where(y == 1, "yes", "no")
    df.to_csv(tmp_path / "data.csv", index=False)
    return TW(tmp_path)


@pytest.fixture
def regression_project(tmp_path: Path) -> TW:
    rng = np.random.default_rng(1)
    n = 1000
    X = rng.normal(size=(n, 3))
    y = np.exp(1 + X[:, 0] * 0.8 + 0.3 * X[:, 1]) + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["price"] = y
    df.to_csv(tmp_path / "houses.csv", index=False)
    return TW(tmp_path)


@pytest.fixture
def text_project(tmp_path: Path) -> TW:
    rng = np.random.default_rng(2)
    vocab = {
        "billing": ["invoice", "charge", "refund", "payment", "card", "billed"],
        "tech": ["crash", "error", "bug", "login", "slow", "broken"],
        "sales": ["pricing", "quote", "upgrade", "plan", "demo", "discount"],
    }
    filler = ["please", "help", "my", "the", "account", "today", "hi", "thanks", "i", "need"]
    rows = []
    for _ in range(600):
        lab = rng.choice(list(vocab))
        words = list(rng.choice(vocab[lab], 3)) + list(rng.choice(filler, 6))
        rng.shuffle(words)
        rows.append({"message": " ".join(words), "intent": lab})
    pd.DataFrame(rows).to_csv(tmp_path / "tickets.csv", index=False)
    return TW(tmp_path)


@pytest.fixture
def image_project(tmp_path: Path) -> TW:
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(3)
    root = tmp_path / "shapes"
    for cls in ["circle", "square"]:
        (root / cls).mkdir(parents=True)
        for i in range(40):
            img = Image.new("RGB", (48, 48), tuple(int(v) for v in rng.integers(0, 80, 3)))
            d = ImageDraw.Draw(img)
            x, y, r = int(rng.integers(12, 36)), int(rng.integers(12, 36)), int(rng.integers(6, 11))
            colour = tuple(int(v) for v in rng.integers(150, 255, 3))
            if cls == "circle":
                d.ellipse([x - r, y - r, x + r, y + r], fill=colour)
            else:
                d.rectangle([x - r, y - r, x + r, y + r], fill=colour)
            img.save(root / cls / f"{i}.png")
    return TW(tmp_path)
