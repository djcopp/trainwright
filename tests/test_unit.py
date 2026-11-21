from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import trainwright as tw
from trainwright.config import (
    ConfigError,
    DataConfig,
    ProjectConfig,
    parse_override,
    resolve_experiment,
)
from trainwright.data.records import Records
from trainwright.data.split import make_split
from trainwright.diagnose import diagnose_history
from trainwright.metrics import compute_metrics
from trainwright.preprocessing import (
    CategoricalEncoder,
    NumericTransformer,
    TabularPreprocessor,
    TargetScaler,
    TextTokenizer,
)


def _cfg(**kw):
    base = dict(
        name="t",
        task="classification",
        modality="tabular",
        data=DataConfig(path="x.csv"),
        metric="accuracy",
    )
    base.update(kw)
    return ProjectConfig(**base)


# --------------------------------------------------------------------------- config


def test_parse_override_types():
    assert parse_override("optim.lr=1e-3") == ("optim.lr", 1e-3)
    assert parse_override("train.max_epochs=5") == ("train.max_epochs", 5)
    assert parse_override("preprocess.hflip=false") == ("preprocess.hflip", False)
    assert parse_override("model.params.backbone=resnet18") == ("model.params.backbone", "resnet18")


def test_unknown_override_key_fails_loudly():
    with pytest.raises(ConfigError, match="optim.lrr"):
        resolve_experiment(_cfg(), overrides=["optim.lrr=0.1"])


def test_switching_model_resets_params():
    exp = resolve_experiment(_cfg(modality="text"), overrides=["model.name=text_cnn"])
    assert exp.model.name == "text_cnn" and exp.model.params == {}


# --------------------------------------------------------------------------- preprocessing


def test_quantile_transform_is_monotone_and_handles_missing_and_ties():
    x = np.array([[1.0], [2.0], [2.0], [2.0], [3.0], [np.nan], [100.0]])
    t = NumericTransformer("quantile").fit(x, ["a"])
    out = t.transform(x)
    assert out.shape == (7, 2)  # value + missing indicator
    finite = out[[0, 1, 4, 6], 0]
    assert np.all(np.diff(finite) > 0)
    assert out[5, 1] == 1.0 and out[0, 1] == 0.0
    assert np.abs(out).max() <= 5.0


def test_state_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    table = {
        "n": rng.normal(size=200),
        "c": np.array(rng.choice(["a", "b", None], 200), dtype=object),
    }
    pre = TabularPreprocessor().fit(table)
    again = TabularPreprocessor.from_state(pre.state_dict())
    a, b = pre.transform(table), again.transform(table)
    np.testing.assert_array_equal(a["x_num"], b["x_num"])
    np.testing.assert_array_equal(a["x_cat"], b["x_cat"])


def test_categorical_unknown_maps_to_zero():
    enc = CategoricalEncoder().fit([["a", "b", "a"]], ["c"])
    assert enc.transform([["a", "z"]]).tolist() == [[enc.vocabs[0]["a"]], [0]]


def test_tokenizer_ngrams_and_padding():
    tok = TextTokenizer(ngrams=2, max_tokens=8, min_freq=1).fit(["the cat sat", "the dog sat"])
    ids = tok.encode(["the cat sat", "unseen words"])
    assert ids.shape == (2, 8)
    assert (ids[0] > 1).sum() == 5  # 3 unigrams + 2 bigrams
    assert set(ids[1][ids[1] > 0].tolist()) == {1}


def test_target_scaler_log1p_roundtrip():
    y = np.array([0.0, 1.0, 10.0, 1000.0])
    s = TargetScaler("log1p").fit(y)
    np.testing.assert_allclose(s.decode(s.encode(y)), y, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- splits


def _records(n=300, groups=None, times=None, task="classification"):
    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b", "c"], n) if task == "classification" else rng.normal(size=n)
    return Records(
        "tabular",
        task,
        [str(i) for i in range(n)],
        np.asarray(y),
        "fp",
        table=pd.DataFrame({"x": rng.normal(size=n)}),
        groups=groups,
        times=times,
    )


def test_stratified_split_is_disjoint_and_deterministic():
    r = _records()
    s1, s2 = make_split(r, _cfg()), make_split(r, _cfg())
    assert s1["split_hash"] == s2["split_hash"]
    parts = [set(s1[k]) for k in ("train", "val", "test")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert sum(map(len, parts)) == 300


def test_group_split_keeps_groups_together():
    groups = np.array([f"g{i // 10}" for i in range(300)])
    r = _records(groups=groups)
    s = make_split(r, _cfg(data=DataConfig(path="x", group_column="g")))
    gid = {i: groups[int(i)] for i in r.ids}
    sets = [{gid[i] for i in s[k]} for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_time_split_is_chronological():
    times = np.arange(300).astype("datetime64[D]")
    r = _records(times=times, task="regression")
    s = make_split(
        r, _cfg(task="regression", metric="rmse", data=DataConfig(path="x", time_column="t"))
    )
    assert (
        max(int(i) for i in s["train"])
        < min(int(i) for i in s["val"])
        < min(int(i) for i in s["test"])
    )


# --------------------------------------------------------------------------- metrics


def test_metrics_basic():
    y = np.array([0, 1, 1, 0])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    m = compute_metrics("classification", y, probs, 2, outputs_are_probs=True)
    assert m["accuracy"] == 0.75 and m["roc_auc"] == 0.75
    r = compute_metrics("regression", np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    assert r["mae"] == 1.0 and abs(r["rmse"] - np.sqrt(2)) < 1e-9


# --------------------------------------------------------------------------- diagnosis


def _hist(train, val, metric_vals, tm=None):
    return [
        {
            "epoch": i + 1,
            "train_loss": t,
            "val_loss": v,
            "val_accuracy": m,
            "train_accuracy": (tm[i] if tm else m),
        }
        for i, (t, v, m) in enumerate(zip(train, val, metric_vals))
    ]


def test_diagnose_detects_overfitting():
    tl = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02]
    vl = [1.0, 0.8, 0.7, 0.72, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    vm = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62, 0.61, 0.6, 0.6, 0.59]
    d = diagnose_history(
        _hist(tl, vl, vm),
        {"best_epoch": 3, "stop_reason": "early stopping", "val": {}},
        {"optim": {"lr": 1e-3, "weight_decay": 1e-5}, "model": {"params": {"dropout": 0.1}}},
        "accuracy",
        "max",
        "tabular",
        "classification",
        3,
    )
    codes = [f["code"] for f in d["findings"]]
    assert "overfitting" in codes
    assert any("weight_decay" in n["command"] for n in d["next_experiments"])


def test_diagnose_detects_not_converged_and_baseline_win():
    tl = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35]
    vm = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    d = diagnose_history(
        _hist(tl, tl, vm),
        {"best_epoch": 8, "stop_reason": "completed max_epochs=8", "val": {"accuracy": 0.65}},
        {"train": {"max_epochs": 8}, "model": {"params": {}}},
        "accuracy",
        "max",
        "tabular",
        "classification",
        3,
        baseline={"id": "002-baseline-hist-gbm", "model": "baseline:hist_gbm", "val_accuracy": 0.8},
    )
    codes = {f["code"]: f["severity"] for f in d["findings"]}
    assert codes.get("not_converged") == "warning"
    assert codes.get("baseline_wins") == "warning"


def test_diagnose_detects_divergence():
    d = diagnose_history(
        _hist([1.0, float("nan")], [1.0, float("nan")], [0.5, 0.5]),
        {"best_epoch": 1, "stop_reason": "diverged: non-finite training loss", "val": {}},
        {"optim": {"lr": 0.1}},
        "accuracy",
        "max",
        "tabular",
        "classification",
        2,
    )
    assert d["verdict"] == "failed" and d["findings"][0]["code"] == "diverged"


# --------------------------------------------------------------------------- python API


def test_lightning_style_python_api():
    tw.seed_everything(0)
    x = torch.randn(512, 2)
    y = (x[:, 0] * x[:, 1] > 0).long()

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Sequential(
                torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
            )

        def forward(self, x):
            return self.f(x)

    task = tw.ClassificationTask(Net(), n_classes=2)
    train = tw.TensorLoader({"x": x[:400], "y": y[:400]}, batch_size=32, shuffle=True)
    val = tw.TensorLoader({"x": x[400:], "y": y[400:]}, batch_size=128)
    trainer = tw.Trainer(
        max_epochs=30, device="cpu", monitor="val_accuracy", mode="max", progress=None
    )
    res = trainer.fit(task, train, val)
    assert res.best_score > 0.85


def test_register_custom_model():
    @tw.register_model("tiny_test_model", modality="tabular", description="test")
    class Tiny(torch.nn.Module):
        def __init__(self, spec, hidden=4):
            super().__init__()
            self.l = torch.nn.Linear(spec["n_num"], spec["n_outputs"])

        def forward(self, x_num, x_cat):
            return self.l(x_num)

    m = tw.build_model(
        "tiny_test_model", {"n_num": 3, "n_outputs": 2, "modality": "tabular"}, {"hidden": 8}
    )
    assert m(torch.randn(2, 3), torch.zeros(2, 0, dtype=torch.long)).shape == (2, 2)
    with pytest.raises(ValueError, match="Unknown params"):
        tw.build_model("tiny_test_model", {"n_num": 3, "n_outputs": 2}, {"nope": 1})


def test_duplicate_aware_split_never_straddles():
    from trainwright.data.dedup import duplicate_groups

    rng = np.random.default_rng(0)
    base = pd.DataFrame({"x": rng.normal(size=100), "z": rng.choice(["a", "b", None], 100)})
    table = pd.concat([base] * 4, ignore_index=True)  # every row appears 4 times
    y = np.array(["p" if v > 0 else "n" for v in table["x"]])
    r = Records("tabular", "classification", [str(i) for i in range(400)], y, "fp", table=table)
    groups, stats = duplicate_groups(r)
    assert stats["n_groups"] == 100 and stats["share_with_duplicates"] == 1.0
    s = make_split(r, _cfg(), groups, stats)
    where = {i: k for k in ("train", "val", "test") for i in s[k]}
    for g in set(groups):
        members = [str(i) for i in np.flatnonzero(groups == g)]
        assert len({where[m] for m in members}) == 1
    assert s["counts"]["val"] > 0 and s["counts"]["test"] > 0


def test_bootstrap_se_matches_binomial_theory():
    from trainwright.metrics import bootstrap_se

    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    correct = rng.random(n) < 0.8
    pred = np.where(correct, y, 1 - y)
    probs = np.eye(2)[pred]
    se = bootstrap_se("classification", "accuracy", y, probs, 2, n_boot=400)
    assert abs(se - np.sqrt(0.8 * 0.2 / n)) < 0.003


def test_diagnose_reports_tie_with_baseline():
    h = _hist([1.0, 0.8, 0.7], [1.0, 0.8, 0.7], [0.80, 0.81, 0.82])
    d = diagnose_history(
        h,
        {
            "best_epoch": 3,
            "stop_reason": "early stopping",
            "val": {"accuracy": 0.82},
            "val_se": {"accuracy": 0.01},
        },
        {"model": {"params": {}}},
        "accuracy",
        "max",
        "tabular",
        "classification",
        2,
        baseline={
            "id": "002-b",
            "model": "baseline:hist_gbm",
            "val_accuracy": 0.83,
            "val_se": 0.01,
        },
    )
    codes = [f["code"] for f in d["findings"]]
    assert "tied_with_baseline" in codes and "baseline_wins" not in codes


def test_group_scan_flags_entity_keys_not_measurements():
    from trainwright.data.profile import _group_scan

    rng = np.random.default_rng(0)
    n, k = 3000, 100
    user = rng.integers(0, k, n)
    user_effect = rng.normal(size=k)[user]  # rows of one user share context
    df = pd.DataFrame(
        {
            "userId": user,
            "f1": user_effect + rng.normal(scale=0.5, size=n),
            "f2": user_effect * 2 + rng.normal(scale=0.5, size=n),
            "duration": rng.integers(0, 300, n),  # integer measurement, not a key
            "colourNo": rng.integers(0, 8, n),  # key-like name but carries no shared context
        }
    )
    y = (df["f1"] > 0).astype(int).to_numpy()
    flagged = [g["column"] for g in _group_scan(df, y, "classification")]
    assert flagged == ["userId"]


def test_every_registered_model_builds_and_runs():
    """Catches registration/signature mistakes for every built-in architecture (no downloads)."""
    specs = {
        "tabular": (
            {"n_num": 3, "cat_cardinalities": [4], "n_outputs": 2, "modality": "tabular"},
            {"x_num": torch.randn(2, 3), "x_cat": torch.zeros(2, 1, dtype=torch.long)},
        ),
        "image": ({"n_outputs": 2, "modality": "image"}, {"image": torch.rand(2, 3, 64, 64)}),
        "text": (
            {"vocab_size": 50, "n_outputs": 2, "modality": "text"},
            {"tokens": torch.randint(1, 50, (2, 8))},
        ),
    }
    offline = {"image_pretrained": {"pretrained": False}}
    for info in tw.list_models():
        if info["name"] in ("text_pretrained", "tiny_test_model", "image_probe"):
            continue  # needs a hub download / registered by another test
        spec, batch = specs[info["modality"]]
        m = tw.build_model(info["name"], spec, offline.get(info["name"], {})).eval()
        with torch.no_grad():
            assert m(**batch).shape == (2, 2), info["name"]


def test_mcp_server_exposes_lifecycle_tools():
    import asyncio

    pytest.importorskip("mcp")
    from trainwright.mcp_server import build_server

    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert {
        "status",
        "profile",
        "split",
        "baseline",
        "train",
        "diagnose",
        "evaluate",
        "export",
    } <= names


def test_greedy_ensemble_weights_drop_useless_members():
    from trainwright.ensemble import _greedy_weights
    from trainwright.metrics import compute_metrics

    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)

    def noisy(strength):
        p = np.clip(0.5 + (y - 0.5) * strength + rng.normal(scale=0.25, size=n), 0.01, 0.99)
        return np.stack([1 - p, p], 1)

    good_a, good_b, junk = noisy(0.6), noisy(0.6), np.full((n, 2), 0.5)
    obj = lambda o: compute_metrics("classification", y, o, 2, outputs_are_probs=True)["log_loss"]  # noqa: E731
    w = _greedy_weights([good_a, good_b, junk], obj)
    assert (
        w[0] > 0 and w[1] > 0 and w[2] == 0
    )  # two independent good models combine; junk is ignored
