"""End-to-end lifecycle tests through the CLI, one per modality."""

from __future__ import annotations

import json

import numpy as np

FAST = ["-o", "train.device=cpu", "-o", "train.min_steps=0"]


def test_tabular_classification_lifecycle(tabular_project):
    tw = tabular_project
    r = tw("init", "--data", "data.csv")
    assert r["result"]["config"]["metric"] == "roc_auc"
    prof = tw("profile")["result"]
    assert "row_id" in prof["auto_dropped"]
    split = tw("split")["result"]
    assert split["counts"]["test"] > 0
    base = tw("baseline")["result"]
    assert len(base["runs"]) == 3
    run = tw("train", *FAST, "-o", "train.max_epochs=8", "-H", "default")["result"]
    assert run["summary"]["val"]["roc_auc"] > 0.8
    rid = run["run"]
    child = tw(
        "train",
        "--from-run",
        rid,
        *FAST,
        "-o",
        "train.max_epochs=4",
        "-o",
        "model.params.num_embedding=periodic",
        "-H",
        "periodic",
    )
    assert child["result"]["summary"]["val"]["roc_auc"] > 0.7
    assert len(tw("runs")["result"]) == 5
    assert tw("diagnose", rid)["result"]["verdict"] in ("healthy", "needs_attention")
    ev = tw("evaluate", rid)["result"]
    assert ev["split"] == "val" and ev["per_class"]
    feats = tw("features", rid, "--repeats", "1")["result"]
    assert feats["importances"][0]["feature"] in ("a", "b", "colour")
    test1 = tw("evaluate", rid, "--split", "test", "--reason", "final")
    assert not test1["warnings"] or "not the best" in test1["warnings"][0]
    exp = tw("export", rid)["result"]
    assert exp["parity_ok"] and exp["smoke_test"]["ok"]
    # the bundle works with the standalone runtime exactly like a deployment
    from trainwright.runtime import Predictor

    pred = Predictor(exp["dir"])
    out = pred.predict([{"a": 1.0, "b": 0.0, "c": None, "d": 0.0, "colour": "red"}])
    assert out[0]["label"] in ("yes", "no") and 0 <= out[0]["confidence"] <= 1
    # the generated FastAPI server
    import sys

    from fastapi.testclient import TestClient

    sys.path.insert(0, exp["dir"])
    import importlib.util

    spec = importlib.util.spec_from_file_location("srv_test", f"{exp['dir']}/server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    client = TestClient(mod.app)
    body = json.loads((tw.root / exp["dir"] / "example_request.json").read_text())
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200 and len(resp.json()["predictions"]) == len(body["instances"])
    st = tw("status")["result"]
    assert {s["stage"]: s["done"] for s in st["stages"]}["export"]
    assert (tw.root / tw("report")["result"]["path"]).exists()


def test_test_set_reuse_is_warned(tabular_project):
    tw = tabular_project
    tw("init", "--data", "data.csv")
    tw("split")
    a = tw("train", *FAST, "-o", "train.max_epochs=2", "-H", "a")["result"]["run"]
    b = tw("train", *FAST, "-o", "train.max_epochs=2", "-H", "b")["result"]["run"]
    tw("evaluate", a, "--split", "test")
    second = tw("evaluate", b, "--split", "test")
    assert any("already used" in w for w in second["warnings"])


def test_regression_lifecycle(regression_project):
    tw = regression_project
    r = tw("init", "--data", "houses.csv", "--target", "price")["result"]
    assert r["config"]["task"] == "regression" and r["config"]["metric"] == "rmse"
    prof = tw("profile")["result"]
    assert any("log1p" in x for x in prof["recommendations"])
    tw("split")
    tw("baseline", "--only", "dummy,linear")
    run = tw(
        "train",
        *FAST,
        "-o",
        "train.max_epochs=15",
        "-o",
        "preprocess.target_transform=log1p",
        "-H",
        "log target",
    )["result"]
    dummy = [x for x in tw("runs")["result"] if x["model"] == "baseline:dummy"][0]
    assert run["summary"]["val"]["rmse"] < dummy["val_rmse"]
    exp = tw("export", run["run"])["result"]
    from trainwright.runtime import Predictor

    p = Predictor(exp["dir"]).predict([{"x1": 0.0, "x2": 0.0, "x3": 0.0}])[0]["prediction"]
    assert 1.0 < p < 6.0  # exp(1) ~ 2.7


def test_text_lifecycle(text_project):
    tw = text_project
    r = tw("init", "--data", "tickets.csv", "--target", "intent")["result"]
    assert r["config"]["modality"] == "text" and r["config"]["data"]["text_column"] == "message"
    tw("profile")
    tw("split")
    tw("baseline")
    run = tw(
        "train",
        *FAST,
        "-o",
        "train.max_epochs=10",
        "-o",
        "preprocess.min_token_freq=1",
        "-H",
        "bag",
    )["result"]
    assert run["summary"]["val"]["accuracy"] > 0.9
    exp = tw("export", run["run"])["result"]
    from trainwright.runtime import Predictor

    assert Predictor(exp["dir"]).predict(["my invoice has a wrong charge"])[0]["label"] == "billing"


def test_image_lifecycle(image_project):
    tw = image_project
    r = tw("init", "--data", "shapes")["result"]
    assert r["config"]["modality"] == "image"
    prof = tw("profile")["result"]
    assert (tw.root / prof["image"]["contact_sheet"]).exists()
    tw("split")
    tw("baseline")
    run = tw(
        "train",
        *FAST,
        "-o",
        "model.name=image_cnn",
        "-o",
        "preprocess.image_size=32",
        "-o",
        "train.max_epochs=12",
        "-o",
        "train.batch_size=16",
        "-o",
        "optim.lr=3e-3",
        "-o",
        "optim.weight_decay=1e-4",
        "-H",
        "cnn",
    )["result"]
    assert run["summary"]["val"]["accuracy"] >= 0.75
    ev = tw("evaluate", run["run"])["result"]
    assert "confusion_png" in ev
    exp = tw("export", run["run"])["result"]
    assert exp["parity_ok"]
    from trainwright.runtime import Predictor

    img = next((tw.root / "shapes" / "circle").glob("*.png"))
    out = Predictor(exp["dir"]).predict([str(img)])
    assert out[0]["label"] in ("circle", "square")


def test_tune_and_sweep(tabular_project):
    tw = tabular_project
    tw("init", "--data", "data.csv")
    tw("split")
    tw("train", *FAST, "-o", "train.max_epochs=4", "-H", "base")
    res = tw("tune", "--budget", "40s", "--trials", "3")["result"]
    assert res["best_run"] is not None and len(res["trials"]) >= 3
    sw = tw(
        "sweep", "-o", "train.device=cpu", "-o", "train.max_epochs=2", "-o", "optim.lr=1e-3,3e-3"
    )["result"]
    assert len(sw["runs"]) == 2


def test_errors_are_actionable(tabular_project):
    tw = tabular_project
    out = tw("status", ok=False)
    assert out.get("ok") is False or "exception" in out
    tw("init", "--data", "data.csv")
    bad = tw("train", "-o", "optim.lrr=1", ok=False)
    assert bad.get("ok") is False or bad.get("exception") is not None


def test_schema_and_models(tabular_project):
    tw = tabular_project
    tw("init", "--data", "data.csv")
    assert "experiment" in tw("schema")["result"]
    names = [m["name"] for m in tw("models")["result"]]
    assert names == ["tabular_mlp"]
    assert np.isfinite(1.0)


FEATURES_PY = """
import pandas as pd

def fit(train):
    return {"colour_rate": {str(k): float(v) for k, v in train.groupby("colour")["a"].mean().items()}}

def transform(df, state):
    df = df.copy()
    df["colour_a_mean"] = df["colour"].astype(str).map(state["colour_rate"]).astype(float)
    df["a_times_b"] = df["a"] * df["b"]
    return df
"""


def test_feature_hook_is_fitted_tracked_and_shipped(tabular_project):
    tw = tabular_project
    (tw.root / "features.py").write_text(FEATURES_PY)
    tw("init", "--data", "data.csv")
    tw("split")
    tw("baseline", "--only", "dummy,hist_gbm")
    run = tw("train", *FAST, "-o", "train.max_epochs=4", "-H", "with features.py")["result"]
    rid = run["run"]
    meta = json.loads((tw.root / "runs" / rid / "meta.json").read_text())
    assert meta["features_sha"]
    exp = tw("export", rid)["result"]
    bundle = json.loads((tw.root / exp["dir"] / "bundle.json").read_text())
    assert "a_times_b" in bundle["preprocessor"]["features"]["numeric_columns"]
    assert (tw.root / exp["dir"] / "tw_runtime" / "featurehook.py").exists()
    from trainwright.runtime import Predictor

    out = Predictor(exp["dir"]).predict([{"a": 1.0, "b": 2.0, "c": 0.0, "d": 0.0, "colour": "red"}])
    assert out[0]["label"] in ("yes", "no")
    # the sklearn baseline bundle applies the same hook
    gbm = [r for r in tw("runs")["result"] if r["model"] == "baseline:hist_gbm"][0]["id"]
    bexp = tw("export", gbm)["result"]
    assert Predictor(bexp["dir"]).predict(
        [{"a": 1.0, "b": 2.0, "c": 0.0, "d": 0.0, "colour": "red"}]
    )
    assert tw("features", rid, "--repeats", "1")["result"]["importances"]


def test_finalize_trains_on_train_plus_val(tabular_project):
    tw = tabular_project
    tw("init", "--data", "data.csv")
    tw("split")
    rid = tw("train", *FAST, "-o", "train.max_epochs=3", "-H", "base")["result"]["run"]
    fin = tw("finalize", rid)
    assert any("no test evaluation" in w for w in fin["warnings"])
    fid = fin["result"]["run"]
    meta = json.loads((tw.root / "runs" / fid / "meta.json").read_text())
    assert (
        meta["kind"] == "final" and meta["trained_on"] == ["train", "val"] and meta["parent"] == rid
    )
    assert fid not in [r["id"] for r in tw("runs")["result"]]  # val score is not held out
    bad = tw("evaluate", fid, "--split", "val", ok=False)
    assert bad.get("ok") is False or bad.get("exception") is not None
    assert tw("export", fid)["result"]["parity_ok"]


def test_finalize_a_baseline_and_ship_it(regression_project):
    tw = regression_project
    tw("init", "--data", "houses.csv", "--target", "price")
    tw("split")
    tw("baseline", "--only", "hist_gbm")
    gbm = tw("runs")["result"][0]["id"]
    tw("evaluate", gbm, "--split", "test", "--reason", "baseline is the final model")
    fid = tw("finalize", gbm)["result"]["run"]
    meta = json.loads((tw.root / "runs" / fid / "meta.json").read_text())
    assert meta["kind"] == "final" and meta["model"] == "baseline:hist_gbm"
    # test was already evaluated, so the production model also learns from it ...
    assert meta["trained_on"] == ["train", "val", "test"]
    # ... and can therefore never be scored on it
    refused = tw("evaluate", fid, "--split", "test", ok=False)
    assert refused.get("ok") is False or refused.get("exception") is not None
    # opting out keeps test held out
    held = tw("finalize", gbm, "--no-include-test")["result"]["run"]
    assert tw("evaluate", held, "--split", "test")["result"]["metrics"]["rmse"] > 0
    exp = tw("export", fid)["result"]
    from trainwright.runtime import Predictor

    assert Predictor(exp["dir"]).predict([{"x1": 0.0, "x2": 0.0, "x3": 0.0}])[0]["prediction"] > 0


def test_ensemble_build_evaluate_finalize_export(tabular_project):
    tw = tabular_project
    tw("init", "--data", "data.csv")
    tw("split")
    tw("baseline", "--only", "dummy,hist_gbm")
    tw("train", *FAST, "-o", "train.max_epochs=4", "-H", "mlp")
    ens = tw("ensemble", "--seeds", "2")["result"]
    eid = ens["run"]
    members = ens["summary"]["members"]
    assert len(members) >= 2 and not any("dummy" in m for m in members)
    assert ens["summary"]["seed_spread"]["n"] == 2
    assert eid in [r["id"] for r in tw("runs")["result"]]
    test = tw("evaluate", eid, "--split", "test", "--reason", "ensemble")["result"]
    assert test["metrics"]["roc_auc"] > 0.8
    fid = tw("finalize", eid)["result"]["run"]
    exp = tw("export", fid)["result"]
    from trainwright.runtime import Predictor

    out = Predictor(exp["dir"]).predict(
        [{"a": 1.0, "b": 0.0, "c": None, "d": 0.0, "colour": "red"}]
    )
    assert out[0]["label"] in ("yes", "no") and 0 <= out[0]["confidence"] <= 1


def test_predict_prefers_the_finalized_model(tabular_project):
    tw = tabular_project
    tw("init", "--data", "data.csv")
    tw("split")
    rid = tw("train", *FAST, "-o", "train.max_epochs=3", "-H", "base")["result"]["run"]
    (tw.root / "new.csv").write_text("a,b,c,d,colour,row_id\n1,0,0,0,red,5\n")
    before = tw("predict", rid, "-i", "new.csv", "--out", "p.csv")
    assert any("finalize" in w for w in before["warnings"]) and before["result"]["model"] == rid
    fid = tw("finalize", rid)["result"]["run"]
    after = tw("predict", rid, "-i", "new.csv", "--out", "p.csv")["result"]
    assert after["model"] == fid


def test_auto_runs_the_whole_search_and_finalizes(regression_project):
    tw = regression_project
    tw("init", "--data", "houses.csv", "--target", "price")
    res = tw("auto", "--budget", "3m", *FAST, "-o", "train.max_epochs=6")["result"]
    assert res["best"] is not None and res.get("final")
    models = {r["model"] for r in tw("runs")["result"]}
    assert {"baseline:hist_gbm", "tabular_mlp"} <= models
    # rerunning skips everything that is already up to date
    again = tw("auto", "--budget", "60s", *FAST, "-o", "train.max_epochs=6", "--no-seeds")
    assert not any("baselines:" in line for line in again["result"]["log"])


def test_profile_flags_a_target_that_is_a_sum_of_columns():
    import pandas as pd

    from trainwright.data.profile import _leakage_scan

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"temp": rng.normal(size=2000), "hour": rng.integers(0, 24, 2000)})
    y = np.exp(df["temp"]) * 50 + df["hour"] * 3
    share = rng.uniform(0, 1, 2000)
    df["casual"], df["registered"] = y * share, y * (1 - share)
    leaks = _leakage_scan(df, y.to_numpy(), "regression")
    assert leaks and set(leaks[0]["column"].split(", ")) == {"casual", "registered"}


def test_auto_background_and_wait(regression_project):
    tw = regression_project
    tw("init", "--data", "houses.csv", "--target", "price")
    started = tw("auto", "--budget", "60s", "--background", *FAST, "-o", "train.max_epochs=4")
    assert started["result"]["pid"] > 0
    assert tw("status")["next_steps"][0]["command"].startswith("trainwright wait")
    done = tw("wait", "--timeout", "15m")["result"]
    assert done["status"] == "done" and done["best"]


def test_bag_scores_out_of_fold_and_finalizes_for_free(tabular_project):
    tw = tabular_project
    tw("init", "--data", "data.csv")
    tw("split")
    tw("baseline", "--only", "dummy,hist_gbm")
    rid = tw("train", *FAST, "-o", "train.max_epochs=4", "-H", "base")["result"]["run"]
    bag = tw("bag", rid, "--folds", "3")["result"]
    s = bag["summary"]
    assert s["val"]["roc_auc"] > 0.7 and s["n_oof"] > s.get("n_val", 0)
    rows = tw("runs")["result"]
    assert not any(r["kind"] == "fold" for r in rows) and any(r["kind"] == "bag" for r in rows)
    tw("ensemble")  # a bag can be an ensemble member (out-of-fold val predictions)
    final = tw("finalize", bag["run"])["result"]
    assert final["run"].startswith(tuple("0123456789"))
    tw("export", final["run"])
    pred = tw("predict", final["run"], "-i", "data.csv", "--limit", "3")["result"]
    assert len(pred) == 3


def test_date_strings_are_expanded_into_features(tmp_path):
    import pandas as pd
    from conftest import TW

    rng = np.random.default_rng(3)
    n = 1500
    ts = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24, n), unit="h")
    y = 10 * (ts.dayofweek >= 5) + 0.5 * ts.hour + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(
        {"when": ts.strftime("%Y-%m-%d %H:%M:%S"), "noise": rng.normal(size=n), "y": y}
    )
    df.to_csv(tmp_path / "d.csv", index=False)
    tw = TW(tmp_path)
    tw("init", "--data", "d.csv", "--target", "y")
    prof = tw("profile")["result"]
    assert prof["date_columns"] == {"when": "iso"}
    tw("split")
    base = tw("baseline", "--only", "dummy,hist_gbm")["result"]["leaderboard"]
    gbm = next(r for r in base if r["model"] == "baseline:hist_gbm")
    dummy = next(r for r in base if r["model"] == "baseline:dummy")
    assert gbm["val_rmse"] < 0.3 * dummy["val_rmse"]  # weekday/hour recovered from the string
    rid = tw("train", *FAST, "-o", "train.max_epochs=3", "-H", "dates")["result"]["run"]
    pred = tw("predict", rid, "-i", "d.csv", "--limit", "2")["result"]
    assert len(pred) == 2


def test_text_columns_in_tables_become_features(tmp_path):
    import pandas as pd
    from conftest import TW

    rng = np.random.default_rng(4)
    n = 1200
    words = {
        "red": ["cherry", "plum", "tannic", "oak"],
        "white": ["citrus", "crisp", "apple", "mineral"],
    }
    filler = [
        "a",
        "lovely",
        "wine",
        "with",
        "notes",
        "of",
        "and",
        "finish",
        "long",
        "bright",
        "the",
    ]
    label = rng.choice(["red", "white"], n)
    desc = [
        " ".join(
            rng.choice(filler, 8).tolist()
            + rng.choice(words[lab], 2).tolist()
            + rng.choice(filler, 6).tolist()
        )
        for lab in label
    ]
    df = pd.DataFrame(
        {
            "description": desc,
            "price": rng.normal(20, 5, n).round(2),
            "points": rng.integers(80, 100, n),
            "country": rng.choice(["fr", "it", "us"], n),
            "colour": label,
        }
    )
    df.to_csv(tmp_path / "w.csv", index=False)
    tw = TW(tmp_path)
    tw("init", "--data", "w.csv", "--target", "colour")
    prof = tw("profile")["result"]
    assert prof["text_features"] == ["description"]
    tw("split")
    rows = tw("baseline", "--only", "dummy,hist_gbm")["result"]["leaderboard"]
    gbm = next(r for r in rows if r["model"] == "baseline:hist_gbm")
    assert gbm["val_roc_auc"] > 0.95  # the label lives only in the text
    rid = tw("train", *FAST, "-o", "train.max_epochs=3", "-H", "text")["result"]["run"]
    tw("export", rid)
    pred = tw("predict", rid, "-i", "w.csv", "--limit", "3")["result"]
    assert len(pred) == 3


def test_guarded_step_is_killed_at_its_deadline(regression_project):
    import os
    import time

    from trainwright._utils import write_json
    from trainwright.autopilot import _guarded, state_path
    from trainwright.project import Project

    tw = regression_project
    tw("init", "--data", "houses.csv", "--target", "price")
    project = Project.load(tw.root)
    project.ensure_dirs()
    # a "running" auto owned by this (alive) process: `wait` would block for 10 minutes
    write_json(
        state_path(project),
        {"status": "running", "pid": os.getpid(), "deadline": time.time() + 999},
    )
    t0 = time.time()
    res, err = _guarded(project, ["wait", "--timeout", "10m"], 1)
    assert res is None and "killed" in err and time.time() - t0 < 60


def test_torch_and_onnx_predictions_match(tabular_project):
    import csv

    tw = tabular_project
    tw("init", "--data", "data.csv")
    tw("split")
    tw("baseline", "--only", "dummy,hist_gbm")
    tw("train", *FAST, "-o", "train.max_epochs=3", "-H", "a")
    tw("train", *FAST, "-o", "train.max_epochs=3", "-o", "train.seed=1", "-H", "b")
    bag = tw("bag", "003", "--folds", "2")["result"]["run"]  # multi-member path
    for ref in ("003", bag, "002"):
        out = {}
        for engine in ("torch", "onnx"):
            p = tw.root / f"p-{engine}.csv"
            tw("predict", ref, "-i", "data.csv", "--out", str(p), "--engine", engine, "--no-final")
            out[engine] = list(csv.DictReader(p.open()))
        assert [r["prediction"] for r in out["torch"]] == [r["prediction"] for r in out["onnx"]]
        diffs = [
            abs(float(a["confidence"]) - float(b["confidence"]))
            for a, b in zip(out["torch"], out["onnx"])
        ]
        assert max(diffs) < 1e-4


def test_errors_finds_group_structure_and_try_promotes_it(tmp_path):
    import pandas as pd
    from conftest import TW

    rng = np.random.default_rng(5)
    users, per = 120, 20
    user = np.repeat(np.arange(users), per)
    offset = rng.normal(scale=3.0, size=users)[user]
    x = offset + rng.normal(size=users * per)
    x2 = offset + rng.normal(size=users * per)
    y = np.where(x - offset + 0.3 * rng.normal(size=users * per) > 0, "up", "down")
    df = pd.DataFrame({"user": user, "x": x, "x2": x2, "noise": rng.normal(size=len(y)), "y": y})
    df.to_csv(tmp_path / "d.csv", index=False)
    tw = TW(tmp_path)
    tw("init", "--data", "d.csv", "--target", "y")
    tw("split")
    rep = tw("errors", "--seeds", "1")["result"]
    assert rep["worst_rows"] and (tmp_path / ".trainwright" / "val_errors.csv").exists()
    cand = next(c for c in rep["candidates"] if c["name"] == "group_user")
    assert cand["verdict"] == "better" and cand["delta"] > 0.05
    kept = tw("try", cand["file"], "--keep", "--seeds", "1", "--hypothesis", "relative x")
    assert kept["result"]["kept"] and (tmp_path / "features.py").exists()
    # with the statistics in features.py, errors stops proposing them
    again = tw("errors", "--seeds", "1")["result"]
    assert again["score"] >= cand["delta"] + rep["score"] - 1e-9
    assert not any(c["name"] == "group_user" for c in again["candidates"])
    dropped = tw("try", "--drop", "x,x_minus_user_mean,x_user_z", "--seeds", "1")["result"]
    assert dropped["verdict"] == "worse" and not dropped["kept"]
    trials = (tmp_path / ".trainwright" / "feature_trials.jsonl").read_text().splitlines()
    assert len(trials) == 2


def test_errors_on_a_regression_run_reports_slices(regression_project):
    tw = regression_project
    tw("init", "--data", "houses.csv", "--target", "price")
    tw("split")
    rid = tw("baseline", "--only", "hist_gbm")["result"]["runs"][0]
    rep = tw("errors", rid, "--no-candidates")["result"]
    assert rep["source"].startswith("run ") and rep["worst_rows"]
    assert "hot_slices" in rep and "biased_slices" in rep


def test_check_lists_what_is_missing_until_done(regression_project):
    tw = regression_project
    tw("init", "--data", "houses.csv", "--target", "price")

    def status():
        res = tw("check")["result"]
        return res["done"], {i["item"]: i["status"] for i in res["items"]}

    done, items = status()
    assert not done and items["profile"] == "fail"
    tw("profile")
    tw("split")
    tw("baseline", "--only", "dummy,hist_gbm")
    done, items = status()
    assert items["profile"] == "pass" and items["error analysis"] == "fail"
    assert items["search"] == "fail" and items["finalized"] == "fail" and items["test"] == "fail"
    tw("errors", "--no-candidates", "--seeds", "1")
    for col in ("x1", "x2", "x3"):
        tw("try", "--drop", col, "--seeds", "1")
    tw("decide", "search", "gradient boosting only: this test checks the checklist, not the model")
    best = tw("runs")["result"][0]["id"]
    tw("finalize", best)
    tw("evaluate", best, "--split", "test", "--reason", "final")
    done, items = status()
    assert done, items
    assert items["error analysis"] == "pass" and items["search"] == "pass"


def test_segmentation_lifecycle(tmp_path):
    from conftest import TW
    from PIL import Image

    rng = np.random.default_rng(6)
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i in range(48):
        img = (rng.random((40, 48, 3)) * 60 + 40).astype(np.uint8)
        m = np.zeros((40, 48), np.uint8)
        cy, cx, r = rng.integers(10, 30), rng.integers(10, 38), rng.integers(4, 9)
        sel = (abs(np.arange(40)[:, None] - cy) < r) & (abs(np.arange(48)[None] - cx) < r)
        img[sel], m[sel] = (220, 40, 40), 1
        Image.fromarray(img).save(tmp_path / "images" / f"s{i:02d}.jpg")
        Image.fromarray(m).save(tmp_path / "masks" / f"s{i:02d}.png")
    tw = TW(tmp_path)
    cfg = tw("init", "--data", "images", "--masks", "masks")["result"]["config"]
    assert cfg["task"] == "segmentation" and cfg["metric"] == "miou"
    prof = tw("profile")["result"]
    assert prof["target"]["n_classes"] == 2 and prof["target"]["classes"] == ["0", "1"]
    tw("split")
    tw("baseline")
    ov = ["-o", "model.params.pretrained=false", "-o", "preprocess.image_size=48"]
    run = tw("train", *FAST, *ov, "-o", "train.max_epochs=2", "-H", "seg")["result"]
    rid = run["run"]
    assert 0 <= run["summary"]["val"]["miou"] <= 1
    ev = tw("evaluate", rid)["result"]
    assert ev["per_class_iou"] and ev["worst_examples"]
    rep = tw("errors", rid)["result"]
    assert rep["per_class_iou"] and rep["worst_rows"]
    out = tw("predict", rid, "-i", "images", "--out", "pred", "--no-final")["result"]
    assert out["n"] == 48
    mask = np.asarray(Image.open(tmp_path / "pred" / "s00.png"))
    assert mask.shape == (40, 48) and set(np.unique(mask)) <= {0, 1}
    exp = tw("export", rid)["result"]
    assert exp["parity_ok"]


def test_tuned_gbm_baseline_finalizes(regression_project):
    tw = regression_project
    tw("init", "--data", "houses.csv", "--target", "price")
    tw("split")
    res = tw("baseline", "--only", "hist_gbm,gbm_tuned")["result"]["leaderboard"]
    tuned = next(r for r in res if r["model"] == "baseline:gbm_tuned")
    assert tuned["val_rmse"] > 0
    fin = tw("finalize", tuned["id"])["result"]
    assert fin["run"]


def test_tuned_tfidf_baseline(text_project):
    tw = text_project
    tw("init", "--data", "tickets.csv", "--target", "intent", "--text-column", "message")
    tw("split")
    res = tw("baseline", "--only", "tfidf_linear,tfidf_tuned")["result"]["leaderboard"]
    assert any(r["model"] == "baseline:tfidf_tuned" for r in res)
