# trainwright

**Agent-first PyTorch training for small, task-specific models.**

A lot of production LLM calls are narrow, repetitive jobs: route this ticket, classify this photo,
score this row. A small model trained for that one task can be as accurate, thousands of times cheaper
and a hundred times faster. Coding agents can build those models, but left to themselves the result
depends on the run: a different stack each time, a default scikit-learn model with no comparison,
a test set peeked at during tuning, no record of what was tried, and nothing deployable at the end.

trainwright is a framework *and* an operating procedure for agents. Like PyTorch Lightning it
standardises the training loop and the data plumbing. Unlike Lightning, every command is built to
be driven by an agent: it returns structured output, refuses steps done out of order, diagnoses
each run, and tells the agent what to try next.

```
profile → decide deployment scenario → split (frozen, duplicate-aware) → baseline
        → train (hypothesis) → diagnose → iterate (features.py, models, tune)
        → evaluate on the sealed test set once → finalize → export (ONNX + server) → benchmark vs the LLM
```

## Does it help? Agent A/B trials

We gave Claude Code (Sonnet) the same vague, realistic request on four lesser-known public datasets,
once with a plain Python ML environment (**control**) and once with trainwright installed. We scored
both on a 15% holdout the agents never saw. Then we read the transcripts, fixed what trainwright got wrong,
and ran it again. Full method, transcripts summary and every change are in
[docs/agent-trials.md](docs/agent-trials.md).

<!-- RESULTS_TABLE -->

What the loop taught us, and what is now built in:

- **Leakage is everywhere and invisible.** The car-parts photos are bursts taken 2 s apart. A random
  split lets a 16×16-pixel linear model score 0.80 macro-F1, and grouping near-duplicates drops it
  to 0.33. Splits are now duplicate-aware by default (perceptual hashes for images, exact matches for rows and text).
- **The playbook must not forbid the winning move.** In round 1 the control agent won eye_movements
  with per-assignment feature engineering, which trainwright had no place for. `features.py` is now
  first-class, and `profile` flags entity keys and asks the agent whether future rows come from seen
  or unseen groups. In round 3 the trainwright agent did exactly that and rose from 0.69 to 0.86.
- **Ship the strongest model, not your favourite.** When an MLP only ties gradient boosting,
  trainwright says so (bootstrap standard errors everywhere). The agent then ships the simpler model, and `finalize`
  refits it on train+val.
- **Single runs are partly luck, for models and for agents.** Three seeds of one MLP recipe ranged
  0.797–0.853 val accuracy (bootstrap SE 0.009), and two identical straight prompts scored 0.908 and
  0.696. `trainwright ensemble --seeds 3` averages seeds and combines runs with greedy weights, and
  the harness runs every condition twice.
- **Give agents modern defaults.** Pretrained text encoders (MiniLM, bge) and frozen DINOv2 features
  were the control agents' best tools, so both are now one override away.

## Install

```bash
uv add "trainwright[all]"          # or: pip install "trainwright[all]"
```

Extras: `vision` (torchvision backbones), `text` (pretrained transformer encoders: MiniLM, bge, DINOv2),
`onnx` + `serve` (export and FastAPI serving), `label` (LLM labelling via the Claude API), `mcp` (MCP server).
Python 3.10+. CUDA, Apple Silicon (MPS) and CPU are picked automatically.

## Use it with an agent

```bash
cd my-repo
trainwright agent-setup            # writes AGENTS.md, a CLAUDE.md import and a Claude Code skill
```

Then ask the way you would anyway:

> "We identify car parts in photos with a Sonnet agent and it's getting expensive. The labelled photos are in
> images/. Can we train a smaller model that runs cheaply in the cloud?"

Here is what the agent did in trial round 2. The skill launched on its own, and the excerpt below is
abridged but verbatim from the transcript:

```text
$ trainwright init --data images && trainwright profile
1092 examples (image, classification)
target: 8 classes, imbalance 23.44:1
  * 67% of examples have a duplicate or near-duplicate (image dHash-256, hamming <= 41; 422 distinct
    groups, largest 85). The split keeps each group inside one split (split.dedup=true) so scores
    are not inflated by near-copies.
$ trainwright split && trainwright baseline
baseline pixels_linear  val_macro_f1=0.7307
$ trainwright train --hypothesis "default resnet18 recipe beats pixel baseline" -o train.time_budget=6m
run 003-image-pretrained: best val_macro_f1=0.9704 at epoch 2/7
  - [warning] early_peak: Best epoch was 2; validation got worse immediately afterwards. LR is likely
    too high or the model is far too large for the data.
  - [info] noisy_validation: ... bootstrap standard error of ±0.024 (259 examples) ...
$ trainwright train --from-run 003-image-pretrained -o optim.lr=0.000333 \
      --hypothesis "early_peak: lower LR preserves pretrained features"
#   run                         kind      val_macro_f1    params  time  epochs
1   004-image-pretrained        neural    0.9835 ±0.0072  11.18M  119s  7/12
2≈  003-image-pretrained        neural    0.9704 ±0.0244  11.18M  57s   2/7
3   002-baseline-pixels-linear  baseline  0.7307 ±0.0613  -       0s    -
$ trainwright evaluate best --split test --reason "final model"        # once
004-image-pretrained on test (n=175): macro_f1=0.9955
$ trainwright finalize best && trainwright export 005-final-004-image-pretrained
$ trainwright benchmark 005-final-004-image-pretrained
cost per 1M predictions (ESTIMATE; edit `cost:` in trainwright.yaml):
  small model on 2 vCPU / 4 GB general-purpose cloud VM @ $0.05/h: $4.8850
  claude-sonnet-5-5 @ 1850 in / 30 out tokens: $4,000.00
```

On the hidden holdout this model scored 0.992 macro-F1. The agent's report also recommended
sending the 2–4% of photos below 0.9 confidence to the existing LLM.

<p align="center">
  <img src="docs/assets/car_parts_samples.jpg" width="430" alt="sample photos per class">
  <img src="docs/assets/car_parts_confusion.png" width="330" alt="validation confusion matrix">
</p>
<p align="center"><img src="docs/assets/car_parts_curves.png" width="900" alt="training curves"></p>
<p align="center"><sub>Photos: <a href="https://huggingface.co/datasets/stevenalbert10/Toyota-Corolla-Car-Parts">Toyota Corolla car parts</a> (Apache-2.0).</sub></p>

There is also an MCP server (`claude mcp add trainwright -- trainwright mcp`) exposing the same steps as tools.

## Use it yourself

```bash
trainwright init --data tickets.csv --target intent   # infers task, modality and the right metric
trainwright profile                                   # leakage, duplicates, group keys, imbalance, ids
trainwright split                                     # frozen, stratified, duplicate-aware
trainwright baseline                                  # dummy / linear / gradient boosting / tf-idf
trainwright train --hypothesis "default recipe"
trainwright diagnose latest                           # findings + ready-to-run next experiments
trainwright train --from-run 004 -o model.params.dropout=0.3 --hypothesis "regularise the gap seen in 004"
trainwright tune --budget 20m                         # successive-halving recipe search
trainwright ensemble --seeds 3                        # seed replicas + greedy-weighted ensemble
trainwright runs                                      # leaderboard with ±standard errors, ≈ marks ties
trainwright evaluate best --split test --reason "final model"
trainwright finalize best                             # retrain the chosen recipe on train+val
trainwright export <final-run> && trainwright benchmark <final-run> && trainwright report
trainwright predict <run> -i new.csv --out predictions.csv
```

Every command takes `--json` and ends with `next:` suggestions. `trainwright status` shows where a
project is in the lifecycle at any time.

### Feature engineering lives in `features.py`

```python
import pandas as pd

def fit(train: pd.DataFrame) -> dict:                       # sees the TRAIN split only
    return {"store_mean": train.groupby("store")["basket"].mean().to_dict()}

def transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:   # also runs at serving time
    df = df.copy()
    df["store_mean"] = df["store"].map(state["store_mean"])
    return df
```

It's fitted on train only, applied to every split and to the baselines, hashed into each run, and
shipped (source plus fitted state) inside the exported bundle.

## What's inside

| | |
|---|---|
| **Data** | CSV/Parquet tables, image folders or manifests, text columns. Numpy-only preprocessing (quantile-normalised numerics, categorical vocabularies, word n-gram or HF tokenizers) fitted on train only and shipped inside the bundle. |
| **Splits** | Frozen once per project with a data fingerprint. Stratified, group or time-based. **Duplicate-aware by default.** `profile` flags entity keys (users, sessions, assignments) and asks whether predictions will be for seen or unseen groups. |
| **Baselines** | Dummy, linear, early-stopped HistGradientBoosting (tabular), TF-IDF + logistic (text), pixel-linear (images), recorded as runs on the same split and metrics, and shippable. |
| **Models** | `tabular_mlp` (categorical embeddings, periodic numeric embeddings, residual blocks), `image_pretrained` (ResNet, MobileNetV3, EfficientNet, ConvNeXt, DINOv2; fine-tune or linear probe), `image_cnn`, `text_pretrained` (MiniLM, bge or any HF encoder), `text_bag`, `text_cnn`. Switching model also applies that model's recipe (e.g. a 20× lower encoder LR). Register your own in `models.py`. |
| **Training** | One `Trainer`: AMP, gradient clipping and accumulation, cosine/one-cycle with warm-up, discriminative LR for pretrained backbones, early stopping, wall-clock budgets, non-finite-loss detection, best checkpointing, on-device image augmentation, cached image decoding. |
| **Tracking** | Plain files, no server: `runs/<id>/{meta.json, config.yaml, metrics.jsonl, summary.json, diagnosis.json, curves.png, checkpoints/}` with hypothesis, parent run, git SHA, environment, data fingerprint, split hash, `features.py` hash and label-correction hash. |
| **Diagnosis** | Overfitting vs over-confidence, generalisation gap, non-convergence, instability, early peaks, divergence, collapse, weak minority classes, calibration, baseline wins and ties, all **noise-aware** (bootstrap standard errors). Each finding comes with concrete overrides and ready-to-run commands. |
| **Ensembles** | `ensemble` picks members from the top runs by greedy forward selection on val log-loss (Caruana et al.), adds seed replicas on request and reports the seed spread. Ensembles evaluate, finalize and export like any run (a bundle of bundles averaged by the runtime). |
| **Evaluation** | Per-class report, confusions, worst examples (an image grid for vision), and accuracy vs coverage for routing low-confidence cases to an LLM. Confidences are temperature-calibrated on val. The test split is sealed: every access is logged and reuse is flagged. |
| **Deployment** | `finalize` (refit on train+val), ONNX export with a parity check, a torch-free runtime (numpy + onnxruntime), a generated FastAPI server, Dockerfile and model card. `benchmark` measures latency and throughput and estimates $/1M predictions against the LLM call it replaces, with every assumption printed. |
| **Distillation** | `label` labels data with Claude under a hard spend cap. `review` ranks likely label errors and can ask a stronger model to adjudicate them blind. `data.corrections` applies fixes without breaking the split. |

## Python API

The building blocks work on their own, Lightning-style:

```python
import trainwright as tw

task = tw.ClassificationTask(MyNet(), n_classes=10)
train = tw.TensorLoader({"x": x_train, "y": y_train}, batch_size=64, shuffle=True)
val = tw.TensorLoader({"x": x_val, "y": y_val}, batch_size=256)
trainer = tw.Trainer(max_epochs=30, monitor="val_accuracy", mode="max", patience=5)
result = trainer.fit(task, train, val)
```

Batches are dicts: `"y"` is the target, and every other key is passed to `forward` as a keyword argument.

## Examples

- [examples/car_parts](examples/car_parts): replace a vision-LLM call with a small image classifier.
- [examples/banking77](examples/banking77): replace an LLM intent router; distil from LLM labels with `label` and `review`.

## Limitations

- Single-machine training; no distributed or multi-GPU support yet.
- Classification and regression only (no detection, segmentation, multi-label or generation).
- Cost figures are estimates from your own assumptions in `trainwright.yaml`, not quotes.
- The agent trials are small (one run per condition, four datasets) and show what happened, not a
  statistical guarantee. The harness is included so you can run your own.

## Development

```bash
uv sync --all-extras
uv run pytest                     # unit + end-to-end lifecycle tests for every modality
uv run ruff check src tests && uv run mypy src
uv run python bench/agent_trials.py --help   # the agent A/B harness
```

## Licence

Apache-2.0
