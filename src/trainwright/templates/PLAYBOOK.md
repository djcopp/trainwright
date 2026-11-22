# trainwright playbook (for AI agents)

You are training a **small, task-specific model**. trainwright owns the training loop, the
data split, the metrics and the experiment log, so your job is the part that needs judgement:
looking at the data, deciding how the model will be used, engineering features, forming
hypotheses, reading diagnoses and deciding what to try next.

## Rules

1. **trainwright owns splitting, training, evaluation and tracking.** Don't write your own
   training loops, train/test splits or scoring code: results must land in `runs/` on the frozen
   split to be comparable. Exploratory analysis with pandas/plots is fine and encouraged.
   **Feature engineering goes in `features.py`** and custom architectures go in `models.py`;
   both are first-class (see below).
2. **Never touch the test split while iterating.** Select models on **val**. Run
   `trainwright evaluate <run> --split test` once, for the final model. Re-running it for
   the same run returns the cached result. `trainwright show <run>` also displays it.
3. **Every training run states a hypothesis**: `--hypothesis "dropout 0.3 reduces the overfitting seen in 004"`.
4. **Change one or two things per run**, starting from the best run: `--from-run <id> -o key=value`.
5. **Read the diagnosis and look at the curves after every run**: `trainwright diagnose <id>`
   and open `runs/<id>/curves.png` (it is an image; view it).
6. **Respect the noise.** The leaderboard shows ±standard errors and marks runs tied with the
   leader with `≈`. Don't chase differences inside the noise band. Among tied runs prefer the
   simpler or cheaper one (including a baseline).
7. **Report honestly.** If a baseline wins or ties, say so. Quote val vs test numbers
   correctly. Cost figures from `benchmark` are estimates; state the assumptions.

## The loop

```bash
trainwright init --data data.csv --target label     # once; writes trainwright.yaml
trainwright profile                     # READ it: leaks, duplicates, group keys, recommendations
#   -> decide the deployment scenario (below); fix data.drop / data.group_column, or
#      record the call: trainwright decide deployment "predictions are for ... because ..."
trainwright split
trainwright errors                      # where a fast proxy model is wrong, plus candidate
                                        #   features it built and measured itself
trainwright try my_features.py --hypothesis "..." --keep   # seconds per idea; repeat
trainwright auto --background           # the search; runs until new rounds stop improving
trainwright wait --timeout 7m           # repeat until it reports "auto finished"
trainwright errors best                 # the leader's mistakes: next round of ideas
trainwright evaluate best --split test --reason "final model"   # ONCE, at the end
trainwright predict -i holdout.csv --out predictions.csv --columns prediction
trainwright check                       # the definition of done: repeat until DONE
#   -> then a reviewer subagent checks your report against the records (below)
trainwright export final && trainwright benchmark final         # ONNX bundle, server, cost vs LLM
```

`trainwright status` tells you where you are. Add `--json` to any command for machine-readable
output. Every command prints `next:` suggestions.

## Where the gains are

A model search moves the score by a percent or two. Information the model doesn't have moves it
by ten or more: on one eye-tracking task, features relative to each assignment's averages
took a gradient-boosted model from 0.72 to 0.81 accuracy. Search is automated; this part is
your job.

1. `trainwright errors` fits early-stopped HistGBM on the current `features.py` in seconds, then
   shows the slices holding most of the loss, slices with systematic bias, confused classes and
   the worst rows (all rows: `.trainwright/val_errors.csv`, load it with pandas). It also writes
   and measures candidates it can build itself: per-group statistics for entity keys and a log
   target. `auto` adopts candidates that are clearly better (z > 2) on its own.
2. Read the slices and worst rows like a domain expert. Why would a model miss these? A missing
   interaction, a ratio, a count per key, a date part, a unit problem, a label error?
3. Write the idea as a `features.py` candidate and `trainwright try cand.py --hypothesis "..."`.
   The proxy reports the gain with a paired-bootstrap error bar. `--keep` promotes it only if
   it is better; `--drop col1,col2` tests removing columns. Every trial is logged in
   `.trainwright/feature_trials.jsonl`.
4. Keep going while ideas still come from the evidence (`check` asks for at least three tested
   ideas, or a recorded reason to stop). Then run `auto`: without `--budget` it keeps adding
   rounds while they beat the leader by more than noise. Run `errors best` on the leader after.
5. Pick the leader on val (ties → simpler/cheaper), evaluate test once and predict.

Targeted runs on the leader stay available: `trainwright train --from-run <id> -o key=value
--hypothesis "..."`, `tune`, `ensemble --seeds 3`, `diagnose <id>` (training curves).
Text and image projects: `trainwright errors best` lists the worst examples with their text or
path; read them. Mislabels are common (`trainwright review`).

## Always have a deliverable

As soon as `auto` (or any run) has a leader, write the requested predictions with it, then keep
improving and overwrite them. `trainwright predict` works on the best run so far even while
`auto` is still running. A better model that isn't delivered scores nothing.

## When you are done

Finished is a checklist, not a clock. `trainwright check` reads the project's own records and
lists what is missing: the profile, leaks dropped or explained, a deployment decision, error
analysis with tested ideas, a converged search, a finalized leader, one test evaluation.
Judgement calls go on the record: `trainwright decide <topic> "<decision and evidence>"`
(topics: `deployment`, `features`, `search`, `model`, `leak:<column>`).

When `check` says DONE, **spawn a reviewer subagent** before you report. Give it the `check`
output, `trainwright runs`, `.trainwright/decisions.jsonl` and your draft report, and ask it to:

- check every number in the report against the records (val vs test, which model is served);
- check the split matches how the model will be used;
- name one idea the evidence points to that was not tested, if there is one;
- answer PASS or a list of fixes.

Fix what it lists (test the idea with `try`, rerun `auto`) and review again until it passes.

## Decide the deployment scenario first

Validation must mimic how the model will be used, or every number is wrong.

- **New entities at prediction time?** If `profile` flags a grouping key (user, session,
  patient, assignment, store...) and future predictions are for *unseen* values of it, set
  `data.group_column: <key>` and `trainwright split --force`. If predictions are for entities
  already in the training data, the default split is right, and group-level features (below)
  can be a large win. If the task doesn't say, choose the cautious option and say which you chose.
- **Forecasting the future?** Set `data.time_column` for a chronological split.
- **Duplicates/near-duplicates** (burst photos, repeated rows or messages) are grouped
  automatically so they never straddle splits. Check `profile`'s duplicate line.

## Event logs: `trainwright rollup`

If the data is a log (transactions, sessions, tickets) and predictions are per customer or user,
build one row per entity first, using only events before the prediction date:

```bash
trainwright rollup --events transactions.csv --key "Customer ID" --time InvoiceDate \
  --entities customers.csv --entity-key customer_id --cutoff 2011-09-01 \
  --expr "revenue=Quantity*Price" --out customer_features.csv
trainwright init --data customer_features.csv --target <label>
```

It adds counts, recency, tenure, 7/30/90/365-day windows, numeric sums/means and categorical
diversity per entity. Run it with the same arguments on fresh events at prediction time (for the
holdout entities: `--entities holdout.csv`). Add domain features on top in `features.py`.

## Feature engineering (tabular): `features.py`

```python
import pandas as pd


def fit(train: pd.DataFrame) -> dict:  # optional; sees the TRAIN split only
    g = train.groupby("assgNo")[["fixcount", "totalFixDur"]].mean()
    return {"assg_means": {str(k): v for k, v in g.to_dict(orient="index").items()}}


def transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:  # must work on one row too
    df = df.copy()
    means = pd.DataFrame.from_dict(state["assg_means"], orient="index")
    for c in means.columns:
        m = df["assgNo"].astype(str).map(means[c])
        df[f"{c}_assg_mean"] = m
        df[f"{c}_minus_assg_mean"] = df[c] - m
    df["dur_per_fix"] = df["totalFixDur"] / (df["fixcount"] + 1)
    return df
```

It is fitted on train only, applied to every split, used by the baselines too, hashed into
each run, and shipped inside exported bundles. Ideas that often win: group aggregates and
deviations from them (only when groups are seen at prediction time), ratios and differences,
date parts, counts per key. Use `trainwright features <run>` (permutation importance) to see
what the model relies on, and drop noise via `data.drop`.

## What to try, by modality (in roughly this order)

**Tabular** (`tabular_mlp`): quantile-normalised numerics with periodic embeddings, categorical
embeddings, a 3x256 residual MLP, AdamW + cosine, early stopping, and at least 2000 optimiser
steps however small the data (`train.min_steps`). Across 76 OpenML datasets this default beats
early-stopped gradient boosting on val ~3 times in 4.
1. Fix the data first: drop leaks the profile flags; decide the deployment scenario; add `features.py`.
2. Smooth targets or tiny noisy data: compare `model.params.num_embedding=none`. More capacity: `hidden=512`.
3. Overfitting → `optim.weight_decay` up (1e-4 → 1e-3 → 1e-2), `model.params.dropout` 0.2-0.4.
4. Imbalance → `train.class_weights=balanced`. Skewed regression target → `preprocess.target_transform=log1p`.
5. `auto` also trains `gbm_tuned` (gradient boosting tuned on val) and blends it with the neural runs.
   If it still wins on its own, ship it: `trainwright finalize <baseline-run>` then `export`.

**Image** (`image_pretrained`, resnet18 at 224px): fine-tunes an ImageNet backbone with a 10x lower backbone LR.
1. Look at `.trainwright/samples.png` first. If left/right matters: `preprocess.hflip=false`.
2. Small dataset (a few thousand images or fewer): compare a linear probe on self-supervised features,
   `-o model.name=image_probe` (frozen DINOv2 + linear head; fast), with the fine-tuned default. Keep
   whichever is better on val; with a small val set run both before concluding anything.
3. Fast iteration: `preprocess.image_size=160` and/or `model.params.backbone=mobilenet_v3_large`.
4. Overfitting → `preprocess.augment=medium|strong`, `train.label_smoothing=0.1`, `train.freeze_backbone_epochs=2`.
5. Accuracy ceiling → `backbone=efficientnet_b0|convnext_tiny`, `optim.backbone_lr_mult=0.3`, more epochs.
6. Tiny / non-photo images (icons, scans, <=64px) → `model.name=image_cnn`.
7. Open `runs/<id>/eval/val_errors.png` to see mistakes; mislabelled data is common (`trainwright review`).

**Text** (`text_pretrained`: fine-tuned bge-small, when `trainwright[text]` is installed; else `text_bag`).
1. `auto` also trains `tfidf_tuned` (words + characters, tuned on val) and blends it in: it is strong on
   short texts and small data.
2. Faster/smaller → `model.params.encoder=sentence-transformers/all-MiniLM-L6-v2` (22M); more accuracy → `BAAI/bge-base-en-v1.5`.
   Text models train slowly on CPU/MPS: set `train.time_budget`, then `finalize` (it retrains, so budget for it).
3. Need microsecond inference or no transformers → `model.name=text_bag` / `text_cnn`.

**Segmentation** (`trainwright init --data images --masks masks`): one PNG mask per image with
the same file stem, pixel value = class index, 255 = ignored. Selection is on mIoU.
1. `profile` lists each class's pixel share. Rare classes dominate mIoU, so check them first.
2. `auto` tries `seg_probe` (frozen DINOv2 + linear head), `image_segmenter` (LR-ASPP
   MobileNetV3, fine-tuned) and DeepLabV3-ResNet50, and keeps the best.
3. `trainwright errors best` lists the weakest classes, the most-confused class pairs and the
   worst images. Look at those images next to their masks: label noise is common.
4. Detail matters: try `-o preprocess.image_size=448` (or 512) on the leader.
5. `trainwright predict <run> -i photos/ --out masks/` writes one PNG mask per photo at its
   original size.

## Distilling an LLM

If labels come from an LLM (or you create them with `trainwright label`), metrics measure
**agreement with that LLM**. After training, `trainwright review` lists items where the
model confidently disagrees with the label (mostly label errors). Fix them in `corrections.csv`
(`data.corrections`). Confidences are temperature-calibrated on val, so `evaluate` shows how
much traffic the small model can handle at a given accuracy, with the rest routed to the LLM.

## Custom models

Create `models.py` in the project root; it is imported automatically:

```python
import torch, trainwright as tw


@tw.register_model("my_mlp", modality="tabular", description="...")
class MyMLP(torch.nn.Module):
    def __init__(self, spec, hidden=128):
        super().__init__()
        d = spec["n_num"] + 8 * len(spec["cat_cardinalities"])
        self.embs = torch.nn.ModuleList(torch.nn.Embedding(c, 8) for c in spec["cat_cardinalities"])
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, spec["n_outputs"])
        )

    def forward(self, x_num, x_cat):
        parts = [x_num] + [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        return self.net(torch.cat(parts, 1))
```

Then `trainwright train -o model.name=my_mlp -o model.params.hidden=256 --hypothesis "..."`.
`forward` receives the batch keys: tabular `x_num, x_cat`; image `image` (float [B,3,H,W] in 0..1);
text `tokens` (int64 [B,L], 0 = pad). Return logits `[B, n_outputs]`.

## Long runs

The CLI blocks while training. `trainwright auto --background` + `trainwright wait` handles the
search. For single runs that may exceed your tool timeout:

- set a wall-clock budget: `-o train.time_budget=15m` (training stops gracefully, best checkpoint kept), and/or
- run it in the background and poll: `trainwright show latest` prints epochs so far.

Images are decoded once into `.trainwright/cache/`, so the first image run is slower than later ones.

## Reading a diagnosis

| code | means | first thing to try |
|---|---|---|
| `overfitting` / `generalisation_gap` | memorising the train set | regularise (weight decay, dropout, augmentation) |
| `overconfident` | val loss up, metric flat | fine as is (calibration is applied); label smoothing |
| `not_converged` | still improving at the last epoch | `train.max_epochs` x2 |
| `baseline_wins` / `tied_with_baseline` | the neural model isn't earning its keep | features, capacity, periodic embeddings; else ship the baseline |
| `noisy_validation` | val set too small to separate runs | stop fine-tuning hyper-parameters; prefer simpler |
| `unstable` / `early_peak` | LR too high | `optim.lr` / 3 |
| `loss_flat` / `collapsed` / `diverged` | not learning | check data/labels; LR x3 or /10 |
| `minority_classes_weak` | majority class dominates | `train.class_weights=balanced` |
