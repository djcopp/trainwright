# trainwright playbook (for AI agents)

You are training a **small, task-specific model**. trainwright owns the training loop, the
data split, the metrics and the experiment log so your job is the part that needs judgement:
looking at data, forming hypotheses, reading diagnoses, and deciding what to try next.

## Hard rules

1. **Use trainwright for every step.** Do not hand-write training loops, ad-hoc sklearn/xgboost
   scripts, or custom evaluation code. Baselines are `trainwright baseline`; custom
   architectures go in `models.py` with `@tw.register_model` (see below).
2. **Never touch the test split while iterating.** Select models on **val**. Run
   `trainwright evaluate <run> --split test` once, for the final model, at the end.
3. **Every training run states a hypothesis**: `--hypothesis "dropout 0.3 reduces the overfitting seen in 004"`.
4. **Change one or two things per run**, starting from the best run: `--from-run <id> -o key=value`.
5. **Read the diagnosis and look at the curves after every run**: `trainwright diagnose <id>`
   and open `runs/<id>/curves.png` (it is an image; view it).
6. **Report honestly.** If a baseline beats the neural model, say so. Quote val vs test
   numbers correctly. Cost figures from `benchmark` are estimates; say which assumptions.

## The loop

```bash
trainwright status                      # where am I, what next? (run this whenever unsure)
trainwright init --data data.csv --target label     # once; writes trainwright.yaml
trainwright profile                     # read recommendations; fix data.drop / metric in trainwright.yaml
trainwright split                       # freeze train/val/test (once)
trainwright baseline                    # floor to beat (dummy, linear, gradient boosting / tf-idf)
trainwright train --hypothesis "default recipe"
trainwright diagnose latest             # findings + ready-to-run next experiments
trainwright train --from-run 004 -o optim.weight_decay=1e-3 --hypothesis "..."
trainwright tune --budget 20m           # optional: automated recipe search around the best run
trainwright runs                        # validation leaderboard
trainwright evaluate best               # val error analysis: confusions, worst examples, image grid
trainwright evaluate best --split test --reason "final model"   # ONCE
trainwright export best                 # ONNX bundle + FastAPI server + Dockerfile + model card
trainwright benchmark best              # latency, throughput, $/1M predictions vs the LLM
trainwright report                      # reports/REPORT.md for the human
```

Add `--json` to any command for machine-readable output. Every command prints `next:`
suggestions; follow them unless you have a reason not to.

## Long runs

The CLI blocks while training. For anything that may exceed your tool timeout:

- set a wall-clock budget: `-o train.time_budget=15m` (training stops gracefully, best checkpoint kept), and/or
- run it in the background and poll: `trainwright show latest` prints epochs so far.

Images are decoded once into `.trainwright/cache/`; the first image run is slower than later ones.

## What to try, by modality (in roughly this order)

**Tabular** (`tabular_mlp`): defaults are quantile-normalised numerics, categorical
embeddings, 3x256 residual MLP, AdamW + cosine, early stopping.
1. Fix data first: drop ids/leaks the profile flags.
2. Overfitting → `optim.weight_decay` up (1e-4 → 1e-3 → 1e-2), `model.params.dropout` 0.2-0.4, smaller `hidden`.
3. Underfitting / boosting baseline wins → `model.params.num_embedding=periodic`, wider (`hidden=512`), more epochs, `optim.lr=3e-3`.
4. Imbalance → `train.class_weights=balanced`. Skewed regression target → `preprocess.target_transform=log1p`.
5. `trainwright features <run>` shows permutation importance; drop noise features via `data.drop`.

**Image** (`image_pretrained`, resnet18, 224px): fine-tunes an ImageNet backbone with a 10x lower backbone LR.
1. Look at `.trainwright/samples.png` first. If left/right matters: `preprocess.hflip=false`.
2. Fast iteration: `preprocess.image_size=160` and/or `model.params.backbone=mobilenet_v3_large`.
3. Overfitting → `preprocess.augment=medium|strong`, `train.label_smoothing=0.1`, `train.freeze_backbone_epochs=2`.
4. Accuracy ceiling → `backbone=efficientnet_b0|convnext_tiny`, `optim.backbone_lr_mult=0.3`, more epochs.
5. Tiny / non-photo images (icons, scans, <=64px) → `model.name=image_cnn`.
6. Open `runs/<id>/eval/val_errors.png` to see what the model gets wrong; mislabelled data is common.

**Text** (`text_bag`, fastText-style): uni+bi-gram embeddings, mean+max pooling.
1. Compare against the `tfidf_linear` baseline: it is strong on small data; report it if it wins.
2. Overfitting → dropout 0.4-0.5, `preprocess.min_token_freq=3`.
3. Word order matters → `model.name=text_cnn`.

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

## Reading a diagnosis

| code | means | first thing to try |
|---|---|---|
| `overfitting` / `generalisation_gap` | memorising the train set | regularise (weight decay, dropout, augmentation) |
| `not_converged` | still improving at the last epoch | `train.max_epochs` x2 |
| `baseline_wins` | the neural model isn't earning its keep | capacity, periodic embeddings, longer training; else report it |
| `unstable` / `early_peak` | LR too high | `optim.lr` / 3 |
| `loss_flat` / `collapsed` / `diverged` | not learning | check data/labels; LR x3 or /10 |
| `minority_classes_weak` | majority class dominates | `train.class_weights=balanced` |
| `poorly_calibrated` | confidences unreliable | label smoothing; don't route on raw confidence |
