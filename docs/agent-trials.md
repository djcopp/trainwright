# Agent trials: does an AI agent build better models with trainwright?

trainwright was developed in a loop: give a coding agent a realistic, vague request on a
lesser-known public dataset, once **with** trainwright and once **without** it, score both on
data the agent never saw, read the transcripts, fix what went wrong, and repeat.

The harness is [`bench/agent_trials.py`](../bench/agent_trials.py).

## Method

- **Agent:** Claude Code in headless mode (`claude -p`, model `sonnet`), tools Bash/Read/Write/Edit/Glob/Grep,
  the user's global settings excluded, about 40 minutes per task.
- **Control:** a Python environment with numpy, pandas, scikit-learn, PyTorch, torchvision and matplotlib.
- **trainwright:** a frozen, non-editable install of trainwright plus `trainwright agent-setup`
  (AGENTS.md, CLAUDE.md import, Claude Code skill). The prompt is identical in both conditions.
- **Scoring:** 15% of each dataset is hidden before the agent starts. For images, whole
  near-duplicate groups are hidden, so burst photos can't leak. Both agents must write
  `predictions.csv` for the unlabelled holdout, which we score against the hidden labels.
- **Datasets:**

| task | data | prompt (abridged) |
|---|---|---|
| eye_movements | [OpenML 1044](https://www.openml.org/d/1044), 10,936 rows, 27 features, 3 classes | "do feature exploration and train a model to predict it" |
| superconduct | [OpenML 44148](https://www.openml.org/d/44148), 21,263 rows, 81 features, regression | "train a model that predicts the critical temperature" |
| banking77 | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77), 13,083 messages, 77 intents | "we route messages with an LLM call; train a small model to replace it" |
| car_parts | [Toyota Corolla car parts](https://huggingface.co/datasets/stevenalbert10/Toyota-Corolla-Car-Parts), 1,353 photos, 8 classes | "we identify parts with a Sonnet agent; can a smaller model run cheaply instead?" |

## Round 1 (trainwright v0.1)


| task | condition | holdout score | wall time | agent cost |
|---|---|---|---|---|
| eye_movements | control | **0.908** acc | 23.6 min | $0.23 |
| eye_movements | trainwright | 0.693 acc | 11.1 min | $0.17 |
| banking77 | control | **0.937** acc | 19.5 min | $0.29 |
| banking77 | trainwright | 0.884 acc | 13.7 min | $0.18 |

The control agents won both tasks. The transcripts show why:

1. **eye_movements: feature engineering.** The control agent noticed that every holdout
   assignment (`assgNo`) also appears in training and built per-assignment aggregates and
   deviations, taking accuracy from 70% to 86%. The trainwright agent followed the playbook's
   rule "use trainwright for every step". With no sanctioned place for feature engineering it
   could only tune models, and it plateaued at the raw-feature ceiling (MLP 0.715, statistically
   tied with gradient boosting 0.709, which it reported honestly).
2. **banking77: no pretrained text encoder.** The control agent fine-tuned `bge-small` (33M params)
   itself. trainwright v0.1 only had bag-of-n-grams and CNN text models, and the trainwright
   agent correctly reported that none beat TF-IDF. It even named a fine-tuned transformer as the
   obvious next step, but the framework didn't offer one.
3. **Process quality was better with trainwright** in both tasks. Both agents ran baselines,
   stated hypotheses, opened the curve images, read the diagnosis, evaluated test once (and
   flagged their own slip when they re-read it), reported ties as ties, and recommended routing
   low-confidence cases to the LLM. They were also 30–50% faster and cheaper.
4. **Harness bug:** `uv run` leaked the development checkout onto the agents' PATH, so part of
   the trainwright condition ran on in-progress code, which added noise-aware diagnosis. The
   harness now pins PATH and records `which trainwright`.

### Changes made because of round 1

| finding | change |
|---|---|
| playbook forbade feature engineering | `features.py` hook (fit on train, transform anywhere), applied to baselines, hashed into runs, shipped in bundles; playbook rewritten to encourage it |
| agents didn't consider whether groups recur at prediction time | `profile` detects entity-key columns (chance-adjusted feature similarity + name) and asks for an explicit seen/unseen-group decision |
| no pretrained text model | `text_pretrained` (MiniLM by default; any HF encoder) with a bundled tokenizer, now the default text model |
| "poorly calibrated" on every run; routing advice depended on confidences | temperature scaling fitted on val, applied in evaluate/export |
| control refit on all data | `trainwright finalize` retrains the chosen recipe on train+val |
| re-reading test numbers created ledger entries | cached test results |
| "overfitting" fired while the metric was still improving | separate `overconfident` finding |
| text cost estimate assumed image-sized prompts | per-modality LLM token estimate at init |

Checked manually on the same eye_movements data: with an assignment-aggregate `features.py`,
trainwright's MLP reaches 0.853 val accuracy (gradient boosting on the same features: 0.770)
and **0.876 on the hidden holdout** after `finalize`. On banking77, `text_pretrained` reaches
0.925 val macro-F1 against 0.873 for TF-IDF.

## Earlier finding: near-duplicate leakage (found before the trials)

While setting up the car-parts task, a linear model on 16×16 pixel thumbnails scored 0.80
macro-F1. The photos turned out to be bursts taken about 2 seconds apart, so a random split puts
near-identical frames in train and test. Grouping near-duplicates (256-bit dHash, Hamming ≤ 41)
dropped that baseline to 0.33. This became trainwright's default duplicate-aware split, and the
benchmark holdout hides whole groups.

## Round 2 (car_parts and superconduct)

| task | condition | holdout score | wall time | agent cost |
|---|---|---|---|---|
| car_parts | control | 0.996 acc / 0.991 macro-F1 | 6.3 min | $0.16 |
| car_parts | trainwright | 0.989 acc / **0.992** macro-F1 | 8.2 min | $0.20 |
| superconduct | control | **RMSE 9.60** (R² 0.923) | 8.3 min | $0.13 |
| superconduct | trainwright | RMSE 10.87 (R² 0.901) | 8.4 min | $0.16 |

**Harness note:** a second PATH bug meant rounds 1 and 2 ran the trainwright condition against
the *live development checkout*, not the frozen build. Round 2 therefore used roughly the v0.2
feature set, plus some of the in-progress v0.3 changes. The harness now verifies and records
`which trainwright`, and round 3 used a frozen, non-editable v0.3.0 install.

- **car_parts: tie** (2 images out of 260). The control agent's best move was frozen
  **DINOv2-small** embeddings with logistic regression. It also noticed the burst photos and
  grouped its CV folds by capture time. The trainwright agent fine-tuned ResNet-18, read the
  `early_peak` diagnosis, lowered the LR, evaluated test once, finalized and benchmarked, and
  recommended routing the 2–4% of images under 0.9 confidence to the LLM.
- **superconduct: control better.** Both agents found that gradient boosting beats an MLP here.
  The trainwright agent correctly shipped the HistGBM baseline because the MLP was statistically
  tied and the baseline is simpler. But `finalize` only handled neural runs, so it shipped a model
  trained on 70% of the data with scikit-learn's default 100 boosting rounds. The control agent
  refit on all the data with 1,500 rounds.
- The group-key detector flagged `range_atomic_radius`, a physical measurement. The agent
  correctly ignored it.

### Changes made because of round 2

| finding | change |
|---|---|
| frozen self-supervised features win on small image sets | `backbone=dinov2_small/base` and `freeze_backbone=true` (linear probe); playbook step |
| shipped baseline was undertrained and fit on 70% of the data | HistGBM baseline uses up to 1,000 early-stopped rounds; `finalize` refits baselines on train+val |
| `evaluate --split test` warned about choosing a run tied with the leader | warning suppressed for statistically tied runs |
| measurement flagged as a group key | detector requires the relationship to be non-monotonic (ID-like), not smooth |
| ONNX weights in a sidecar file were reported as a 0.13 MB model | single-file bundles under 1.8 GB; size counts all model files |
| the MCP server failed on `mcp` 2.x (`FastMCP` renamed) | supports both versions; tested |

## Round 3 (trainwright v0.3.0, frozen and verified)

All four tasks were re-run in the trainwright condition against a frozen, non-editable v0.3.0
install (`which trainwright` recorded), in folders outside the repo. Controls are reused from
rounds 1–2, since their environment didn't change.

| task | metric | trainwright (r1/r2) | **trainwright (r3)** | control | wall time r3 vs control |
|---|---|---|---|---|---|
| eye_movements | accuracy | 0.693 | **0.865** | 0.908 | 7.3 vs 23.6 min |
| banking77 | accuracy | 0.884 | **0.930** | 0.937 | 25.8 vs 19.5 min |
| superconduct | RMSE ↓ | 10.87 | **10.16** | 9.60 | 13.9 vs 8.3 min |
| car_parts | macro-F1 | 0.992 | 0.971 | 0.991 | 11.5 vs 6.3 min |

- **eye_movements:** `profile` flagged `assgNo`. The agent checked that every holdout assignment
  appears in training, kept the random split on purpose, wrote a `features.py` with
  per-assignment means and deviations, and picked the simpler of two tied runs. The MLP beat
  gradient boosting on the same features by 6.6 points (val 84.4% vs 77.8%). Remaining gap to
  control: the control also used holdout rows' features for its group statistics (transductive)
  and tuned longer.
- **banking77:** the default `text_pretrained` (MiniLM) took the agent from 0.884 to 0.930. It
  also tried bge-small, found it statistically tied, and shipped the smaller model. The gap to
  control (bge, refit on all data) is about one standard error on 1,960 messages.
- **superconduct:** both conditions shipped gradient boosting. The remaining gap is training
  data. `finalize` kept the test split out of the production model (about 72% of the rows vs the
  control's 85%), even though test had already been scored.
- **car_parts:** a regression caused by the round-2 playbook change. The agent followed the new
  "try a DINOv2 linear probe first" step and made a single run. But `freeze_backbone` reused the
  fine-tuning recipe (LR 1e-3, dropout, label smoothing, 15 epochs), which under-trains a probe
  head (test macro-F1 0.933).

### Changes made because of round 3

| finding | change |
|---|---|
| linear probe under-trained | dedicated `image_probe` model with the standard probe recipe (LR 1e-2, no dropout/smoothing, 30 epochs) |
| production model withheld already-scored test data | `finalize` includes test automatically once the test evaluation is recorded (`--no-include-test` to opt out) |
| exporting a production model warned "no test evaluation" | bundles and model cards report the parent run's held-out metrics |
| benchmark throughput was dominated by resizing 12 MP photos, not the model | benchmark reports model-only vs end-to-end throughput and what dominates |

<!-- ROUND4 -->


