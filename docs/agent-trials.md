# Agent trials

The same vague request goes to Claude Code twice, once with trainwright and once without. Both
predict a hidden holdout. Then the transcripts are read and the failures fixed. Harness:
[`bench/agent_trials.py`](../bench/agent_trials.py).

## Method

| | |
|---|---|
| agent | `claude -p --model sonnet`, tools Bash/Read/Write/Edit/Glob/Grep, user settings excluded, ~40 min |
| prompt only | Python with numpy, pandas, scikit-learn, PyTorch, torchvision, matplotlib |
| with trainwright | the same, plus a frozen non-editable trainwright install and `trainwright agent-setup` |
| scoring | 15% hidden before the agent starts (whole near-duplicate groups for images); the agent writes `predictions.csv` |
| isolation | trial folders outside the repo; PATH pinned; `which trainwright` recorded |

## Results by round

Holdout scores. **Bold** = better side. "→" = before → after the fix.

| round | version | task | prompt only | trainwright | what the transcripts showed | fix |
|---|---|---|---|---|---|---|
| 1 | 0.1 | eye_movements | **0.908** | 0.693 | control built per-assignment features; the playbook said "trainwright for every step" | `features.py` hook; entity-key detection in `profile` |
| 1 | 0.1 | banking77 | **0.937** | 0.884 | control fine-tuned bge-small; trainwright had no pretrained text model | `text_pretrained`; temperature scaling; `finalize` |
| 2 | 0.2* | car_parts | 0.991 F1 | **0.992 F1** | control used frozen DINOv2 + logistic regression | DINOv2 backbones, linear probe |
| 2 | 0.2* | superconduct | **9.60** | 10.87 | shipped HistGBM had 100 rounds, trained on 70% of rows | 1000 early-stopped rounds; `finalize` refits baselines |
| 3 | 0.3.0 | car_parts | **0.991 F1** | 0.971 F1 | probe reused the fine-tune recipe (LR 1e-3, 15 epochs) | `image_probe` with a probe recipe |
| 3 | 0.3.0 | superconduct | **9.60** | 10.16 | production model left out already-scored test rows | `finalize` adds test once it is scored |
| 4 | 0.3.1 | car_parts, superconduct | 0.991 F1, 9.60 | **1.000 F1**, 9.83 | | |
| 5 | 0.4.0 | eye_movements ×2 | 0.908, 0.696 | **0.859, 0.848** | the prompt-only agent finds the group trick only sometimes; agents skipped `finalize` | `predict` serves the finalized model; bge-small default |
| 6 | 0.4.1 | banking77 ×2 | 0.931, 0.937 | 0.932, 0.927 | bge-small got 2 epochs: 128-token padding, fp32 | auto `max_length`; bf16 on Apple Silicon (90 s → 26 s/epoch) |
| 7 | 0.4.2 | banking77 ×2 | **0.931, 0.937** | 0.932, 0.933 | tie | |
| 8 | 0.5.0 | 8 new tasks | 2 wins | **6 wins** | `auto` ran 3–12 of 40 min; roberta-base missing | `auto --background` + `wait`; roberta-base; bagging |
| 9 | 0.5.2 | 12 tasks | 5 wins | **7 wins** | a step overran its cap; ties broken by run order; `auto` stopped at 7 of 18 min | guarded steps; log-loss tie-break; budget-filling replicas (0.5.4) |

\* rounds 1–2 accidentally ran against the live dev checkout (harness PATH bug, fixed in round 3).

## Round 9 in full (0.5.2)

| task | metric | prompt only | trainwright |
|---|---|---|---|
| bike_sharing | RMSE ↓ | 40.08, 40.36 | **35.99** |
| app_reviews | RMSE ↓ | 1.077, 1.085 | **0.973** |
| eye_movements | accuracy | 0.908, 0.696 | **0.870** |
| eurosat | accuracy | 0.976, 0.954 | **0.986** |
| telco_churn | accuracy | 0.773, 0.789 | **0.791** |
| mercedes | RMSE ↓ | 8.320, 8.315 | **8.247** |
| miniboone | accuracy | 0.946, 0.946 | **0.953** |
| beans | accuracy | **0.954**, 0.949 | 0.949 |
| tweet_irony | accuracy | **0.761** (a 2nd run scored 0.855 using a checkpoint trained on the same public data: excluded) | 0.749 |
| car_parts | accuracy | **0.996** | 0.981 |
| superconduct | RMSE ↓ | **9.60, 10.16** | 10.04 |
| banking77 | accuracy | **0.931, 0.937** | 0.893 |

Per task, vs the prompt-only mean: 7 of 12, **+2.1%** (median +0.8%, best +10.5%, worst −4.3%).
Per head-to-head run (22 pairs): 14 wins, 1 tie, 7 losses, **+2.6%** (median +0.8%, best +25.0%,
worst −4.7%). Wins average +4.9%, losses −1.9%. Error cut on the wins: EuroSAT 60%, eye movements
34%, MiniBooNE 13%, bike sharing 10.5%, app reviews 10%.

The four losses:

| task | log | fixed in |
|---|---|---|
| banking77 | `roberta-base` epoch took 417 s under load, the step hung, the agent killed `auto` and shipped TF-IDF | 0.5.3: steps run in a subprocess with a hard deadline; roberta only if 3 epochs fit |
| car_parts | probe, probe-base and ensemble all scored val 1.000; run order picked the smallest | 0.5.3: exact ties broken by val log-loss |
| superconduct | `auto` finished at 7.0 of 18 min (one seed replica) | 0.5.4: replicas added while the remaining budget covers finalize |
| tweet_irony | roberta-base ran and was finalized (val AUC 0.861); 1.2 points on 690 tweets is within noise (SE ≈ 1.6) | not a bug |

## Without an agent

**OpenML-CC18 + CTR23, pipeline vs early-stopped HistGBM, sealed test** (`bench/suite.py`, v0.4.2):

| scored | better | exact ties | mean | median | median classification-error cut |
|---|---|---|---|---|---|
| 98 | 63 | 25 | +4.9% | +0.6% | 4.8% |

**Recipe variants, 76 suite datasets, val only** (`bench/variants.py`):

| tabular MLP recipe | beats HistGBM | mean vs HistGBM |
|---|---|---|
| 0.4 default | 49 / 76 | −3.5% |
| + `train.min_steps=2000` | 51 / 76 | +0.9% |
| + periodic embeddings (0.5 default) | **58 / 76** | **+4.7%** |
| + TabM-mini (`k_ensemble=16`), 35 datasets | +0.6% vs default | (kept as a tune option) |

**5-fold bagging, 16 small suite datasets, sealed test** (`bench/bagtest.py`):

| comparison | mean | median | record |
|---|---|---|---|
| bag vs the single run it copies | +4.9% | +2.9% | 13 wins, 1 tie, 2 losses |
| bag vs HistGBM | +10.0% | | 12 / 16 |
| single run vs HistGBM | +5.2% | | 8 / 16 |

A bag's out-of-fold val score is a single-model estimate (blood-transfusion: bag val 0.715 vs
0.749 for the single run; test 0.748 vs 0.716), so a bag supersedes the run it copies.
