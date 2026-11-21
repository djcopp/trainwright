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

<!-- ROUND2 -->
