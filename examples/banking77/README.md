# Example: replace an LLM intent router with a fine-tuned small encoder

```bash
uv run python examples/banking77/prepare.py data/banking77.csv
cd data && trainwright init --data banking77.csv --target intent
trainwright profile && trainwright split && trainwright baseline       # TF-IDF + logistic ≈ 0.87 macro-F1
trainwright train --hypothesis "fine-tuned MiniLM (default text model)"  # ≈ 0.92 macro-F1
trainwright evaluate best        # per-intent errors + "confidence >= 0.9 covers X% of traffic at Y% accuracy"
```

## If you have no labels, only an LLM

```bash
trainwright label --input unlabelled.csv --column message --labels "$(cat intents.txt)" \
    --model claude-sonnet-5-5 --limit 50 --estimate      # price a pilot first
trainwright label --input unlabelled.csv --column message --labels "$(cat intents.txt)" \
    --model claude-sonnet-5-5 --max-cost-usd 10 --out labelled.csv
trainwright init --data labelled.csv --target label --text-column message
# ... split / baseline / train as above, then find label errors:
trainwright review best --adjudicate claude-opus-5-5 --max-cost-usd 2 --apply-agreed
# set `data.corrections: corrections.csv` in trainwright.yaml and retrain
```

Metrics on LLM-produced labels measure agreement with the LLM. Hand-check a sample before you
call the small model "as good as" the LLM.
