# Example: replace a vision-LLM call with a small image classifier

```bash
uv run python examples/car_parts/download.py data/car_parts
cd data && trainwright init --data car_parts --name car-parts
trainwright profile        # flags burst photos: 68% of images have a near-duplicate
trainwright split          # duplicate-aware: bursts never straddle train/val/test
trainwright baseline
trainwright train --hypothesis "pretrained resnet18 at 224px"
trainwright diagnose latest
trainwright evaluate best --split test --reason "final model"
trainwright finalize best
trainwright export <final-run> && trainwright benchmark <final-run>
```

Or open the folder in Claude Code after `trainwright agent-setup` and ask it to replace your LLM
call with a cheaper model.
