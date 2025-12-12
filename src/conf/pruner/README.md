# Pruners

## Hyperband Pruner
- Fast and stable
- Balanced trade-off between quality and runtime
- Recommended as default

## Successive Halving Pruner
- Very fast
- Aggressively prunes trials
- Suitable for large search spaces

## Median Pruner
- Safe baseline
- Least aggressive
- Slower than other pruners

## Patient Pruner
- Adaptive pruning
- Waits for trials to improve before pruning
- Useful for noisy or slow-converging metrics

## Nop Pruner
- Does not prune any trials
- Useful for baseline comparisons
- Safe and deterministic
