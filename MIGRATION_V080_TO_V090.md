# Migration from FastSymbolicGP V0.8.0 to V0.9.0

Existing named constructor arguments remain valid. Most V0.8.0 scripts can be run without modification.

## New default behaviour

When Numba is available, `fitness_engine="auto"` uses fused scoring for supported metrics. `evaluation_cache=True`, which already existed in the API, now skips exact repeated fitness work.

Small floating-point reduction differences can change tie-breaking in stochastic evolution. For a conservative comparison with an earlier experiment, use:

```python
model = FastSymbolicClassifier(
    fitness_engine="materialized",
    evaluation_cache=False,
    # retain the remaining V0.8.0 parameters and seed
)
```

or:

```python
model = FastSymbolicClassifier(preset="compatibility", random_state=42)
```

## Unsupported fused metrics

With `fitness_engine="auto"`, ROC AUC, average precision, and regression MAE automatically use materialized population scoring. With `fitness_engine="fused"`, unsupported combinations raise a clear `ValueError`.

## New history fields

Code reading `history_` may optionally consume:

```text
fitness_engine
fitness_cache_hits
fitness_cache_misses
intra_population_reuse
actual_program_evaluations
fitness_cache_hit_rate
```

Existing history fields are retained.

## Evaluation budgets

In panmictic evolution, `evaluation_budget` now refers to actual novel program evaluations. Cached elites no longer consume the budget. This is more faithful to computational work but can allow additional generations compared with V0.8.0 under the same nominal budget.
