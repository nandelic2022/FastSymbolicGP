# Migrating FastSymbolicGP V1.0.x experiments to V1.1.0

## What stays familiar

The primary estimators remain:

- `FastSymbolicRegressor`
- `FastSymbolicClassifier`
- `FastSymbolicTransformer`

Common V1.0.x experiment parameters such as `population_size`, `generations`, `max_depth`, `max_nodes`, `validation_fraction`, `selection_tolerance`, `exact_champion_candidates`, `evaluation_backend`, `fitness_engine`, `preset`, `random_state`, and `n_jobs` remain accepted.

## Regression scaling

For regression, use:

```python
fitness_scaling="linear"
```

or:

```python
fitness_scaling="none"
```

Classification-only logistic scalers are rejected by `FastSymbolicRegressor` before evolution begins.

## Legacy fused fitness request

V1.1.0 adds arbitrary scientific/custom metrics. Those metrics are evaluated through the materialized objective layer. Therefore:

```python
fitness_engine="fused"
```

is retained as a compatibility request, but resolves transparently to:

```text
resolved_fitness_engine = materialized
```

The reason is stored in `model.execution_plan_["fitness_engine_fallback_reason"]`.

Expression execution itself can still use the optional Numba postfix backend:

```python
evaluation_backend="numba"
```

or simply:

```python
evaluation_backend="auto"
```

## sklearn search precedence

Parameters applied with `set_params()` by `GridSearchCV`/`RandomizedSearchCV` are marked as explicit search parameters and are not overwritten by presets under the default `preset_override_policy="defaults_only"`.

## Scientific search additions

New major options include:

```python
selection_method="lexicase"
selection_method="epsilon_lexicase"
constraints={...}
feature_units={...}
target_unit="W"
unit_policy="penalty"  # or "enforce"
uncertainty="split_conformal"
```

## Recommended CCPP configuration

```python
FastSymbolicRegressor(
    preset="balanced",
    selection_method="epsilon_lexicase",
    fitness_metric="rmse",
    fitness_scaling="linear",
    evaluation_backend="auto",
    fitness_engine="auto",
    uncertainty="split_conformal",
    random_state=42,
)
```
