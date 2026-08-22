# Migration from FastSymbolicGP V0.7.0 to V0.8.0

## Existing scripts

Most V0.7.0 scripts run without modification. The main differences are performance-oriented defaults.

## Reproduce conservative V0.7.0 behaviour

```python
model = FastSymbolicClassifier(
    preset="compatibility",
    semantic_duplicate_elimination=True,
    subtree_cache=True,
    fitness_scaling="exact_logistic",
    n_jobs=1,
)
```

Use this when comparing a V0.8.0 run directly with historical V0.7.0 results.

## Recommended V0.8.0 configuration

```python
model = FastSymbolicClassifier(
    evaluation_backend="population_numba",
    dag_execution="off",
    fitness_scaling="elite_logistic",
    fitness_scaling_top_k=12,
    fitness_scaling_interval=2,
    semantic_duplicate_elimination=False,
    n_jobs=-1,
)
```

## Large factorial experiments

For cross-validation grids, avoid nested all-core parallelism. Use one of the following:

- parallelise inside FastSymbolicGP with outer folds sequential;
- parallelise folds/combinations externally and set each estimator to `n_jobs=1`.

Do not use all cores at both levels.

## Progressive sampling

V0.8.0 accepts:

```python
max_samples=(0.20, 1.00)
```

A one-value float and integer sample count keep their previous meanings.

## Scaling names

- `elite_logistic`: approximate all programs and exactly refine elites;
- `exact_logistic` or `logistic`: exact iterative scaling for every program;
- `fixed_sigmoid`: no scaling during evolution;
- `none` or `identity`: identity scaling.

The final selected binary classifier is exactly scaled after evolution in every case.
