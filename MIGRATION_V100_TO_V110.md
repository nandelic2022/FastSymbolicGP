# Migration from FastSymbolicGP 1.0.x to 1.1.0

Most basic estimator usage remains unchanged:

```python
from fastsymbolicgp import FastSymbolicRegressor, FastSymbolicClassifier
```

## Important changes

1. Presets are task-safe. Regression is never assigned classifier-only logistic scaling.
2. `selection_method` can be `tournament`, `lexicase`, or `epsilon_lexicase`.
3. `constraints`, `feature_units`, `target_unit`, and `unit_policy` add scientific search restrictions.
4. Regression can calibrate split-conformal intervals through `uncertainty="split_conformal"`.
5. Missing-value handling supports `error`, `median`, `constant`, and protected `native` modes.
6. Safe JSON is recommended for portable model exchange. Joblib checkpoints remain available for trusted local recovery only.
7. `resolved_config_` reports the selected execution backend and scientific configuration.

## Common 1.0.x parameters retained

`population_size`, `generations`, `tournament_size`, `max_depth`, `max_nodes`, `const_probability`, `parsimony_coefficient`, `patience`, `selection_tolerance`, `exact_champion`, `exact_champion_candidates`, `validation_fraction`, `preset`, `fitness_engine`, `evaluation_backend`, `n_islands`, `migration_interval`, `migration_size`, `low_memory`, `random_state`, and `verbose` remain accepted.

## Performance migration note

V1.1 includes an optional Numba postfix evaluator. Before replacing a V1.0.x production benchmark environment, rerun the same dataset, seed, compute budget, and metric protocol because the V1.1 scientific search layer changes the search dynamics and does not claim numerical identity with historical runs.
