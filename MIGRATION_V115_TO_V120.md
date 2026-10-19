# Migration from FastSymbolicGP V1.1.5 to V1.2.0

V1.2.0 is backward compatible with the certified V1.1.5 estimator surface. Existing configurations continue to use the legacy defaults because semantic diversity, age management and memetic search are disabled by default.

## New opt-in search controls

```python
FastSymbolicRegressor(
    semantic_diversity=True,
    selection_method="semantic_tournament",
    semantic_archive_size=64,
    semantic_mutation_rate=0.10,
    age_management="layered",
    age_gap=10,
    age_layers=4,
    memetic_search="elites",
    memetic_elites=2,
    memetic_interval=5,
    memetic_budget=120,
    simplification="canonical",
    model_selection="pareto_knee",
)
```

`pareto_objectives` additionally accepts `semantic_redundancy` and `age`.

Checkpoint schema 4 remains load-compatible with V1.1.x state fields; new V1.2.0 checkpoints additionally persist the semantic archive and memetic objective-evaluation count.
