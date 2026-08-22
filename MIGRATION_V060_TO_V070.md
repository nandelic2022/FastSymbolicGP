# Migration from V0.6.0 to V0.7.0

Most V0.6.0 scripts require no changes.

## Reporting changes

Use:

```python
model.evaluation_backend_   # configured postfix backend
model.execution_engine_     # actual execution path
```

Multiclass statistics now use explicit fields such as:

```text
generations_total
generations_mean_per_class
generations_max_per_class
nodes_total
nodes_mean_per_model
depth_max_per_model
```

## New island mode

```python
evolution_model="islands"
n_islands=4
migration_interval=10
```

## New shared multiclass mode

```python
multiclass_strategy="shared_softmax"
shared_n_components=8
multiclass_calibration="temperature"
```

## DAG/postfix selection

The default `dag_execution="auto"` activates cached DAG execution when subtree caching is enabled. To force the postfix NumPy/Numba evaluator:

```python
dag_execution="off"
evaluation_backend="numba"
```

## Export boundary

`export_python` supports all fitted estimator modes. C/C++/Java/Kotlin/JavaScript export in V0.7.0 is limited to binary classifiers and regressors.
