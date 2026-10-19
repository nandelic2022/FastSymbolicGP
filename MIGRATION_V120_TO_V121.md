# Migration: FastSymbolicGP V1.2.0 → V1.2.1

V1.2.1 is source-compatible with normal V1.2.0 estimator usage, with one intentional default change for performance.

## Default optimization change

V1.2.0 defaulted to:

```python
optimization="pareto"
```

V1.2.1 defaults to:

```python
optimization="scalar"
```

This prevents ordinary/basic runs from paying NSGA-II and Pareto-archive costs unless multiobjective search was actually requested.

To reproduce explicit V1.2.0 Pareto search behavior, set:

```python
model = FastSymbolicRegressor(
    optimization="pareto",
    ...
)
```

The publication preset retains Pareto automatically:

```python
model = FastSymbolicRegressor(preset="publication")
```

## Profiling

V1.2.1 adds:

```python
print(model.profile_["stage_seconds"])
print(model.profile_["nsga2_calls"])
print(model.profile_["pareto_archive_updates"])
print(model.profile_["state_snapshot_count"])
print(model.profile_["tracked_stage_seconds"])
print(model.profile_["untracked_overhead_seconds"])
```
