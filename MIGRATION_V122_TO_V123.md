# Migration: FastSymbolicGP V1.2.2 → V1.2.3

V1.2.3 is source-compatible with normal V1.2.2 regressor/classifier workflows. No constructor changes are required.

## Performance behavior

For fused regression with MSE, RMSE or R2 and linear/identity scaling, training fitness now uses an exact one-pass sufficient-statistics calculation. MAE and Huber retain the exact residual-wise second pass.

The standard scalar/tournament path keeps tournament sampling without replacement but removes unnecessary candidate-list allocation and repeated winner comparisons.

## Diagnostics

After fitting, inspect:

```python
print(model.execution_plan_.get("fused_training_passes"))
print(model.execution_plan_.get("fused_single_pass_squared_error"))
print(model.profile_.get("variation_engine"))
print(model.profile_.get("tournament_sampling"))
```

For the normal fast RMSE path the expected values are `1`, `True`, `fast_shared_tree_exact_tournament_v2`, and `without_replacement_exact_fast`.
