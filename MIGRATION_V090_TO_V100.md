# Migration from V0.9.0 to V1.0.0

Most V0.9.0 code runs unchanged.

## Preset semantics

V0.9.0 named presets could replace explicit parameter values. V1.0.0 applies named presets only to parameters still at their constructor defaults. Non-default explicit values are preserved.

Use `preset_override_policy="always"` for the legacy preset-wins behavior.

## Exact champion

`exact_champion=True` is the V1 default. Elite-logistic screening may remain approximate for most candidates, but champion candidates are exactly refined every generation and early stopping uses the exact global archive.

## Auto execution planning

`preset="auto"` now adapts island topology to workload size. Named presets do not have their topology silently rewritten by the planner.

## New stable helpers

- `warmup()`
- `model.summary()`
- `model.profile_`
- `model.execution_plan_`
- `model.resolved_config_`
- `model.model_manifest()`
- `model.save()` / `Class.load()`

## Experimental namespace

`FastSymbolicNetworkClassifier` remains importable at top level for compatibility and is also available from `fastsymbolicgp.experimental`.
