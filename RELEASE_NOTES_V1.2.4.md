# FastSymbolicGP V1.2.4 — Expression Quality & Simplification

V1.2.4 preserves the V1.2.3 execution/search core and targets the readability and correctness of exported symbolic equations.

## Highlights
- Protected-semantics-aware globally safe rewrites (`x/0 -> 1`, `x/x -> 1`, neutral-element removal, constant folding, double-negation/idempotence cleanup).
- Empirically certified final pruning (`final_simplification="certified"`) with strict prediction-equivalence checks on the model-development reference set.
- `preset="publication"` resolves final simplification to `safe`, avoiding any claim of global algebraic equivalence from an empirical certificate.
- `get_raw_expression()` returns the GP tree; regressor `get_expression()` now returns the actual fitted equation including affine scale/intercept.
- `get_simplification_report()` reports node reduction, certification scope, tolerance, trials, accepted rewrites, and maximum prediction delta.
- `verbose=2` now emits every generation using a compact progress line; `verbose=1` remains periodic.

## Benchmark quality gate
On the fixed scikit-learn Diabetes split used throughout V1.2 development (population 500, 100 generations, seed 42, add/sub/mul/div, Numba + fused):
- test R²: 0.46235049
- test RMSE: 53.3718
- nodes before final certified simplification: 59
- nodes after final certified simplification: 51
- held-out test score unchanged in the measured gate.

Certified simplification is an empirical equivalence claim on its reference set, not a theorem of global equivalence outside that set. Use `final_simplification="safe"` when only globally protected-semantics-safe rewrites are desired.
