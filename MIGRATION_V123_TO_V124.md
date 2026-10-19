# Migration: FastSymbolicGP V1.2.3 → V1.2.4

Normal regressor/classifier constructor usage remains source compatible.

Important behavioral change: `FastSymbolicRegressor.get_expression()` now returns the complete fitted equation including the affine scale/intercept used by `predict()`. Use `get_raw_expression()` or `get_expression(include_scaling=False)` for the raw GP tree.

New parameters:
- `final_simplification="certified"`
- `simplification_tolerance=1e-10`
- `simplification_reference_points=2048`

For publication workflows, `preset="publication"` resolves final simplification to the globally safe mode.
