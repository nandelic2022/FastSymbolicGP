# FastSymbolicGP V1.2.8 — Batched Tournament Scheduler & Generation Planner

V1.2.8 removes the largest remaining standard-search orchestration hotspot: repeated Python calls to `Generator.choice(..., replace=False)` for every parent selection.

## Highlights
- Vectorized exact-distribution best-of-k tournament scheduler.
- Dedicated deterministic selection RNG, isolated from crossover/mutation RNG.
- Selection RNG state persisted in checkpoints for reproducible resume.
- Rootless Numba-first genome engine from V1.2.7 retained.
- Fused single-pass RMSE/MSE/R2 regression retained.
- Scientific/typed/constraint workflows retain their safe fallback paths.
- Profiler now reports `tournament_sampling=batched_exact_distribution` and `tournament_scheduler=vectorized_cdf_v1`.

## Fixed benchmark gate
Diabetes, split random_state=42, population=500, generations=100, add/sub/mul/div, max_depth=8, max_nodes=60.

Source-tree validation on this runtime:
- runtime about 2.47 s after JIT warm-up
- test R2 = 0.4793856149
- test RMSE = 52.5194632
- LinearRegression baseline R2 = 0.4526027630

The tournament scheduler preserves the exact marginal winner distribution of classic best-of-k sampling without replacement. It does not preserve the exact V1.2.7 random trajectory because selection uses a dedicated RNG stream; this is intentional and deterministic for a fixed `random_state`.

No claim is made here about gplearn performance because gplearn is not installed in the release-build runtime. `examples/benchmark_vs_gplearn.py` is included for direct comparison on systems where gplearn is available.
