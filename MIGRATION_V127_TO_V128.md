# Migration: V1.2.7 to V1.2.8

No API changes are required for normal estimators.

The main behavioral difference is tournament scheduling. Standard sorted-population tournament selection now uses a vectorized exact-distribution scheduler with a dedicated RNG stream. Runs remain deterministic for a fixed `random_state`, but an identical seed will not necessarily produce the identical expression obtained with V1.2.7.

Profiler changes:
- `tournament_sampling`: `batched_exact_distribution`
- `tournament_scheduler`: `vectorized_cdf_v1`

Checkpoint schema remains backward-compatible; V1.2.8 additionally stores `selection_rng_state` when available.
