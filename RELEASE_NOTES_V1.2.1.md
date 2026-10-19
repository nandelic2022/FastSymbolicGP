# FastSymbolicGP V1.2.1 — Performance Regression Hotfix

V1.2.1 fixes the Python-side evolution regressions introduced as the V1.2 search stack grew. The scientific, semantic, typed-unit, memetic, checkpoint and publication surfaces from V1.2.0 are retained.

## Main changes

- Default `optimization` is now `scalar` for normal estimators. `preset="publication"` explicitly retains Pareto/NSGA-II search.
- NSGA-II is assigned at most once per generation when Pareto search is enabled.
- Removed redundant Pareto ranking from fused evaluators, island-local preparation and memetic passes.
- Pareto objective vectors are precomputed once per ranking pass.
- Pareto archive maintenance is skipped during scalar generations and a reporting front is built once at fit completion.
- Pareto archive maintenance uses incremental first-front updates.
- Hall-of-fame maintenance is bounded to previous top-K plus current top-K rather than cloning/sorting the full population every generation.
- Full evolution-state snapshots are created only for requested checkpoint generations and the terminal generation.
- `Program` now caches immutable structural properties (`paths`, `size`, `depth`, DAG size and structural hash) and invalidates those caches on structural/constant changes.
- Evolutionary clone operations no longer deep-copy compiled postfix execution arrays.
- New profiler stages: `nsga2`, `pareto_archive`, `state_snapshot`.
- New profile counters: `nsga2_calls`, `pareto_archive_updates`, `state_snapshot_count`, `variation_attempts`, `variation_rejections`, `variation_fallbacks`.
- Profile now reports `tracked_stage_seconds` and `untracked_overhead_seconds`.

## Performance measurements in the release runtime

These are workload-specific smoke measurements, not universal speedup guarantees.

- 200 individuals × 2 generations, fused Numba, diabetes dataset:
  - V1.2.0 default: ~1.96 s
  - V1.2.1 default: ~0.13 s in the initial hotfix benchmark
- 500 individuals × 2 generations:
  - V1.2.0 default: ~13.10 s
  - V1.2.1 default: ~0.46 s before the later structural-cache optimizations
- 500 individuals × 100 generations using `add/sub/mul/div/sqrt/log/abs`, max depth 10 and max nodes 100:
  - V1.2.1: ~10.9 s, all 100 generations completed, 37,666 actual program evaluations.

The 500×100 V1.2.0 attempt exceeded the 120-second execution limit in the same development environment.

## Compatibility

V1.2.1 keeps explicit Pareto search available through `optimization="pareto"`. Pareto reporting remains available after scalar fits via `get_pareto_front()`; the reporting front is produced once at the end instead of continuously maintained.
