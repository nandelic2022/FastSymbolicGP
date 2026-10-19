# FastSymbolicGP V1.2.6 — Flat Genome / Ultra-Fast Variation Engine

V1.2.6 is a performance release focused on removing tree-to-postfix recompilation from the standard fast GP path without changing the evolutionary decisions or benchmark quality.

## Main changes

- Exact flat-genome inheritance for crossover, subtree mutation and point mutation.
- Tree and flat genome are certified equivalent in automated tests.
- Preallocated population packing for the fused Numba evaluator.
- Genome-backed structural hashing when a compiled genome is available.
- Cached topology size/depth on persistent shared nodes.
- Typed/unit-aware/scientific paths remain on the conservative generic engine.

## Fixed quality gate

Diabetes benchmark, random_state=42, 500 population, 100 generations, add/sub/mul/div, max_depth=8, max_nodes=60, Numba + fused + scalar:

- Runtime in the release build environment after JIT warm-up: approximately 4.95 s.
- Test R2: 0.46235049.
- Test RMSE: 53.3718.
- Actual evaluations: 41,731.
- Best simplified model: 51 nodes.
- Variation engine: `flat_genome_inheritance_v1`.

Timings are machine-dependent and are not a universal speed claim. The quality gate is used to prevent speed improvements that degrade the fixed benchmark result.
