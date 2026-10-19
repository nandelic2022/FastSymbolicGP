# FastSymbolicGP V1.2.7 — Numba-First Rootless Evolution Core

V1.2.7 moves the standard scalar+tournament fast path from persistent Python trees to a rootless postfix-genome representation during evolution. The Python tree is materialized only when a cold-path feature needs it, such as final simplification, expression export, scientific constraints, or model inspection.

## Highlights

- Rootless postfix individuals during standard fast evolution.
- Numba topology analysis for subtree starts, depths, and preorder node mapping.
- Genome-only crossover, subtree mutation, and point mutation.
- Random subtree generation directly into postfix arrays while preserving the same RNG call sequence as the tree generator.
- Function/operator sampling hot path changed from `Generator.choice(tuple)` to equivalent integer indexing with the same RNG state/results.
- Exact quality-preserving tournament behavior remains unchanged.
- Automatic safe fallback to the tree engine for constraints, unit-aware/typed GP, Semantic GP, memetic search, and evolution-time canonical simplification.
- Final champion is materialized back to the public tree representation, so expression export and the existing API remain intact.
- Added `examples/benchmark_vs_gplearn.py` for an apples-to-apples local comparison when gplearn is installed.

## Quality-preserving speed gate

Fixed Diabetes split, population 500, generations 100, add/sub/mul/div, seed 42:

- V1.2.6 median-like local runs: roughly 5.0–5.4 s.
- V1.2.7 local runs after JIT warm-up: roughly 3.5–3.7 s.
- Test R² remains exactly `0.4623504885509383` in the release gate.
- Test RMSE remains `53.371796851642`.

These timings are environment-specific and are not universal performance claims.

## gplearn note

The release includes a direct comparison script, but gplearn was not available in the isolated build environment and internet access was unavailable, so V1.2.7 does **not** claim a verified universal accuracy or speed advantage over gplearn. Run the included benchmark on the target machine for that comparison.
