# FastSymbolicGP V1.0.0 Release Notes

FastSymbolicGP V1.0.0 is the first stable major release. It is built directly on V0.9.0 and preserves its high-performance compiled evaluator while fixing configuration semantics and exact-champion consistency.

## Acceptance criteria achieved in this build

- 29 automated tests pass.
- Python source compilation passes.
- forced 120-generation Breast Cancer benchmark is below 5 seconds on the build environment after warm-up.
- V1 benchmark retained or improved predictive quality versus V0.9.0 in the recorded deterministic run.
- explicit non-default overrides survive named presets.
- `preset=auto` selects a single-island panmictic topology for small datasets.
- exact champion fitness is finite and monotonically non-increasing across the global archive.
- classifier/regressor persistence and existing retained V0.9.0 capabilities continue to pass regression tests.

## Compatibility note

The top-level `FastSymbolicNetworkClassifier` import remains available for compatibility, but symbolic-network functionality is not part of the V1 stable API guarantee. New code may import it from `fastsymbolicgp.experimental`.
