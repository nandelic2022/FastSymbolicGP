# Migration: FastSymbolicGP V1.2.1 → V1.2.2

V1.2.2 is intended as a source-compatible performance hotfix. No changes are required for the normal `FastSymbolicRegressor` or `FastSymbolicClassifier` API.

The main intentional internal change is that the fast tournament path uses bulk tournament draws with replacement. Tournament selection with replacement is standard GP practice and removes a large number of small NumPy allocations. Scientific typed/unit/hard-constraint paths remain on their stricter generic variation implementation.

For the fastest ordinary regression path use `evaluation_backend="numba"`, `fitness_engine="fused"`, `optimization="scalar"`, `semantic_diversity=False`, `age_management="off"`, and `memetic_search="off"`.
