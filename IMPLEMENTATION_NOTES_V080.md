# FastSymbolicGP V0.8.0 Implementation Notes

## Implemented in this release

- packed population postfix representation;
- one-call Numba population evaluator using `prange`;
- automatic Numba thread configuration for `n_jobs`;
- memory-aware population batching;
- fused training and validation expression execution;
- cached fused reference matrices;
- cached dataset tokens for DAG evaluation;
- population-aware classification scoring;
- one-step approximate logistic scaling;
- globally selected exact elite refinement across memory batches;
- exact final scaling of the selected classifier;
- progressive sample fractions over generations;
- corrected all-core island scheduling;
- compatibility, fast, turbo and large-dataset presets;
- updated diagnostics, documentation, examples, tests, wheel and validation reports.

## Deliberately deferred

The release does not yet replace symbolic trees with a fully flat evolutionary genome, fuse every fitness metric into one Numba kernel, or add GPU execution. Those are larger architectural changes suitable for a later major performance release. V0.8.0 concentrates on high-return improvements that preserve the public API and pass the retained functional tests.
