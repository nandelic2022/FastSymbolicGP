# FastSymbolicGP V1.2.3 — Ultra-Fast Quality-Preserving Core

V1.2.3 is a performance/quality release focused on the ordinary symbolic-regression path. It does not remove the scientific, typed, semantic, Pareto, checkpointing or publication features from the V1.2 line.

## Highlights

- Exact single-pass fused MSE/RMSE/R2 training loss after affine scaling.
- Faster NumPy-based postfix population packing.
- Cheaper Numba compile-cache lookup.
- Quality-preserving true without-replacement fast tournament selection.
- Explicit diagnostics for fused training passes and tournament implementation.
- Dedicated 500×100 Diabetes speed/quality benchmark.

## Certified benchmark

On the release runtime after Numba warm-up, repeated source runs were approximately 11–13 seconds for 500 individuals × 100 generations. The release candidate produced test R2 0.46235 and RMSE 53.37 on the fixed Diabetes split, while scikit-learn LinearRegression produced R2 0.45260. Performance is hardware/runtime dependent.

## Compatibility

V1.2.3 retains the V1.2.2 API. MAE and Huber remain exact but still require a residual-wise second training pass. GPU execution is not introduced in this release.
