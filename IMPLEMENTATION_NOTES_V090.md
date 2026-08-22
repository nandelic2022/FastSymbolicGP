# FastSymbolicGP V0.9.0 Implementation Notes

## Fused classifier kernel

The classifier kernel parallelises over packed programs. Each worker allocates one reusable postfix stack and one thread-local raw-training vector for the active program. The first pass accumulates weighted sufficient statistics for the one-step logistic scaler. The second pass over the thread-local vector computes stable logistic loss or weighted confusion-matrix counts. Validation rows are evaluated after scaling without retaining their outputs.

Supported fused metrics are log loss, accuracy, balanced accuracy, F1, and MCC. Metrics requiring output ordering, such as ROC AUC and average precision, fall back to materialized scoring in `auto` mode.

## Fused regressor kernel

For every program, the kernel accumulates weighted sums of the raw output, target, squared terms, and cross-products. Linear scaling is calculated analytically. Training SSE is obtained from the sufficient statistics, while validation SSE is accumulated in one pass with the fixed training scaler. MSE, RMSE, and R2 are then calculated without population prediction arrays.

MAE requires residual magnitudes and therefore uses the materialized path.

## Exact fitness cache

The cache stores only scalar snapshots:

- training fitness;
- validation fitness;
- scale;
- intercept;
- scaling label.

Keys combine the exact compiled program digest, data and target fingerprints, sample-weight fingerprints, and scoring configuration. Elite-logistic snapshots are cached before generation-specific exact top-k refinement so each generation retains the same provisional ranking semantics as V0.8.0.

## Duplicate reuse

Within one population call, programs sharing the exact execution hash are grouped. One representative is evaluated and its pre-refinement scalar snapshot is copied to aliases. This remains correct even when human-readable structural hashes round constants because execution hashes use raw argument bytes.

## Evaluation accounting

`evaluation_count_` counts novel program evaluations in panmictic evolution. Generation history also exposes the population size and reuse counts, allowing benchmark scripts to distinguish search population size from actual evaluator work.

## Deliberately deferred

V0.9.0 does not add GPU execution, distributed evolution, a flat genome variation engine, or approximate semantic pruning. These changes would alter architecture or search behaviour more substantially and remain candidates for V1.0.0.
