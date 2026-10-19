# Changelog

## 1.2.8 — Batched Tournament Scheduler & Generation Planner
- Added vectorized exact-distribution tournament winner scheduling.
- Added dedicated deterministic selection RNG with checkpoint persistence.
- Removed more than 100k per-fit Python tournament sampling calls on the 500x100 gate.
- Retained V1.2.7 rootless Numba-first genome variation and V1.2.4 expression-quality safeguards.
- Added V1.2.8 determinism/distribution/checkpoint acceptance tests.

## 1.2.7 — Numba-First Rootless Evolution Core

- Added rootless postfix-genome evolution for the standard fast path.
- Added Numba topology analysis and genome-only variation operators.
- Preserved exact benchmark search quality for the fixed 500x100 Diabetes gate.
- Added automatic safe fallback for scientific/typed/constraint workflows.
- Added gplearn comparison example.

## 1.2.6 — Flat Genome / Ultra-Fast Variation Engine

- Added quality-preserving flat-genome inheritance for crossover, subtree mutation and point mutation.
- Offspring inherit exact postfix genomes by splice/update, avoiding full tree recompilation before fused evaluation.
- Added cached Node size/depth topology metadata for persistent shared trees.
- Population packing now uses one preallocated opcode/argument buffer instead of repeated concatenate passes.
- Structural hashes use compiled genomes when available, with the string representation retained as a fallback.
- Preserved the exact tournament RNG path and model quality of V1.2.5.
- Added genome-vs-tree equivalence tests for every fast variation operator.
- Runtime console now reports `flat_genome_inheritance_v1` when the fast path is active.

## 1.2.6
- Added dependency-free startup dashboard and final training summary.
- `verbose=2` now prints every generation.
- Added live throughput, ETA, BEST marker, bottleneck, cache and equation reporting.
- Preserved V1.2.4 search semantics.

## 1.2.4
- Added protected-semantics-aware safe simplification.
- Added empirically certified final pruning with a detailed simplification report.
- Added `get_raw_expression()` and corrected regressor `get_expression()` to include affine scaling.
- `preset="publication"` uses globally safe simplification; normal runs can use certified cleanup.
- `verbose=2` now reports every generation with compact progress output.
- Added V1.2.4 expression-quality acceptance tests.

## 1.2.3

- Added exact single-pass fused training loss for MSE/RMSE/R2 using first-pass sufficient statistics.
- Preserved exact affine scaling and numerical equivalence with the materialized reference path.
- Reworked population genome packing to avoid per-node Python list expansion.
- Replaced structural-hash compile-cache checks with immutable-root identity checks.
- Added quality-preserving fast tournament selection with true without-replacement sampling.
- Added `fused_training_passes` and `fused_single_pass_squared_error` execution diagnostics.
- Added the exact 500×100 Diabetes benchmark script and V1.2.3 speed/quality acceptance tests.
- Preserved V1.0–V1.2.2 public/scientific/search features.

## 1.2.2

- Added the fast variation engine for standard scalar/tournament GP runs.
- Removed per-parent population list copies and vectorized tournament parent schedules.
- Added persistent-tree structural sharing for crossover/mutation hot paths.
- Program paths are cached as immutable tuples; size/depth are computed together in one traversal.
- Added fast basic admissibility checks when typed/unit/hard scientific constraints are inactive.
- Evolution snapshots use cheap immutable program clones instead of deep-copying every tree.
- Removed the internal sklearn feature-name warning after DataFrame fitting.
- Added V1.2.2 variation-performance acceptance tests.
- Preserved V1.0–V1.2.1 API and scientific/search features.

## 1.2.1
- Fixed V1.2.0 Python-side performance regression in default evolution.
- Default optimization changed from Pareto to scalar; `preset="publication"` retains Pareto search.
- Removed redundant NSGA-II ranking from evaluator, memetic, island and global paths; Pareto ranks are now assigned once per generation.
- Precomputed Pareto objective vectors for dominance sorting.
- Added incremental non-dominated archive maintenance.
- Scalar search no longer updates the Pareto archive each generation; a reporting front is built once at fit completion.
- Full evolution state is cloned only at requested checkpoints and terminal fit state.
- Added `nsga2`, `pareto_archive`, and `state_snapshot` profiler stages and counters.
- Added V1.2.1 performance-regression acceptance tests.

# Changelog

## 1.2.0
- Semantic GP 2.0 with novelty archive, semantic tournament and guided semantic mutation.
- Age-layered lineage management and age-aware selection.
- Canonical algebraic simplification and stronger structural duplicate normalization.
- In-evolution memetic constant optimization with bounded local search.
- Final model-selection policies: best validation, smallest within tolerance, Pareto knee and semantic consensus.
- Pareto semantic-redundancy and age objectives.
- Checkpoint schema 4 and new semantic/evolution diagnostics.

## 1.1.5 — Publication & Scientific Certification

- Added resumable publication benchmarking with incremental fold persistence.
- Added deterministic Nguyen 1–10 and Friedman-1 symbolic-regression datasets.
- Added paired common-block statistical analysis: Friedman, pairwise Wilcoxon, Holm correction, rank-biserial effect sizes, average ranks, and Nemenyi critical difference when SciPy is available.
- Added `stability_report()` and `robustness_report()`.
- Added machine-readable scientific certificates and `fastsymbolicgp-certify`.
- Upgraded reproducibility bundles to v2 with safe-model export, reference arrays, environment capture, scientific certificate, robustness summary, and SHA-256 content manifest.
- Preserved the V1.1.4 scientific-constraints/typed-GP and V1.1.3 performance core.
- Full source suite: 87 tests passing.

## 1.1.4 — Scientific Constraints & Typed GP Hardening

### Dimensional algebra and typed constants
- engineering-unit aliases/prefixes such as kW, MW, kWh, bar and degC resolve to their physical dimensions;
- strongly typed grammar builds a bounded closure of available dimensions, allowing deeper physically valid combinations;
- typed ephemeral constants can carry units through `constant_units={unit: (min,max)}` and survive mutation, JSON serialization and expression rendering;
- champion constant optimization respects per-unit constant bounds.

### Hard scientific constraints
- hard behavioral constraints are true feasibility conditions: violating models receive infinite search fitness even when `constraint_penalty` is not a Pareto objective;
- deterministic virtual constraint points support explicit points and Latin-hypercube/uniform sampling over user ranges;
- monotonicity, output bounds and boundary conditions report detailed violation diagnostics;
- optional physics residual callables support soft or hard conservation/equation residual constraints over real + virtual points;
- fitted models expose `get_constraint_report()` and `get_unit_report()`.

### Generalized Pareto search
- `pareto_objectives` is configurable instead of hard-coded;
- supported objectives include validation/training loss, tree/DAG size, depth, variables, nonlinear operators, weighted complexity/inference cost and constraint penalty;
- NSGA-II dominance is feasibility-aware and the persistent archive prefers feasible solutions whenever available;
- Pareto DataFrame export exposes all selected objective values.

### Validation
- V1.1.4 adds dedicated engineering-unit, typed-constant, virtual-point, physics-residual, hard-monotonicity, safe-JSON and generalized-Pareto tests;
- all historical V1.0–V1.1.3 acceptance suites remain part of release certification.

## 1.1.3 — Performance & Evolution Engine Hardening

### Throughput and caching
- fused regression/classification now structural-deduplicate each new population batch before entering Numba kernels;
- exact fitness is propagated to duplicate programs in the same batch and reused across compatible subsequent batches;
- bounded `evaluation_cache` / `fitness_cache_max_entries` runtime controls exposed on core estimators;
- compiled postfix genomes cached per unchanged program structure;
- cross-island population batching reduces repeated fused-kernel setup while preserving island-local selection and migration.

### Unit-aware evolution
- strongly typed dimensional tree generation activated automatically for hard unit enforcement;
- unit-preserving subtree crossover, subtree mutation and point mutation;
- maximum-depth terminal failures reject the local grammar production rather than recursing indefinitely;
- `typed_generation=auto|on|off` added for reproducible comparison with the V1.1.2 generate/validate path.

### Multiclass and CPU scheduling
- parallel OVR training with `ovr_parallel` and `ovr_n_jobs`;
- child OVR estimators use `n_jobs=1` whenever outer OVR parallelism is active, preventing nested Numba oversubscription;
- class ordering and random-state offsets remain deterministic.

### Profiling
- stage-level runtime counters for evaluation, lexicase cache/sort, variation, migration and checkpoint I/O;
- history records unique structures, fitness-cache hits/misses and per-generation evaluation/selection-preparation time;
- profile reports cache hit rate, same-batch duplicate reuse, island evaluation mode and cross-island batching state.

### Validation
- dedicated V1.1.3 performance-hardening tests cover typed generation/variation, compiled postfix reuse, same-batch fused duplicate reuse, cross-island batching, stage profiling, parallel OVR scheduling and sklearn constructor stability;
- full source suite passes 71 tests before packaging.

## 1.1.2 — Core Integrity & Backward Compatibility

### Core integrity
- presets no longer mutate public constructor/sklearn parameters during `fit`; runtime values are separated into resolved configuration;
- hard depth/node admission for initialization and variation;
- hard dimensional-unit enforcement now applies to offspring as well as initial programs;
- final requested generation no longer creates an unused next population;
- custom primitive programming exceptions are surfaced instead of silently converted to zero outputs.

### Evolution and search
- real island populations with ring migration;
- NSGA-II rank/crowding connected to multiobjective parent/survivor ordering and whole-population persistent Pareto archive updates;
- generation-level lexicase/epsilon-lexicase case-loss cache removes repeated full-population re-evaluation per parent selection;
- within-generation structural duplicate fitness reuse for materialized scoring.

### Checkpoint and recovery
- checkpoint schema v3 stores next-generation populations/islands, RNG state, generation, history, champion, Pareto archive, stagnation, evaluation count and dataset fingerprint;
- loaded checkpoints are prediction-capable;
- generation-boundary resume continues from the saved evolutionary state;
- atomic primary/`.prev` recovery behavior retained.

### V1 stable surface restoration
- restored `SymbolicProgram` compatibility helpers, symbolic network classifier, symbolic distillation and interpretability/stability helpers;
- restored persistence, branching, report/history/expression output and deployment export entry points;
- restored native missing-value primitives `is_missing` and `coalesce`;
- restored publication benchmark CLI with classification and regression modes.

### Acceleration
- retained V1.1.1 fused regression for weighted MSE, RMSE, MAE, R²-loss and Huber;
- restored fused binary-classification scoring for log loss, accuracy, balanced accuracy, F1 and MCC;
- `elite_logistic` performs fused approximate screening followed by exact logistic refinement of elite candidates;
- incompatible forced-fused configurations fail early while `auto` reports a materialized fallback.

### Validation
- V1.0 usage compatibility tests added alongside V1.1 scientific/core tests;
- fused binary-classification numerical equivalence test added;
- network/distillation, checkpoint/resume, units, islands, Pareto, lexicase caching, persistence and exports smoke-tested.

## 1.1.1 — Fused Regression Performance & Recovery

### Performance
- real fused Numba population scoring for MSE, RMSE, MAE, R²-loss, and Huber regression objectives;
- training affine scaling and validation scoring execute inside the fused kernel without a population-by-sample prediction matrix;
- automatic population chunking when `memory_budget_mb` requires a smaller working set;
- `fitness_engine="auto"` selects fused scoring only when metric, scaling, primitives, and Numba are genuinely compatible;
- forced incompatible fused requests fail early with a concrete compatibility error instead of silently pretending a fused kernel ran.

### Reliability
- weighted fused objectives numerically cross-checked against the materialized metric layer;
- rotating atomic checkpoints retain a validated `.prev` recovery copy;
- corrupt primary checkpoints automatically recover from the previous checkpoint when possible;
- checkpoint schema metadata added to execution diagnostics.

### Validation
- V1.1.1 performance/reliability suite adds fused/materialized equivalence tests for linear/no scaling and weighted MSE/RMSE/MAE/R²/Huber;
- custom metrics and custom primitives verify safe auto fallback and forced-fused rejection;
- memory-budget chunking and checkpoint corruption recovery are regression tested.

## 1.1.0 — Scientific Search, Constraints & Reliability

### Scientific search
- tournament, lexicase and epsilon-lexicase selection;
- adaptive random immigrants/restart behavior;
- persistent Pareto archive;
- safe simplification and canonical commutative ordering;
- deterministic champion constant refinement.

### Scientific constraints
- forbidden-operator, mandatory-variable, max-variable and nonlinear-complexity constraints;
- output lower/upper bounds;
- monotonicity constraints;
- boundary-condition penalties;
- physical unit parsing and dimension-aware expression validation;
- `unit_policy=off|penalty|enforce`.

### Extensibility
- runtime custom primitive registry;
- runtime custom metric registry;
- richer regression metrics including Huber, quantile/pinball, log-cosh, MAPE, sMAPE, RMSLE and MedAE.

### Reliability and uncertainty
- split-conformal regression intervals;
- sklearn clone/search compatibility improvements;
- preset precedence for parameters explicitly supplied by sklearn search via `set_params`;
- transparent backend/fitness-engine resolution;
- atomic checkpoint writing;
- safe JSON model round-trip;
- run fingerprints and reproducibility bundles;
- diagnostics and self-test CLI.

### Acceleration
- optional Numba postfix execution for built-in primitives;
- automatic NumPy fallback for unsupported/custom primitives.

### Compatibility
- V1.0.x constructor parameters used by existing experiments are retained where practical;
- legacy `fitness_engine="fused"` is accepted but transparently resolved to the V1.1.0 materialized scientific metric engine.
