# FastSymbolicGP

**Current release: 1.2.8** · Python 3.11+ · [GitHub](https://github.com/nandelic2022/FastSymbolicGP) · [PyPI](https://pypi.org/project/fastsymbolicgp/)

## V1.2.8 — Batched Tournament Scheduler

V1.2.8 accelerates the standard scalar+tournament path with a vectorized exact-distribution tournament scheduler, while retaining the Numba-first rootless genome engine, fused regression scoring, scientific constraints, unit-aware GP, publication tooling, and expression simplification from prior releases.

For the fastest standard regression path use built-in primitives, `evaluation_backend="numba"`, `fitness_engine="fused"`, and `optimization="scalar"`.


## V1.2.7 Numba-first rootless fast path

For the standard scalar+tournament workflow, FastSymbolicGP now evolves postfix genomes without keeping a Python tree for every candidate. Numba analyzes genome topology, while crossover/mutation operate directly on postfix arrays. The final champion is materialized back to the normal symbolic tree before public inspection/export.

After fitting:

```python
print(model.profile_["variation_engine"])
# numba_rootless_genome_v2
```

Scientific constraints, unit-aware typed GP, Semantic GP, memetic search and evolution-time algebraic simplification automatically use the safe tree-backed path.

An optional `examples/benchmark_vs_gplearn.py` script is included for local speed/quality comparison when gplearn is installed.


## V1.2.6 fast path

The standard scalar+tournament workflow can now inherit the exact postfix genome during crossover/mutation instead of recompiling each offspring tree before fused Numba evaluation. Inspect it with:

```python
print(model.profile_["variation_engine"])
# flat_genome_inheritance_v1
```

This optimization preserves the tree representation for equations, scientific constraints, serialization and explainability.


## V1.2.4 — Expression Quality & Simplification

V1.2.4 keeps the V1.2.3 ultra-fast search core and focuses on equation quality. The default final regressor pass performs empirically certified pruning: candidate rewrites are accepted only when the fitted predictions remain numerically equivalent on the model-development reference set. `preset="publication"` deliberately resolves to globally safe simplification only.

New public APIs include `get_raw_expression()`, a corrected `get_expression()` that includes the fitted affine scale/intercept, `get_simplification_report()`, `safe_simplify_program()`, and `certified_simplify_program()`. `verbose=2` now reports every generation with a compact progress line; `verbose=1` keeps periodic output.

FastSymbolicGP is a scikit-learn-style symbolic machine-learning library for symbolic regression, classification, scientific constraints, physical-unit-aware search, uncertainty, reproducibility, and high-throughput CPU execution.



## V1.2.3 — Ultra-Fast Quality-Preserving Core

V1.2.3 keeps the search behavior scientifically useful while removing more evaluator overhead. For MSE/RMSE/R2 objectives, the fused Numba kernel now computes the exact scaled training loss from first-pass sufficient statistics, eliminating the second symbolic-expression pass over the training split. Population genomes are packed with NumPy array concatenation rather than per-node Python list expansion, and Numba compile-cache checks use immutable root identity instead of repeatedly computing structural hashes.

The standard scalar/tournament path now uses a quality-preserving fast tournament: it retains true without-replacement tournament sampling while exploiting the fact that the population is already fitness-sorted, avoiding candidate-list allocation and repeated winner comparisons. Scientific constraints, typed GP, Semantic GP, Pareto/NSGA-II and checkpoint behavior remain available through their safe generic paths.

On the bundled runtime, the 500-population × 100-generation Diabetes benchmark after JIT warm-up completed in roughly **11–13 s** across repeated certification runs. The final clean-wheel run was **12.99 s** and achieved **test R2 = 0.46235** and **RMSE = 53.37**, versus **R2 = 0.45260** for scikit-learn `LinearRegression` on the same split. These are workload-specific measurements, not universal performance or accuracy guarantees.

## V1.2.2 — Fast Variation Engine Hotfix

V1.2.2 targets the dominant bottleneck found by profiling the 500-population × 100-generation Diabetes benchmark. It keeps the V1.2.1 fused-Numba/scalar fast path and substantially reduces Python-side variation overhead through zero-copy tournament handling, bulk tournament schedules, persistent-tree structural sharing, cached immutable paths, single-pass size/depth statistics, fast basic admissibility checks, and cheap terminal evolution snapshots. Internal DataFrame/NumPy transitions no longer emit the sklearn feature-name warning seen in V1.2.1.

On the bundled runtime, the exact 500×100 four-operator Diabetes speed test completed in about **10.1 s** after JIT warm-up, with variation about **4.7 s** and fused evaluation about **4.8 s**. This is a workload-specific measurement and should be compared on the same machine/environment.

## V1.2.1 — Performance Regression Hotfix

V1.2.1 removes the Python-side evolution bottlenecks discovered in V1.2.0 while preserving the V1.2 search-intelligence features. Normal regressors/classifiers now default to `optimization="scalar"`; the `publication` preset explicitly retains Pareto/NSGA-II search. Pareto reporting remains available in scalar mode and is built once at the end instead of being maintained every generation.

Key hot-path changes:

- NSGA-II ranking is performed at most once per generation when Pareto search is explicitly enabled.
- Objective vectors are precomputed once per NSGA-II pass instead of being rebuilt inside pairwise dominance comparisons.
- Pareto archive maintenance is skipped during scalar generations and performed once at the end for reporting.
- Pareto archive updates use incremental first-front maintenance rather than a full repeated nondominated sort of archive + population.
- Full resumable evolution-state cloning occurs only for requested checkpoints and the terminal generation.
- New profiler stages expose `nsga2`, `pareto_archive`, and `state_snapshot` costs, plus counters for the number of calls/updates/snapshots.

On the bundled development runtime, a 500-individual, 2-generation fused-Numba diabetes smoke run fell from about 13.1 s in V1.2.0 to about 0.45 s in V1.2.1. This is a workload-specific measurement, not a universal speedup claim.

## V1.2.0 — Search Intelligence & Memetic Evolution

V1.2.0 builds on the fully certified V1.1.5 line with Semantic GP 2.0, age-layered evolution, canonical symbolic forms, in-evolution memetic constant refinement and explicit final-model selection. All new search-intelligence systems are opt-in by default so existing V1.1.x workflows retain their behavior.

### V1.2.0 example

```python
model = FastSymbolicRegressor(
    semantic_diversity=True,
    selection_method="semantic_tournament",
    semantic_mutation_rate=0.10,
    age_management="layered",
    memetic_search="elites",
    simplification="canonical",
    model_selection="pareto_knee",
    random_state=42,
)
```

Diagnostics are available through `get_semantic_report()` and `get_evolution_diagnostics()`.


## V1.1.5 — Publication & Scientific Certification

V1.1.5 turns FastSymbolicGP experiments into resumable, statistically analyzable publication workflows without replacing the stable V1.1.4 scientific/evolution core.

### What is new in 1.1.5

- **Resumable repeated cross-validation benchmarking** with fold-level incremental persistence.
- Deterministic **Nguyen 1–10** symbolic-regression functions and **Friedman-1** benchmark data.
- Paired common-block **Friedman** omnibus tests.
- Pairwise **Wilcoxon signed-rank** tests with **Holm** multiplicity correction.
- **Rank-biserial effect sizes**, average ranks, and Nemenyi critical difference when SciPy is available.
- Statistical analyses exclude incomplete blocks rather than silently mixing paired and unpaired results.
- Cross-run structural/prediction **stability reports**.
- Input-perturbation **robustness reports**.
- Machine-readable **scientific certificates** combining fit state, dimensional/constraint certification, complexity, run fingerprint, and profile information.
- **Reproducibility bundle v2** with safe-model export, reference arrays, environment capture and SHA-256 content hashes.
- New `fastsymbolicgp-certify` CLI.
- V1.1.4 typed-unit/hard-constraint support and V1.1.3 performance hardening are retained.

## Install

```bash
python -m pip install "fastsymbolicgp[acceleration]==1.2.8"
```

Optional acceleration and scientific dependencies:

```bash
python -m pip install "fastsymbolicgp[acceleration]"
python -m pip install "fastsymbolicgp[science]"
```

Then verify the installation:

```bash
fastsymbolicgp-self-test
```

## Regression

```python
from fastsymbolicgp import FastSymbolicRegressor

model = FastSymbolicRegressor(
    preset="balanced",
    fitness_engine="auto",
    evaluation_cache=True,
    fitness_cache_max_entries=100000,
    uncertainty="split_conformal",
    random_state=42,
)
model.fit(X_train, y_train)

print(model.get_expression())
y_pred = model.predict(X_test)
y_pred, lower, upper = model.predict_interval(X_test, confidence=0.95)
print(model.resolved_config_)
print(model.profile_["stage_seconds"])
```

## Classification and parallel OVR

```python
from fastsymbolicgp import FastSymbolicClassifier

model = FastSymbolicClassifier(
    preset="balanced",
    fitness_scaling="elite_logistic",
    fitness_engine="auto",
    probability_calibration="auto",
    ovr_parallel=True,
    ovr_n_jobs="auto",
    random_state=42,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)
pred = model.predict(X_test)
print(model.profile_)
```

For multiclass OVR, outer class models can train concurrently. When more than one OVR worker is active, child estimators are automatically limited to `n_jobs=1` to avoid nested CPU oversubscription.

## Strongly typed scientific units

```python
model = FastSymbolicRegressor(
    function_set=("add", "sub", "mul", "div", "sqrt", "abs"),
    feature_units={
        "power_1": "W",
        "power_2": "W",
        "ratio": "1",
    },
    target_unit="W",
    unit_policy="enforce",
    typed_generation="auto",
    random_state=42,
)
```

With `unit_policy="enforce"` and known feature/target units, `typed_generation="auto"` activates dimension-aware tree construction and unit-preserving variation. `typed_generation="off"` retains the V1.1.2 generate/validate path if required for comparison studies.

### Typed constants, virtual constraints and Pareto objectives

```python
model = FastSymbolicRegressor(
    function_set=("add", "sub", "mul", "div"),
    feature_units={"pressure": "bar", "flow": "m^3/s", "temperature": "degC"},
    target_unit="kW",
    unit_policy="enforce",
    constant_units={"kW": (-50.0, 50.0), "1": (-2.0, 2.0)},
    constraints={
        "mode": "hard",
        "monotonic": {"pressure": "increasing"},
        "virtual_n_points": 128,
        "virtual_strategy": "latin_hypercube",
        "virtual_ranges": {"pressure": [1.0, 20.0], "flow": [0.01, 2.0]},
    },
    pareto_objectives=("validation_loss", "weighted_complexity", "variables"),
    random_state=42,
)
model.fit(X_train, y_train)

print(model.get_unit_report())
print(model.get_constraint_report())
print(model.get_pareto_front_dataframe())
```

Physics-informed residuals may be supplied in `constraints["residuals"]`. A residual callable receives `(X, prediction)` or `(X, prediction, program)` and returns a scalar or one residual per constraint point. Because Python callables are runtime objects, they should be recreated when reloading a portable JSON model if post-load constraint recertification is required.

## Island evolution and cross-island fused batching

```python
model = FastSymbolicRegressor(
    evolution_model="islands",
    n_islands=4,
    island_parallel=True,
    migration_interval=5,
    migration_size=2,
    fitness_engine="auto",
    random_state=42,
)
```

`population_size` remains the total population. In compatible fused workloads, all islands are scored in one population-level Numba batch and then continue with island-local selection/migration.

## Fitness and compiled-program caching

```python
model = FastSymbolicRegressor(
    evaluation_cache=True,
    fitness_cache_max_entries=100000,
)
```

The cache reuses exact structural duplicates across and within batches. Compiled postfix genomes are also cached per unchanged program structure. Diagnostics are exposed in `profile_` and `history_`.

## Checkpoint and resume

```python
model = FastSymbolicRegressor(
    generations=200,
    checkpoint_path="run.chk",
    checkpoint_interval=5,
    random_state=42,
)
model.fit(X, y)

restored = FastSymbolicRegressor.load_checkpoint("run.chk")
prediction = restored.predict(X[:10])
restored.resume_from_checkpoint = True
restored.generations = 300
restored.fit(X, y)
```

Checkpoints use atomic rotation and keep a `.prev` recovery copy. Resume validates the training-data fingerprint and continues from the saved next-generation population/RNG state.

## V1 compatibility surface

```python
from fastsymbolicgp import (
    FastSymbolicRegressor,
    FastSymbolicClassifier,
    FastSymbolicTransformer,
    FastSymbolicNetworkClassifier,
    DistilledSymbolicClassifier,
    SymbolicProgram,
    feature_stability,
    subexpression_stability,
)
```

Persistence/report helpers include `save_model`, `load_model`, `save`, `load`, `save_history_csv`, `save_report_json`, `save_expression`, `profile_prediction`, `summary`, `model_manifest` and `branch`.

## Deployment and reproducibility

```python
model.export_python("model.py")
model.export_c("model.c")
model.export_cpp("model.cpp")
model.export_javascript("model.js")
model.export_reproducibility_bundle("run_001")
```

Python/C/C++/JavaScript portable export is supported for compatible estimator surfaces. Java/Kotlin portable code generation remains intentionally unavailable in V1.2.3 rather than returning an unverified export. Safe JSON and LaTeX/SymPy paths remain available where supported.

## Benchmark CLI

```bash
fastsymbolicgp-benchmark --task regression --datasets diabetes --algorithms fastsymbolicgp,linear,random_forest,hist_gradient_boosting --folds 5 --runs 3 --output-dir benchmark_results
```

or classification:

```bash
fastsymbolicgp-benchmark --task classification --datasets breast_cancer,iris,wine --algorithms fastsymbolicgp,logistic,random_forest,hist_gradient_boosting --folds 5 --runs 3 --output-dir benchmark_results
```

## Scope and known limits

V1.2.4 remains CPU-first and single-workstation. Numba acceleration is optional. Multiclass uses OVR; the older experimental shared-softmax path is not certified. Java/Kotlin portable code export is not certified. Strongly typed unit generation uses a bounded dimensional grammar closure: if the requested target dimension cannot be produced from the available feature dimensions and primitive set, fitting fails clearly instead of silently admitting invalid programs.

## Citation

If FastSymbolicGP contributes to research, cite the FastSymbolicGP publication and record the exact package version, resolved configuration, run fingerprint, and benchmark protocol used for the experiment.

## V1.2.6 runtime console

FastSymbolicGP now shows a dependency-free runtime dashboard by default when `verbose>=1`.
Use `verbose=2` to print every generation, or `display="off"` for silent execution.
