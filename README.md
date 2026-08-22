# FastSymbolicGP V1.0.0

FastSymbolicGP is a scikit-learn-compatible symbolic machine-learning library for fast symbolic classification, regression, symbolic feature construction, distillation, and deployment export.

V1.0.0 is the first **stable API release**. It consolidates the V0.9.0 fused Numba engine with exact champion selection, adaptive execution planning, stable preset semantics, richer performance telemetry, cached program indexing, optional final constant refinement, and publication-grade reproducibility controls.

## Installation

```bat
python -m pip install fastsymbolicgp-1.0.0-py3-none-any.whl
python -m pip install numba
```

## Minimal Spyder example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from fastsymbolicgp import FastSymbolicClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = FastSymbolicClassifier(
    population_size=64,
    generations=120,
    preset="balanced",
    random_state=42,
    verbose=1,
)
model.fit(X_train, y_train)

print("Balanced accuracy:", balanced_accuracy_score(y_test, model.predict(X_test)))
print("Expression:", model.get_expression())
model.summary()
```

Named presets are now applied as defaults: explicitly supplied **non-default** parameter values are preserved. Use `preset_override_policy="always"` when the preset must replace them. `preset="auto"` additionally activates the workload-aware execution planner.

## V1.0.0 core improvements

- stable `FastSymbolicClassifier`, `FastSymbolicRegressor`, and `FastSymbolicTransformer` APIs;
- population-level Numba postfix execution and fused classification/regression fitness;
- exact run-level fitness reuse and intra-population duplicate reuse;
- cached program size, depth, DAG, compiled postfix, structural hash, feature counts, function counts, and indexed variation paths;
- flat compiled genome metadata through `program.flat_genome()`;
- persistent thread pool for island evaluation instead of recreating workers every generation;
- exact validation champion archive that is never lost through migration or Pareto replacement;
- exact champion candidates refined every generation even when broader elite refinement runs less often;
- exact-fitness-only final selection pool when exact candidates are available;
- adaptive execution planner for `preset="auto"`;
- progressive sampling for large datasets;
- structural and optional semantic duplicate elimination;
- adaptive operator and function weighting options;
- adaptive parsimony and NSGA-II-style Pareto optimisation;
- optional final champion constant optimisation with deterministic coordinate search;
- binary threshold optimisation, calibration, class/sample weighting, OVR and shared-softmax multiclass;
- time budgets and evaluation budgets;
- checkpoint/resume, warm start, and branching;
- model persistence, manifest generation, deployment export, and inference profiling;
- `warmup()` for precompiling common kernels before timed experiments;
- `profile_`, `resolved_config_`, `execution_plan_`, and `summary()` for transparent runs;
- Publication Benchmark V4 with common folds and time/evaluation budget support.

## Presets

- `superfast` — aggressive exploratory throughput;
- `fast` — general high-speed search;
- `balanced` — stronger search with moderate complexity;
- `accurate` — larger search and final champion constant refinement;
- `interpretable` — compact final expressions;
- `large_dataset` — progressive sampling and low-memory execution;
- `publication` — full-data, reproducibility-oriented configuration with exact champion handling and constant refinement;
- `compatibility` — conservative materialized scoring for comparisons;
- `auto` — chooses a preset and then adapts execution topology to workload size.

## Warm-up before timing

```python
from fastsymbolicgp import warmup
print(warmup("all"))
```

This isolates first-use JIT compilation from the training time you want to report.

## Exact champion model selection

`elite_logistic` may use approximate scaling for broad screening. V1.0.0 always refines a configurable set of champion candidates exactly (`exact_champion_candidates`) and maintains `exact_validation_champion_`. Early stopping and the final exact-selection pool use this consistent validation scale.

Useful attributes include:

```python
print(model.best_validation_program_)
print(model.profile_)
print(model.execution_plan_)
print(model.resolved_config_)
print(model.fitness_cache_statistics_)
```

## Publication benchmark V4

```bat
fastsymbolicgp-benchmark ^
  --datasets breast_cancer,iris,wine ^
  --algorithms fastsymbolicgp,logistic,random_forest,hist_gradient_boosting,svm ^
  --runs 6 ^
  --folds 5 ^
  --population-size 64 ^
  --generations 120 ^
  --backend population_numba ^
  --fitness-engine fused ^
  --evaluation-budget 100000 ^
  --output-dir benchmark_v100_results
```

You can use `--time-budget` instead of, or together with, `--evaluation-budget` for equal-compute studies.

## Stable versus experimental

The V1 stable surface includes classifier, regressor, transformer, multiclass learning, Pareto search, checkpointing, export, profiling, benchmark utilities, and model persistence. Experimental symbolic-network functionality remains available for backward compatibility and is also exposed through `fastsymbolicgp.experimental`.

## Local V1 acceptance benchmark

On the build environment, the forced 120-generation Breast Cancer workload (64 programs, two islands, Numba population execution, fused fitness, no early stopping, dashboard off, warmed kernels) completed in **4.19 s** with V1.0.0 versus **4.33 s** with V0.9.0. Test balanced accuracy was **0.9579** for V1.0.0 versus **0.9501** for V0.9.0 in that deterministic run. These are build-machine validation figures, not universal performance guarantees.

## Citation

If FastSymbolicGP contributes to your research, please cite:

```bibtex
@article{Andelic2026FastSymbolicGP,
  author         = {Anđelić, Nikola},
  title          = {FastSymbolicGP: A Lightweight Python Library for Efficient Symbolic Regression and Classification},
  journal        = {Inventions},
  year           = {2026},
  volume         = {11},
  number         = {4},
  article-number = {80},
  doi            = {10.3390/inventions11040080}
}
```
