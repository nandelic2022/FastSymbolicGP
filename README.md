# FastSymbolicGP V0.7.0

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Finventions11040080-blue)](https://doi.org/10.3390/inventions11040080)
[![PyPI](https://img.shields.io/pypi/v/fastsymbolicgp)](https://pypi.org/project/fastsymbolicgp/)
[![Python](https://img.shields.io/pypi/pyversions/fastsymbolicgp)](https://pypi.org/project/fastsymbolicgp/)
[![License](https://img.shields.io/github/license/nandelic2022/FastSymbolicGP)](https://github.com/nandelic2022/FastSymbolicGP/blob/main/LICENSE)

FastSymbolicGP is the open-source implementation accompanying the peer-reviewed
article published in *Inventions*:

> Anđelić, N. (2026). FastSymbolicGP: A Lightweight Python Library for
> Efficient Symbolic Regression and Classification. *Inventions*, 11(4), 80.
> https://doi.org/10.3390/inventions11040080
FastSymbolicGP is a scikit-learn-compatible symbolic machine-learning library for classification, regression, supervised symbolic feature construction, model distillation, and experimental symbolic networks.

V0.7.0 is the **Scalable Symbolic Learning** release. It expands the V0.6.0 engine with island evolution, DAG execution, a persistent bounded subtree cache, shared-backbone multiclass learning, resource budgets, deployment export, model profiling, native missing-value operators, robustness objectives, and Publication Benchmark V3.

## Main V0.7.0 capabilities

- binary classification and symbolic regression;
- multiclass OVR and `shared_softmax` symbolic learning;
- NSGA-II-style Pareto ranking and compact final-model selection;
- symbolic probability ensembles and ensemble distillation;
- island evolution with migration and heterogeneous island profiles;
- DAG compilation and common-subexpression compression;
- bounded persistent LRU subtree cache;
- NumPy/Numba postfix evaluator fallback when DAG execution is disabled;
- `FastSymbolicTransformer` V2 with mRMR/diversity/Pareto selection;
- checkpoint, resume, warm start, and checkpoint branching;
- automatic presets and time/evaluation/memory budgets;
- native missing-value primitives and robust-fitness perturbations;
- local feature-occlusion explanations and experimental counterfactual search;
- Python, C, C++, Java, Kotlin, and JavaScript source export;
- inference latency and structural-cost profiling;
- reproducible Publication Benchmark V3 with common saved folds and LaTeX tables;
- a live island-grid terminal dashboard.

## Installation

From the release wheel:

```bat
python -m pip install fastsymbolicgp-0.7.0-py3-none-any.whl
python -m fastsymbolicgp.cli
```

For optional Numba acceleration:

```bat
python -m pip install "fastsymbolicgp[acceleration]"
```

The source ZIP also includes `install_fastsymbolicgp_v070.bat`.

## Binary classification with islands and the live dashboard

```python
from fastsymbolicgp import FastSymbolicClassifier

model = FastSymbolicClassifier(
    population_size=32,
    generations=120,
    optimization="nsga2",
    evolution_model="islands",
    n_islands=4,
    island_profiles=("accurate", "compact", "diverse", "robust"),
    migration_interval=10,
    prediction_mode="symbolic_ensemble",
    ensemble_size=5,
    subtree_cache=True,
    dag_execution="auto",
    probability_calibration="auto",
    preset="balanced",
    display="grid",
    dashboard_interval=5,
    random_state=42,
)
model.fit(X_train, y_train)
```

The report distinguishes the requested postfix evaluator from the actual execution path:

```python
print(model.evaluation_backend_)  # e.g. numba
print(model.execution_engine_)    # e.g. dag_cached_numpy
```

## Shared-backbone multiclass model

```python
model = FastSymbolicClassifier(
    multiclass_strategy="shared_softmax",
    shared_n_components=8,
    multiclass_calibration="temperature",
    population_size=24,
    generations=80,
    optimization="nsga2",
    random_state=42,
)
model.fit(X_train, y_train)
```

The model evolves a shared bank of symbolic features and trains one joint multinomial softmax head. OVR remains available through `multiclass_strategy="ovr"`.

## Symbolic feature construction

```python
from fastsymbolicgp import FastSymbolicTransformer

transformer = FastSymbolicTransformer(
    n_components=12,
    component_selection="mrmr",
    max_correlation=0.90,
    include_original_features=True,
    model_params={"population_size": 30, "generations": 50, "verbose": 0},
    random_state=42,
)
Z_train = transformer.fit_transform(X_train, y_train)
```

## Distillation

```python
distilled = model.distill(
    X_train,
    max_nodes=35,
    population_size=48,
    generations=80,
    verbose=0,
)
```

The distilled model learns the teacher's probability surface using one compact symbolic expression per required output.

## Native missing values and robust fitness

```python
model = FastSymbolicClassifier(
    missing_value_strategy="native",
    function_set=("add", "sub", "mul", "div", "is_missing", "coalesce"),
    robustness_training=True,
    robustness_method="combined",
    robustness_weight=0.05,
)
```

## Resource budgets

```python
model = FastSymbolicClassifier(
    preset="auto",
    time_budget=120,
    evaluation_budget=1_000_000,
    memory_budget_mb=2048,
)
```

`stop_reason_` records whether evolution ended by maximum generations, early stopping, time budget, or evaluation budget.

## Deployment export and profiling

```python
model.export_python("deployed_model.py")
model.export_c("deployed_model.c")
model.export_cpp("deployed_model.hpp")
model.export_java("FastSymbolicModel.java")
model.export_kotlin("FastSymbolicModel.kt")
model.export_javascript("fast_symbolic_model.js")

print(model.profile_prediction(X_test[:100], repeats=100))
```

Python export supports binary, multiclass OVR/shared-softmax, and regression. Portable C/C++/Java/Kotlin/JavaScript export supports binary classification and regression in V0.7.0.

## Checkpoint branching

```python
compact_branch = model.branch(
    parsimony_target_nodes=25,
    generations=180,
)
compact_branch.continue_evolution(X_train, y_train, additional_generations=60)
```

## Publication Benchmark V3

```bat
fastsymbolicgp-benchmark ^
  --datasets breast_cancer,iris,wine ^
  --algorithms fastsymbolicgp,logistic,random_forest,hist_gradient_boosting,svm ^
  --runs 6 ^
  --folds 5 ^
  --population-size 30 ^
  --generations 120 ^
  --islands 4 ^
  --output-dir benchmark_v070_results
```

The benchmark stores common fold indices, raw and summary results, failures, expressions, environment data, algorithm ranks, and LaTeX-ready tables.

## Stable versus experimental

Stable in V0.7.0:

- classifier, regressor, OVR/shared-softmax multiclass;
- island evolution, DAG/cache, Pareto selection, ensembles;
- transformer V2, distillation, checkpointing;
- Python export, binary/regression portable exports;
- benchmark framework and reporting.

Experimental in V0.7.0:

- `FastSymbolicNetworkClassifier`;
- greedy counterfactual search;
- adaptive population sizing and dynamic function weighting;
- robustness fitness for deployment-hardening studies.

These experimental components are implemented and tested but should be benchmarked carefully before strong scientific claims.

## Citation

If FastSymbolicGP contributes to your research, please cite the following article:

bibtex
@article{Andelic2026FastSymbolicGP,
  author         = {Anđelić, Nikola},
  title          = {FastSymbolicGP: A Lightweight Python Library for Efficient
                    Symbolic Regression and Classification},
  journal        = {Inventions},
  year           = {2026},
  volume         = {11},
  number         = {4},
  article-number = {80},
  doi            = {10.3390/inventions11040080}
}
### APA citation

Anđelić, N. (2026). FastSymbolicGP: A lightweight Python library for efficient
symbolic regression and classification. *Inventions, 11*(4), 80.
https://doi.org/10.3390/inventions11040080

## Author

**Nikola Anđelić**

University of Rijeka, Faculty of Engineering  
Department of Computer Engineering
