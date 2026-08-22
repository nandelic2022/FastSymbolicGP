# Changelog

## 1.0.0

### Stable API

- first stable major release of the classifier, regressor, transformer, checkpoint, export, profiling, and benchmark surfaces;
- stable `summary()`, `save()`, `load()`, and `model_manifest()` helpers;
- experimental symbolic network also exposed under `fastsymbolicgp.experimental`.

### Performance

- persistent island thread pool;
- cached indexed program paths for variation operations;
- cached feature/function counts and flat compiled genome metadata;
- retained population Numba, fused fitness, exact fitness cache, duplicate reuse, progressive sampling, and DAG/postfix backends.

### Correctness and model selection

- global exact validation champion archive;
- exact champion candidates refined every generation;
- early stopping based on consistent exact validation fitness when available;
- final selection restricted to exact-scored candidates when available;
- checkpoint persistence for the exact champion.

### Configuration and observability

- named presets preserve explicitly supplied non-default overrides by default;
- `preset_override_policy="always"` restores preset-wins behavior;
- adaptive execution plan for `preset="auto"`;
- `resolved_config_`, `execution_plan_`, `fitness_cache_statistics_`, and `profile_`;
- dashboard separates subtree cache, fitness cache, duplicate reuse, and global exact validation fitness;
- `warmup()` helper for JIT precompilation.

### Search quality

- optional deterministic champion constant optimisation;
- accurate and publication presets enable final constant refinement.

### Benchmarking

- Publication Benchmark V4 naming;
- CLI evaluation-budget support in addition to time budgets;
- V1 forced-120-generation acceptance benchmark.


## 0.9.0

### Added

- fused Numba classification and regression fitness kernels;
- exact bounded fitness cache for unchanged programs;
- intra-population exact duplicate evaluation reuse;
- exact compiled execution hashes with unrounded constants;
- `fitness_engine`, `fitness_cache_max_entries`, and `fitness_cache_scope` parameters;
- `superfast` and `publication` presets;
- actual-evaluation and cache statistics in generation history;
- V0.9.0 numerical-equivalence, cache, and performance tests.

### Changed

- automatic scoring uses the fused engine when the selected metric supports it;
- evaluation budgets count novel panmictic evaluations;
- CLI benchmark accepts `--fitness-engine`;
- diagnostics expose fused-fitness and exact-cache capabilities.

### Retained

- all V0.8.0 estimators, packed population execution, multiclass strategies, islands, transformers, distillation, checkpointing, export, interpretation, robustness, and publication benchmarking.

## 0.8.0

### Added

- packed population-level Numba executor;
- `population_numba` execution backend;
- fused training/validation program evaluation;
- fast one-step logistic scaler for population ranking;
- true elite-only iterative logistic refinement;
- progressive sampling through a two-value `max_samples` tuple;
- `compatibility`, `turbo`, and revised performance presets;
- V0.8.0 performance benchmark and regression tests;
- exact/approximate scaling counts in generation history.

### Changed

- auto backend prefers population Numba when available;
- DAG execution in auto mode is no longer forced by cache availability;
- dataset tokens and fused reference matrices are cached;
- default `n_jobs` is `-1`;
- default semantic duplicate elimination and subtree cache are disabled;
- default elite logistic refinement interval is two generations;
- island scheduling correctly interprets `n_jobs=-1`.

### Retained

- V0.7.0 classifier, regressor, multiclass, transformer, distillation, network, checkpoint, export, profiling, interpretation, resource-budget, robust-fitness, and benchmark APIs.

## 0.7.0

- island-model evolution with migration and profiles;
- DAG execution and bounded subtree caching;
- shared-softmax multiclass learning;
- symbolic transformers, distillation, export, resource budgets, missing-value operators, robust objectives, and publication benchmarking.
