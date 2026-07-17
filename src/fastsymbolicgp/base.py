"""Shared evolutionary engine for FastSymbolicGP estimators.

V0.7.0 adds island evolution, DAG execution, persistent subtree caching,
resource budgets, adaptive populations, dynamic primitive weighting, native
missing-value handling, robustness objectives, checkpoint branching, and a
richer live terminal dashboard.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import copy
import os
import platform
import sys
import tracemalloc

import joblib
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ._version import __version__
from .backend import NUMBA_AVAILABLE
from .cache import ArrayLRUCache, dataset_token
from .dag import execute_dag
from .functions import DEFAULT_FUNCTION_SET, validate_function_set
from .history import save_history_csv, save_json
from .presets import apply_preset
from .program import (
    SymbolicProgram, random_tree, subtree_crossover, subtree_mutation,
    hoist_mutation, point_mutation, enforce_limits, semantic_hash,
)
from .simplification import simplify_program
from .validation import check_array_finite


_OPERATOR_NAMES = ("crossover", "subtree", "hoist", "point")


class BaseFastSymbolicGP(BaseEstimator):
    """Shared validation-aware symbolic evolutionary engine."""

    def __init__(
        self,
        population_size=500,
        generations=50,
        tournament_size=20,
        function_set=DEFAULT_FUNCTION_SET,
        init_depth=(2, 6),
        max_depth=12,
        max_nodes=255,
        p_crossover=0.70,
        p_subtree_mutation=0.15,
        p_hoist_mutation=0.05,
        p_point_mutation=0.10,
        crossover_rate=None,
        subtree_mutation_rate=None,
        hoist_mutation_rate=None,
        point_mutation_rate=None,
        mutation_depth=(1, 4),
        const_range=(-1.0, 1.0),
        const_probability=0.15,
        max_samples=1.0,
        parsimony_coefficient=0.0,
        parsimony="adaptive",
        parsimony_target_nodes=40,
        parsimony_growth_rate=1.05,
        validation_fraction=0.20,
        selection_metric="validation",
        validation_gap_penalty=0.25,
        final_selection="smallest_within_tolerance",
        selection_tolerance=0.02,
        optimization="scalar",
        pareto_algorithm="nsga2",
        patience=20,
        min_delta=1e-6,
        elitism=2,
        hall_of_fame_size=50,
        duplicate_elimination=True,
        semantic_duplicate_elimination=True,
        semantic_sample_size=128,
        max_duplicate_attempts=20,
        simplify_expression=True,
        operator_adaptation=False,
        operator_adaptation_interval=5,
        operator_min_probability=0.02,
        reject_oversized_offspring=True,
        tarpeian_rate=0.0,
        evaluation_cache=True,
        subtree_cache=True,
        subtree_cache_scope="run",
        subtree_cache_max_mb=512,
        dag_execution="auto",
        complexity_measure="dag",
        evaluation_backend="auto",
        batch_size="auto",
        n_jobs=1,
        thread_limit="auto",
        evolution_model="panmictic",
        n_islands=1,
        migration_interval=10,
        migration_size=2,
        migration_strategy="ring",
        island_profiles=None,
        island_parallel=True,
        adaptive_population=False,
        population_min=None,
        population_max=None,
        function_set_adaptation=False,
        function_adaptation_interval=10,
        time_budget=None,
        evaluation_budget=None,
        memory_budget_mb=None,
        missing_value_strategy="error",
        missing_value_constant=0.0,
        robustness_training=False,
        robustness_method="combined",
        robustness_weight=0.0,
        robustness_noise=0.01,
        feature_dropout_rate=0.02,
        preset=None,
        checkpoint_path=None,
        checkpoint_interval=0,
        resume_from_checkpoint=False,
        random_state=None,
        verbose=1,
        display="dashboard",
        dashboard_interval=1,
        use_color="auto",
        warm_start=False,
        low_memory=False,
        **kwargs,
    ):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.function_set = function_set
        self.init_depth = init_depth
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.crossover_rate = crossover_rate
        self.subtree_mutation_rate = subtree_mutation_rate
        self.hoist_mutation_rate = hoist_mutation_rate
        self.point_mutation_rate = point_mutation_rate
        self.mutation_depth = mutation_depth
        self.const_range = const_range
        self.const_probability = const_probability
        self.max_samples = max_samples
        self.parsimony_coefficient = parsimony_coefficient
        self.parsimony = parsimony
        self.parsimony_target_nodes = parsimony_target_nodes
        self.parsimony_growth_rate = parsimony_growth_rate
        self.validation_fraction = validation_fraction
        self.selection_metric = selection_metric
        self.validation_gap_penalty = validation_gap_penalty
        self.final_selection = final_selection
        self.selection_tolerance = selection_tolerance
        self.optimization = optimization
        self.pareto_algorithm = pareto_algorithm
        self.patience = patience
        self.min_delta = min_delta
        self.elitism = elitism
        self.hall_of_fame_size = hall_of_fame_size
        self.duplicate_elimination = duplicate_elimination
        self.semantic_duplicate_elimination = semantic_duplicate_elimination
        self.semantic_sample_size = semantic_sample_size
        self.max_duplicate_attempts = max_duplicate_attempts
        self.simplify_expression = simplify_expression
        self.operator_adaptation = operator_adaptation
        self.operator_adaptation_interval = operator_adaptation_interval
        self.operator_min_probability = operator_min_probability
        self.reject_oversized_offspring = reject_oversized_offspring
        self.tarpeian_rate = tarpeian_rate
        self.evaluation_cache = evaluation_cache
        self.subtree_cache = subtree_cache
        self.subtree_cache_scope = subtree_cache_scope
        self.subtree_cache_max_mb = subtree_cache_max_mb
        self.dag_execution = dag_execution
        self.complexity_measure = complexity_measure
        self.evaluation_backend = evaluation_backend
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.thread_limit = thread_limit
        self.evolution_model = evolution_model
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.migration_strategy = migration_strategy
        self.island_profiles = island_profiles
        self.island_parallel = island_parallel
        self.adaptive_population = adaptive_population
        self.population_min = population_min
        self.population_max = population_max
        self.function_set_adaptation = function_set_adaptation
        self.function_adaptation_interval = function_adaptation_interval
        self.time_budget = time_budget
        self.evaluation_budget = evaluation_budget
        self.memory_budget_mb = memory_budget_mb
        self.missing_value_strategy = missing_value_strategy
        self.missing_value_constant = missing_value_constant
        self.robustness_training = robustness_training
        self.robustness_method = robustness_method
        self.robustness_weight = robustness_weight
        self.robustness_noise = robustness_noise
        self.feature_dropout_rate = feature_dropout_rate
        self.preset = preset
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.resume_from_checkpoint = resume_from_checkpoint
        self.random_state = random_state
        self.verbose = verbose
        self.display = display
        self.dashboard_interval = dashboard_interval
        self.use_color = use_color
        self.warm_start = warm_start
        self.low_memory = low_memory
        self._extra_kwargs = kwargs

    # ----------------------------- data and validation -----------------------------
    def _apply_preset(self, X, y=None):
        if self.preset not in {None, "none", "custom", "off"}:
            apply_preset(self, X, y)
        else:
            self.preset_ = "custom"
            self.preset_parameters_ = {}

    def _allow_nan(self):
        return str(self.missing_value_strategy).lower() in {"native", "median", "constant"}

    def _prepare_X_fit(self, X):
        feature_names = getattr(X, "columns", None)
        X = check_array_finite(X, dtype=np.float64, ensure_2d=True, allow_nan=self._allow_nan())
        self.n_features_in_ = X.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        strategy = str(self.missing_value_strategy).lower()
        if strategy == "error":
            self.missing_fill_values_ = None
        elif strategy == "median":
            fill = np.nanmedian(X, axis=0)
            fill = np.where(np.isfinite(fill), fill, 0.0)
            self.missing_fill_values_ = fill
            X = np.where(np.isnan(X), fill, X)
        elif strategy == "constant":
            fill = np.full(X.shape[1], float(self.missing_value_constant), dtype=float)
            self.missing_fill_values_ = fill
            X = np.where(np.isnan(X), fill, X)
        elif strategy == "native":
            self.missing_fill_values_ = None
        else:
            raise ValueError("missing_value_strategy must be error, median, constant, or native")
        return np.ascontiguousarray(X)

    def _prepare_X_predict(self, X):
        X = check_array_finite(X, dtype=np.float64, ensure_2d=True, allow_nan=self._allow_nan())
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        if getattr(self, "missing_fill_values_", None) is not None:
            X = np.where(np.isnan(X), self.missing_fill_values_, X)
        return np.ascontiguousarray(X)

    def _validate_common_parameters(self):
        if int(self.population_size) < 2:
            raise ValueError("population_size must be >= 2")
        if int(self.generations) < 1:
            raise ValueError("generations must be >= 1")
        if int(self.tournament_size) < 2:
            raise ValueError("tournament_size must be >= 2")
        if not (0 <= float(self.validation_fraction) < 0.5):
            raise ValueError("validation_fraction must be in [0, 0.5)")
        if not (0.0 <= float(self.tarpeian_rate) <= 1.0):
            raise ValueError("tarpeian_rate must be in [0, 1]")
        if not (0.0 <= float(self.robustness_weight) <= 1.0):
            raise ValueError("robustness_weight must be in [0, 1]")
        functions = list(validate_function_set(self.function_set))
        if str(self.missing_value_strategy).lower() == "native":
            for name in ("is_missing", "coalesce"):
                if name not in functions:
                    functions.append(name)
        self._function_set = tuple(functions)
        self._function_weights = {name: 1.0 for name in self._function_set}
        self._init_depth = tuple(map(int, self.init_depth))
        self._mutation_depth = tuple(map(int, self.mutation_depth))
        rates = [
            self.p_crossover if self.crossover_rate is None else self.crossover_rate,
            self.p_subtree_mutation if self.subtree_mutation_rate is None else self.subtree_mutation_rate,
            self.p_hoist_mutation if self.hoist_mutation_rate is None else self.hoist_mutation_rate,
            self.p_point_mutation if self.point_mutation_rate is None else self.point_mutation_rate,
        ]
        rates = np.asarray(rates, dtype=float)
        if np.any(rates < 0) or rates.sum() <= 0:
            raise ValueError("genetic operator probabilities must be non-negative and sum to > 0")
        self._operator_rates = rates / rates.sum()
        if self.evaluation_backend not in {"numpy", "numba", "auto", "tree"}:
            raise ValueError("evaluation_backend must be numpy, numba, auto, or tree")
        self.evaluation_backend_ = (
            "numba" if self.evaluation_backend == "auto" and NUMBA_AVAILABLE
            else "numpy" if self.evaluation_backend == "auto"
            else self.evaluation_backend
        )
        dag_mode = str(self.dag_execution).lower()
        dag_active = dag_mode in {"true", "on", "dag"} or (
            dag_mode == "auto" and bool(self.subtree_cache)
        )
        self.execution_engine_ = (
            "dag_cached_numpy" if dag_active and bool(self.subtree_cache)
            else "dag_numpy" if dag_active
            else f"postfix_{self.evaluation_backend_}"
        )
        if self.selection_metric not in {"training", "validation", "combined"}:
            raise ValueError("selection_metric must be training, validation, or combined")
        if self.final_selection not in {"raw", "validation", "smallest_within_tolerance", "pareto"}:
            raise ValueError("Unsupported final_selection")
        if self.optimization not in {"scalar", "pareto", "nsga2"}:
            raise ValueError("optimization must be scalar, pareto, or nsga2")
        if self.pareto_algorithm not in {"nsga2", "lexicographic"}:
            raise ValueError("pareto_algorithm must be nsga2 or lexicographic")
        if str(self.evolution_model).lower() not in {"panmictic", "single", "islands", "island"}:
            raise ValueError("evolution_model must be panmictic or islands")
        if str(self.migration_strategy).lower() not in {"ring", "random", "best_to_worst", "diversity_guided"}:
            raise ValueError("Unsupported migration_strategy")
        if str(self.complexity_measure).lower() not in {"tree", "dag"}:
            raise ValueError("complexity_measure must be tree or dag")
        cache_mb = float(self.subtree_cache_max_mb)
        if self.memory_budget_mb is not None:
            cache_mb = min(cache_mb, max(0.0, float(self.memory_budget_mb) * 0.65))
        if not hasattr(self, "_subtree_cache_") or str(self.subtree_cache_scope).lower() != "run":
            self._subtree_cache_ = ArrayLRUCache(int(cache_mb * 1024 * 1024))
        else:
            self._subtree_cache_.max_bytes = int(cache_mb * 1024 * 1024)

    # ----------------------------- population -----------------------------
    def _initial_population(self, rng, n_features, population_size=None):
        target_size = int(self.population_size if population_size is None else population_size)
        population, seen, attempts = [], set(), 0
        min_depth = max(1, min(int(self._init_depth[0]), int(self.max_depth)))
        requested_max = max(min_depth, min(int(self._init_depth[1]), int(self.max_depth)))
        safe_full_depth = max(1, int(np.floor(np.log2(max(2, int(self.max_nodes) + 1)))))
        while len(population) < target_size:
            grow = bool(len(population) % 2)
            upper = requested_max if grow else min(requested_max, safe_full_depth)
            target_depth = int(rng.integers(min_depth, upper + 1)) if upper > min_depth else min_depth
            root = random_tree(
                rng, n_features, self._function_set, min_depth, target_depth,
                tuple(self.const_range), float(self.const_probability), grow,
                function_weights=self._function_weights,
            )
            program = SymbolicProgram(root)
            attempts += 1
            if not enforce_limits(program, int(self.max_depth), int(self.max_nodes)):
                if attempts <= target_size * 200:
                    continue
                root = random_tree(
                    rng, n_features, self._function_set, 1, min(3, int(self.max_depth)),
                    tuple(self.const_range), float(self.const_probability), True,
                    function_weights=self._function_weights,
                )
                program = SymbolicProgram(root)
            key = program.structural_hash()
            if not self.duplicate_elimination or key not in seen or attempts > target_size * 100:
                population.append(program); seen.add(key)
        return population

    def _use_dag(self, program):
        mode = str(self.dag_execution).lower()
        if mode in {"false", "off", "none", "postfix"}:
            return False
        if mode in {"true", "on", "dag"}:
            return True
        return bool(self.subtree_cache) or program.dag_compression_ratio > 0.0

    def _execute_program(self, program, X):
        if self._use_dag(program):
            cache = self._subtree_cache_ if bool(self.subtree_cache) else None
            return execute_dag(X, program.compile_dag(), cache=cache, cache_namespace=dataset_token(X))
        return program.execute(X, backend=self.evaluation_backend_)

    def _evaluate_program_cached(self, program, X):
        before = self._subtree_cache_.statistics()["hits"] if hasattr(self, "_subtree_cache_") else 0
        values = self._execute_program(program, X)
        after = self._subtree_cache_.statistics()["hits"] if hasattr(self, "_subtree_cache_") else before
        return values, max(0, after - before)

    def _perturb_X(self, X, rng):
        method = str(self.robustness_method).lower()
        perturbed = np.asarray(X, dtype=float).copy()
        if method in {"gaussian_noise", "combined", "noise"} and float(self.robustness_noise) > 0:
            scales = np.nanstd(perturbed, axis=0)
            scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
            perturbed += rng.normal(0.0, float(self.robustness_noise), size=perturbed.shape) * scales
        if method in {"feature_dropout", "combined", "dropout"} and float(self.feature_dropout_rate) > 0:
            mask = rng.random(perturbed.shape) < float(self.feature_dropout_rate)
            perturbed[mask] = 0.0
        if method in {"quantization", "combined"}:
            perturbed = np.round(perturbed, decimals=4)
        return np.ascontiguousarray(perturbed)

    def _robustness_loss(self, program, raw, y, sample_weight):
        return None

    def _evaluate_population(
        self, population, X_train, y_train, X_val, y_val,
        sample_weight_train, sample_weight_val, rng, parallel_jobs=None,
    ):
        median_nodes = float(np.median([p.size for p in population]))
        tarpeian_draws = rng.random(len(population))
        X_train_robust = X_val_robust = None
        if self.robustness_training and float(self.robustness_weight) > 0:
            X_train_robust = self._perturb_X(X_train, rng)
            if X_val is not None:
                X_val_robust = self._perturb_X(X_val, rng)

        def evaluate_one(index, program):
            if float(self.tarpeian_rate) > 0 and program.size > median_nodes and tarpeian_draws[index] < float(self.tarpeian_rate):
                program.raw_fitness_ = 1e12
                program.validation_fitness_ = 1e12
                program.metadata_["tarpeian"] = True
                return program, 0
            raw_train, hits = self._evaluate_program_cached(program, X_train)
            raw_val = None
            if X_val is not None:
                raw_val, val_hits = self._evaluate_program_cached(program, X_val)
                hits += val_hits
            scored = self._score_program(program, raw_train, y_train, raw_val, y_val, sample_weight_train, sample_weight_val)
            if X_train_robust is not None:
                robust_raw, robust_hits = self._evaluate_program_cached(program, X_train_robust)
                hits += robust_hits
                robust_loss = self._robustness_loss(program, robust_raw, y_train, sample_weight_train)
                if robust_loss is not None:
                    w = float(self.robustness_weight)
                    scored.metadata_["robust_training_loss"] = float(robust_loss)
                    scored.raw_fitness_ = (1.0 - w) * float(scored.raw_fitness_) + w * float(robust_loss)
                if X_val_robust is not None:
                    robust_val_raw, robust_val_hits = self._evaluate_program_cached(program, X_val_robust)
                    hits += robust_val_hits
                    robust_val_loss = self._robustness_loss(program, robust_val_raw, y_val, sample_weight_val)
                    if robust_val_loss is not None:
                        scored.metadata_["robust_validation_loss"] = float(robust_val_loss)
                        scored.validation_fitness_ = (1.0 - w) * float(scored.validation_fitness_) + w * float(robust_val_loss)
            return scored, int(hits)

        jobs = int(self.n_jobs) if parallel_jobs is None else int(parallel_jobs)
        if jobs != 1 and len(population) >= 32 and float(self.tarpeian_rate) == 0:
            limit = 1 if self.thread_limit == "auto" else int(self.thread_limit)
            with threadpool_limits(limits=limit):
                output = Parallel(n_jobs=jobs, prefer="threads")(
                    delayed(evaluate_one)(i, p) for i, p in enumerate(population)
                )
        else:
            output = [evaluate_one(i, p) for i, p in enumerate(population)]
        return [item[0] for item in output], sum(item[1] for item in output)

    # ----------------------------- ranking -----------------------------
    def _complexity(self, program):
        return program.dag_size if str(self.complexity_measure).lower() == "dag" else program.size

    def _selection_value(self, program, profile=None):
        train = float(program.raw_fitness_)
        val = train if program.validation_fitness_ is None else float(program.validation_fitness_)
        if self.selection_metric == "training":
            base = train
        elif self.selection_metric == "validation":
            base = val
        else:
            base = val + float(self.validation_gap_penalty) * abs(val - train)
        coefficient = float(self._current_parsimony_coefficient)
        if str(profile).lower() == "compact":
            coefficient += max(1e-8, abs(base) * 0.002)
        return base + coefficient * self._complexity(program)

    def _dominates(self, a, b):
        af = float(a.selection_fitness_); bf = float(b.selection_fitness_)
        ac = self._complexity(a); bc = self._complexity(b)
        return (af <= bf and ac <= bc and a.depth <= b.depth) and (af < bf or ac < bc or a.depth < b.depth)

    def _assign_nsga2(self, population):
        domination_count = [0] * len(population)
        dominated = [[] for _ in population]
        fronts = [[]]
        for i, a in enumerate(population):
            for j, b in enumerate(population):
                if i == j:
                    continue
                if self._dominates(a, b):
                    dominated[i].append(j)
                elif self._dominates(b, a):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                a.metadata_["pareto_rank"] = 0
                fronts[0].append(i)
        rank = 0
        while rank < len(fronts) and fronts[rank]:
            next_front = []
            for i in fronts[rank]:
                for j in dominated[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].metadata_["pareto_rank"] = rank + 1
                        next_front.append(j)
            rank += 1
            fronts.append(next_front)
        fronts = [front for front in fronts if front]
        objectives = (
            lambda p: float(p.selection_fitness_),
            lambda p: float(self._complexity(p)),
            lambda p: float(p.depth),
        )
        for indices in fronts:
            for i in indices:
                population[i].metadata_["crowding_distance"] = 0.0
            if len(indices) <= 2:
                for i in indices:
                    population[i].metadata_["crowding_distance"] = float("inf")
                continue
            for key in objectives:
                ordered = sorted(indices, key=lambda idx: key(population[idx]))
                population[ordered[0]].metadata_["crowding_distance"] = float("inf")
                population[ordered[-1]].metadata_["crowding_distance"] = float("inf")
                lo, hi = key(population[ordered[0]]), key(population[ordered[-1]])
                if hi <= lo:
                    continue
                for k in range(1, len(ordered) - 1):
                    idx = ordered[k]
                    if not np.isinf(population[idx].metadata_["crowding_distance"]):
                        population[idx].metadata_["crowding_distance"] += (
                            key(population[ordered[k + 1]]) - key(population[ordered[k - 1]])
                        ) / (hi - lo)
        return fronts

    def _rank_population(self, population, profile=None):
        for program in population:
            program.selection_fitness_ = self._selection_value(program, profile=profile)
        if self.optimization in {"pareto", "nsga2"} and self.pareto_algorithm == "nsga2":
            self._assign_nsga2(population)
            return sorted(population, key=lambda p: (
                int(p.metadata_.get("pareto_rank", 10**9)),
                -float(p.metadata_.get("crowding_distance", 0.0)),
                p.selection_fitness_, self._complexity(p), p.depth,
            ))
        return sorted(population, key=lambda p: (p.selection_fitness_, self._complexity(p), p.depth))

    def _tournament(self, population, rng):
        indices = rng.integers(0, len(population), size=min(int(self.tournament_size), len(population)))
        candidates = [population[int(i)] for i in indices]
        if self.optimization in {"pareto", "nsga2"}:
            return min(candidates, key=lambda p: (
                int(p.metadata_.get("pareto_rank", 10**9)),
                -float(p.metadata_.get("crowding_distance", 0.0)),
                p.selection_fitness_, self._complexity(p),
            ))
        return min(candidates, key=lambda p: (p.selection_fitness_, self._complexity(p), p.depth))

    # ----------------------------- variation -----------------------------
    def _make_offspring(self, population, rng):
        operator = int(rng.choice(4, p=self._operator_rates))
        parent = self._tournament(population, rng)
        if operator == 0:
            child = subtree_crossover(parent, self._tournament(population, rng), rng)
        elif operator == 1:
            child = subtree_mutation(
                parent, rng, self.n_features_in_, self._function_set, self._mutation_depth,
                tuple(self.const_range), float(self.const_probability), self._function_weights,
            )
        elif operator == 2:
            child = hoist_mutation(parent, rng)
        else:
            child = point_mutation(
                parent, rng, self.n_features_in_, self._function_set,
                tuple(self.const_range), self._function_weights,
            )
        if not enforce_limits(child, int(self.max_depth), int(self.max_nodes)):
            if self.reject_oversized_offspring:
                child = parent.clone()
            else:
                child = simplify_program(child)
                if not enforce_limits(child, int(self.max_depth), int(self.max_nodes)):
                    child = parent.clone()
        child.metadata_["operator"] = operator
        return child

    def _next_generation(self, ranked, rng, X_semantic, target_size=None):
        target_size = int(self.population_size if target_size is None else target_size)
        elite_count = max(1, min(int(self.elitism), len(ranked), target_size))
        next_population = [p.clone() for p in ranked[:elite_count]]
        structural_seen = {p.structural_hash() for p in next_population}
        semantic_seen = set()
        if self.semantic_duplicate_elimination:
            for p in next_population:
                semantic_seen.add(semantic_hash(self._execute_program(p, X_semantic)))
        duplicate_retries = 0
        while len(next_population) < target_size:
            child = self._make_offspring(ranked, rng)
            structural = child.structural_hash()
            duplicate = self.duplicate_elimination and structural in structural_seen
            semantic = None
            if self.semantic_duplicate_elimination and not duplicate:
                semantic = semantic_hash(self._execute_program(child, X_semantic))
                duplicate = semantic in semantic_seen
            if duplicate and duplicate_retries < int(self.max_duplicate_attempts):
                duplicate_retries += 1
                continue
            duplicate_retries = 0
            next_population.append(child); structural_seen.add(structural)
            if semantic is not None:
                semantic_seen.add(semantic)
        return next_population

    def _adapt_operators(self, ranked):
        successes = np.ones(4, dtype=float)
        for survivor in ranked[: max(1, len(ranked) // 2)]:
            op = survivor.metadata_.get("operator")
            if op is not None:
                successes[int(op)] += 1.0
        rates = np.maximum(successes / successes.sum(), float(self.operator_min_probability))
        self._operator_rates = rates / rates.sum()

    def _adapt_functions(self, ranked):
        counts = {name: 1.0 for name in self._function_set}
        for program in ranked[: max(2, len(ranked) // 3)]:
            quality = 1.0 / max(1, program.metadata_.get("pareto_rank", 0) + 1)
            for name, amount in program.function_counts().items():
                counts[name] = counts.get(name, 1.0) + quality * amount
        total = sum(counts.values())
        mean = total / max(len(counts), 1)
        self._function_weights = {name: max(0.05, value / mean) for name, value in counts.items()}

    def _update_hall_of_fame(self, hall, population):
        unique = {}
        for program in list(hall) + list(population):
            key = program.structural_hash()
            previous = unique.get(key)
            if previous is None or program.selection_fitness_ < previous.selection_fitness_:
                unique[key] = program
        ranked = sorted(unique.values(), key=lambda p: (p.selection_fitness_, self._complexity(p), p.depth))
        return ranked[: int(self.hall_of_fame_size)]

    def _pareto_front(self, programs):
        front = []
        for candidate in programs:
            if not any(other is not candidate and self._dominates(other, candidate) for other in programs):
                front.append(candidate.clone())
        return sorted(front, key=lambda p: (p.selection_fitness_, self._complexity(p), p.depth))

    def _select_final(self, hall):
        if not hall:
            raise RuntimeError("Evolution produced an empty hall of fame")
        if self.final_selection == "raw":
            chosen = min(hall, key=lambda p: (p.raw_fitness_, self._complexity(p)))
        elif self.final_selection == "validation":
            chosen = min(hall, key=lambda p: (
                p.validation_fitness_ if p.validation_fitness_ is not None else p.raw_fitness_,
                self._complexity(p),
            ))
        elif self.final_selection == "pareto":
            chosen = min(self._pareto_front(hall), key=lambda p: (p.selection_fitness_, self._complexity(p), p.depth))
        else:
            best = min(float(p.selection_fitness_) for p in hall)
            tolerance = max(abs(best) * float(self.selection_tolerance), float(self.selection_tolerance) * 1e-3, 1e-12)
            eligible = [p for p in hall if p.selection_fitness_ <= best + tolerance]
            chosen = min(eligible, key=lambda p: (self._complexity(p), p.selection_fitness_, p.depth))
        return chosen.clone()

    # ----------------------------- dashboard -----------------------------
    def _colors_enabled(self):
        if self.use_color is True:
            return True
        if self.use_color is False:
            return False
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def _c(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self._colors_enabled() else text

    def _print_start_banner(self, X_train, X_val):
        if int(self.verbose) <= 0:
            return
        title = f" FASTSYMBOLICGP {__version__} // SYMBOLIC EVOLUTION GRID ONLINE "
        print("\n╔" + "═" * 104 + "╗")
        print("║" + self._c(title.center(104), "1;96") + "║")
        print("╠" + "═" * 104 + "╣")
        mode = "ISLAND NSGA-II" if str(self.evolution_model).lower() in {"islands", "island"} else (
            "NSGA-II PARETO" if self.optimization in {"pareto", "nsga2"} else "VALIDATION-AWARE SCALAR"
        )
        backend = str(getattr(self, "execution_engine_", self.evaluation_backend_)).upper()
        islands = max(1, int(self.n_islands)) if "ISLAND" in mode else 1
        print(f"║ MODE {mode:<24} │ BACKEND {backend:<7} │ ISLANDS {islands:>2} │ POP {int(self.population_size):>5} │ PRESET {str(getattr(self, 'preset_', 'custom')).upper():<14} ║")
        print(f"║ DATA train {X_train.shape[0]:>8} × {X_train.shape[1]:<5} │ validation {0 if X_val is None else X_val.shape[0]:>7} │ DAG {str(self.dag_execution).upper():<5} │ CACHE {float(self.subtree_cache_max_mb):>6.0f} MB       ║")
        budgets = f"time={self.time_budget or '∞'}s evals={self.evaluation_budget or '∞'} memory={self.memory_budget_mb or 'auto'}MB"
        print(f"║ LIMIT depth≤{int(self.max_depth):<3} nodes≤{int(self.max_nodes):<5} generations≤{int(self.generations):<5} patience={int(self.patience):<4} │ {budgets:<45} ║")
        print("╚" + "═" * 104 + "╝")

    def _format_progress(self, generation, best, elapsed, generation_time, diversity, cache_hits, stagnation, mean_nodes, pareto_size, population_size, migration_count=0):
        if int(self.verbose) <= 0:
            return
        interval = max(1, int(self.dashboard_interval))
        if generation != 1 and generation != int(self.generations) and generation % interval != 0:
            return
        progress = generation / int(self.generations)
        width = 34
        filled = min(width, int(round(progress * width)))
        bar = "█" * filled + "░" * (width - filled)
        val = best.validation_fitness_ if best.validation_fitness_ is not None else best.raw_fitness_
        eta = max(0.0, elapsed / max(generation, 1) * (int(self.generations) - generation))
        rates = self._operator_rates
        expr = best.to_string()
        if len(expr) > 74:
            expr = expr[:71] + "..."
        cache = self._subtree_cache_.statistics() if hasattr(self, "_subtree_cache_") else {"hit_rate": 0.0, "bytes_used": 0}
        dag = best.dag_stats()
        if self.display in {"dashboard", "live", "cool", "grid"}:
            print("╔" + "═" * 104 + "╗")
            print(f"║ {self._c('GEN', '1;95')} {generation:4d}/{int(self.generations):4d} {self._c(bar, '96')} {100*progress:6.2f}% │ elapsed {elapsed:7.1f}s │ ETA {eta:7.1f}s ║")
            print("╠" + "─" * 104 + "╣")
            print(f"║ FITNESS train {best.raw_fitness_:10.6f} │ valid {val:10.6f} │ selected {best.selection_fitness_:10.6f} │ Pareto {pareto_size:3d}       ║")
            print(f"║ MODEL tree {best.size:4d} │ DAG {dag['dag_nodes']:4d} │ compression {dag['compression_ratio']:6.1%} │ depth {best.depth:3d} │ mean nodes {mean_nodes:6.1f} ║")
            print(f"║ SEARCH pop {population_size:4d} │ diversity {diversity:6.1%} │ stagnation {stagnation:3d}/{int(self.patience):<3} │ migrations {migration_count:3d} │ gen {generation_time:6.2f}s  ║")
            print(f"║ CACHE hit {cache['hit_rate']:6.1%} │ generation hits {cache_hits:6d} │ memory {cache['bytes_used']/1024**2:7.1f} MB │ evaluations {int(getattr(self, 'evaluation_count_', 0)):9d}    ║")
            print(f"║ OPS C {rates[0]:5.1%} │ Sub {rates[1]:5.1%} │ Hoist {rates[2]:5.1%} │ Point {rates[3]:5.1%} │ robust {float(self.robustness_weight):5.1%}             ║")
            print(f"║ CHAMPION {expr:<93} ║")
            print("╚" + "═" * 104 + "╝")
        else:
            print(f"Gen {generation:4d}/{int(self.generations):4d} [{bar}] best={best.selection_fitness_:.6f} val={val:.6f} dag={dag['dag_nodes']:3d} div={diversity:5.1%} time={generation_time:6.2f}s")

    def _format_island_progress(self, generation, island_rows, elapsed, migration_count):
        if int(self.verbose) <= 0:
            return
        interval = max(1, int(self.dashboard_interval))
        if generation != 1 and generation != int(self.generations) and generation % interval != 0:
            return
        progress = generation / int(self.generations)
        width = 28
        bar = "█" * int(round(width * progress)) + "░" * (width - int(round(width * progress)))
        print("╔" + "═" * 104 + "╗")
        print(f"║ {self._c('ISLAND GRID', '1;95')} GEN {generation:4d}/{int(self.generations):4d} {self._c(bar, '96')} {100*progress:6.2f}% │ elapsed {elapsed:7.1f}s │ migrations {migration_count:3d} ║")
        print("╠" + "─" * 104 + "╣")
        print("║ ID  PROFILE       BEST VALID    DAG  DEPTH   DIVERSITY   STAGNATION   POP   STATUS                    ║")
        for row in island_rows[:8]:
            status = "EXPLORING" if row["diversity"] > 0.85 else "EVOLVING"
            print(f"║ {row['island']:>2}  {row['profile']:<12} {row['valid']:>11.6f} {row['dag']:>6d} {row['depth']:>6d} {row['diversity']:>10.1%} {row['stagnation']:>8d}/{int(self.patience):<3} {row['population']:>5d}   {status:<20} ║")
        cache = self._subtree_cache_.statistics()
        print("╠" + "─" * 104 + "╣")
        print(f"║ GLOBAL Pareto {len(self._pareto_front(getattr(self, 'hall_of_fame_', []))):3d} │ cache {cache['hit_rate']:6.1%} │ cache memory {cache['bytes_used']/1024**2:7.1f} MB │ evaluations {int(getattr(self, 'evaluation_count_', 0)):9d}              ║")
        print("╚" + "═" * 104 + "╝")

    def _print_final_banner(self):
        if int(self.verbose) <= 0:
            return
        dag = self.best_program_.dag_stats()
        cache = self._subtree_cache_.statistics() if hasattr(self, "_subtree_cache_") else {"hit_rate": 0.0, "bytes_used": 0}
        print("\n╔" + "═" * 104 + "╗")
        print("║" + self._c(" EVOLUTION COMPLETE // COMPACT SYMBOLIC SYSTEM DEPLOYABLE ".center(104), "1;92") + "║")
        print("╠" + "═" * 104 + "╣")
        print(f"║ generations {int(self.n_generations_):5d} │ runtime {float(self.run_time_seconds_):8.2f}s │ stop {str(getattr(self, 'stop_reason_', 'complete')):<18} │ Pareto {len(self.pareto_front_):4d}       ║")
        print(f"║ tree nodes {self.best_program_.size:4d} │ DAG nodes {dag['dag_nodes']:4d} │ compression {dag['compression_ratio']:6.1%} │ depth {self.best_program_.depth:3d} │ cache {cache['hit_rate']:6.1%}        ║")
        expr = self.best_expression_
        if len(expr) > 88:
            expr = expr[:85] + "..."
        print(f"║ expression {expr:<93} ║")
        print("╚" + "═" * 104 + "╝\n")

    # ----------------------------- checkpoint -----------------------------
    def _checkpoint_file(self):
        if not self.checkpoint_path:
            return None
        path = Path(self.checkpoint_path)
        return path if path.suffix else path / "checkpoint.fsgp-state"

    def _save_evolution_checkpoint(self, generation, population, hall, rng, stagnation, best_seen, islands=None):
        path = self._checkpoint_file()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": __version__, "estimator": self.__class__.__name__,
            "generation": int(generation), "population": population, "hall": hall,
            "islands": islands, "history": self.history_, "rng_state": rng.bit_generator.state,
            "stagnation": int(stagnation), "best_seen": float(best_seen),
            "parsimony": float(self._current_parsimony_coefficient),
            "operator_rates": self._operator_rates,
            "function_weights": self._function_weights,
            "evaluation_count": int(getattr(self, "evaluation_count_", 0)),
            "n_features_in": int(self.n_features_in_),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(payload, tmp)
        tmp.replace(path)
        self.last_checkpoint_path_ = str(path)

    def _load_evolution_checkpoint(self):
        path = self._checkpoint_file()
        if path is None or not path.exists():
            return None
        payload = joblib.load(path)
        if int(payload.get("n_features_in", -1)) != int(self.n_features_in_):
            raise ValueError("Checkpoint feature count does not match current data")
        return payload

    # ----------------------------- evolution utilities -----------------------------
    def _sample_training_data(self, X_train, y_train, sample_weight_train, rng):
        if isinstance(self.max_samples, float) and 0.0 < float(self.max_samples) < 1.0:
            sample_count = max(2, int(round(float(self.max_samples) * X_train.shape[0])))
        elif isinstance(self.max_samples, (int, np.integer)) and int(self.max_samples) > 1:
            sample_count = min(int(self.max_samples), X_train.shape[0])
        else:
            sample_count = X_train.shape[0]
        if sample_count < X_train.shape[0]:
            idx = rng.choice(X_train.shape[0], size=sample_count, replace=False)
            return X_train[idx], y_train[idx], None if sample_weight_train is None else sample_weight_train[idx], sample_count
        return X_train, y_train, sample_weight_train, sample_count

    def _update_parsimony(self, mean_nodes, fitnesses):
        if self.parsimony != "adaptive":
            return
        target = max(float(self.parsimony_target_nodes), 1.0)
        excess = max(0.0, (mean_nodes - target) / target)
        typical_loss = max(float(np.median(np.abs(fitnesses))), 1e-6)
        desired = excess * typical_loss * 0.05 / max(mean_nodes, 1.0)
        response = min(0.5, max(0.05, float(self.parsimony_growth_rate) - 1.0))
        self._current_parsimony_coefficient = (
            (1.0 - response) * self._current_parsimony_coefficient + response * desired
        )
        if mean_nodes <= target:
            self._current_parsimony_coefficient *= 0.9

    def _budget_exhausted(self, elapsed):
        if self.time_budget is not None and elapsed >= float(self.time_budget):
            self.stop_reason_ = "time_budget"
            return True
        if self.evaluation_budget is not None and int(self.evaluation_count_) >= int(self.evaluation_budget):
            self.stop_reason_ = "evaluation_budget"
            return True
        return False

    def _population_target(self, current, stagnation, diversity):
        if not self.adaptive_population:
            return current
        lower = int(self.population_min or max(8, int(self.population_size) // 2))
        upper = int(self.population_max or max(int(self.population_size), int(self.population_size) * 3))
        if stagnation >= max(3, int(self.patience) // 3) or diversity < 0.35:
            return min(upper, max(current + 2, int(np.ceil(current * 1.20))))
        if diversity > 0.90 and stagnation == 0:
            return max(lower, int(np.floor(current * 0.95)))
        return current

    def _finalize_evolution(self, ranked, hall, start):
        self._population = [] if self.low_memory else [p.clone() for p in ranked]
        self.hall_of_fame_ = hall
        self.pareto_front_ = self._pareto_front(hall)
        self.best_raw_program_ = self._select_final(hall)
        self.best_program_ = simplify_program(self.best_raw_program_) if self.simplify_expression else self.best_raw_program_.clone()
        self.best_expression_ = self.best_program_.to_string(getattr(self, "feature_names_in_", None))
        self.n_generations_ = len({row.get("generation") for row in self.history_ if "generation" in row})
        self.run_time_seconds_ = perf_counter() - start
        self.cache_statistics_ = self._subtree_cache_.statistics()
        self.dag_statistics_ = self.best_program_.dag_stats()
        self.feature_usage_ = self._build_feature_usage()
        self.function_usage_ = self.best_program_.function_counts()
        self._print_final_banner()
        return self

    def _build_feature_usage(self):
        counts = self.best_program_.feature_counts()
        total = max(sum(counts.values()), 1)
        names = getattr(self, "feature_names_in_", None)
        return {
            (str(names[index]) if names is not None else f"X{index}"): {
                "count": int(count), "fraction": float(count / total), "index": int(index),
            }
            for index, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        }

    # ----------------------------- panmictic evolution -----------------------------
    def _evolve_panmictic(self, X_train, y_train, X_val, y_val, sw_train, sw_val, checkpoint):
        rng = np.random.default_rng(self.random_state)
        if checkpoint is not None:
            population = [p.clone() for p in checkpoint["population"]]
            hall = [p.clone() for p in checkpoint["hall"]]
            self.history_ = list(checkpoint.get("history", []))
            rng.bit_generator.state = checkpoint["rng_state"]
            start_generation = int(checkpoint["generation"]) + 1
            stagnation = int(checkpoint.get("stagnation", 0))
            best_seen = float(checkpoint.get("best_seen", np.inf))
            self._current_parsimony_coefficient = float(checkpoint.get("parsimony", self.parsimony_coefficient))
            self._operator_rates = np.asarray(checkpoint.get("operator_rates", self._operator_rates), dtype=float)
            self._function_weights = dict(checkpoint.get("function_weights", self._function_weights))
            self.evaluation_count_ = int(checkpoint.get("evaluation_count", 0))
        elif self.warm_start and hasattr(self, "_population") and self._population:
            population = [p.clone() for p in self._population]
            hall = [p.clone() for p in getattr(self, "hall_of_fame_", [])]
            self.history_ = list(getattr(self, "history_", []))
            start_generation = max([r.get("generation", 0) for r in self.history_] or [0]) + 1
            stagnation = 0
            best_seen = min((p.selection_fitness_ for p in hall), default=np.inf)
            self._current_parsimony_coefficient = float(getattr(self, "_current_parsimony_coefficient", self.parsimony_coefficient))
            self.evaluation_count_ = int(getattr(self, "evaluation_count_", 0))
        else:
            population = self._initial_population(rng, self.n_features_in_)
            hall, self.history_ = [], []
            start_generation, stagnation, best_seen = 1, 0, np.inf
            self._current_parsimony_coefficient = float(self.parsimony_coefficient)
            self.evaluation_count_ = 0

        semantic_count = min(int(self.semantic_sample_size), X_train.shape[0])
        X_semantic = X_train[rng.choice(X_train.shape[0], semantic_count, replace=False)]
        start = perf_counter()
        self.stop_reason_ = "max_generations"
        self._print_start_banner(X_train, X_val)
        ranked = self._rank_population(population) if start_generation > int(self.generations) else population
        target_population = len(population)

        for generation in range(start_generation, int(self.generations) + 1):
            generation_start = perf_counter()
            if str(self.subtree_cache_scope).lower() == "generation":
                self._subtree_cache_.clear()
            X_fit, y_fit, sw_fit, sample_count = self._sample_training_data(X_train, y_train, sw_train, rng)
            population, cache_hits = self._evaluate_population(population, X_fit, y_fit, X_val, y_val, sw_fit, sw_val, rng)
            self.evaluation_count_ += len(population) * (2 if self.robustness_training and float(self.robustness_weight) > 0 else 1)
            ranked = self._rank_population(population)
            if self.operator_adaptation and generation > 1 and generation % int(self.operator_adaptation_interval) == 0:
                self._adapt_operators(ranked)
            if self.function_set_adaptation and generation > 1 and generation % int(self.function_adaptation_interval) == 0:
                self._adapt_functions(ranked)

            best = min(ranked, key=lambda p: (p.selection_fitness_, self._complexity(p), p.depth))
            hall = self._update_hall_of_fame(hall, ranked)
            fitnesses = np.asarray([p.selection_fitness_ for p in ranked], dtype=float)
            diversity = len({p.structural_hash() for p in ranked}) / len(ranked)
            mean_nodes = float(np.mean([p.size for p in ranked]))
            generation_time = perf_counter() - generation_start
            elapsed = perf_counter() - start
            val_best = best.validation_fitness_ if best.validation_fitness_ is not None else best.raw_fitness_
            pareto_size = len(self._pareto_front(hall))
            row = {
                "generation": int(generation),
                "best_selection_fitness": float(best.selection_fitness_),
                "best_training_fitness": float(best.raw_fitness_),
                "best_validation_fitness": float(val_best),
                "mean_selection_fitness": float(np.mean(fitnesses)),
                "median_selection_fitness": float(np.median(fitnesses)),
                "best_nodes": int(best.size), "best_dag_nodes": int(best.dag_size),
                "dag_compression_ratio": float(best.dag_compression_ratio),
                "mean_nodes": mean_nodes, "best_depth": int(best.depth),
                "diversity": float(diversity), "pareto_size": int(pareto_size),
                "cache_hits": int(cache_hits), "cache_hit_rate": float(self._subtree_cache_.statistics()["hit_rate"]),
                "training_samples_used": int(sample_count), "population_size": int(len(population)),
                "evaluation_count": int(self.evaluation_count_),
                "parsimony_coefficient": float(self._current_parsimony_coefficient),
                "generation_time_seconds": float(generation_time),
                "elapsed_seconds": float(elapsed), "stagnation": int(stagnation),
                "evaluation_backend": self.evaluation_backend_,
                "execution_engine": self.execution_engine_, "evolution_model": "panmictic",
                "p_crossover_effective": float(self._operator_rates[0]),
                "p_subtree_mutation_effective": float(self._operator_rates[1]),
                "p_hoist_mutation_effective": float(self._operator_rates[2]),
                "p_point_mutation_effective": float(self._operator_rates[3]),
            }
            self.history_.append(row)
            self._format_progress(generation, best, elapsed, generation_time, diversity, cache_hits, stagnation, mean_nodes, pareto_size, len(population))

            if best.selection_fitness_ < best_seen - float(self.min_delta):
                best_seen, stagnation = float(best.selection_fitness_), 0
            else:
                stagnation += 1
            self._update_parsimony(mean_nodes, fitnesses)
            target_population = self._population_target(target_population, stagnation, diversity)

            if int(self.checkpoint_interval) > 0 and generation % int(self.checkpoint_interval) == 0:
                self._save_evolution_checkpoint(generation, ranked, hall, rng, stagnation, best_seen)
            if self._budget_exhausted(elapsed):
                break
            if int(self.patience) > 0 and stagnation >= int(self.patience):
                self.stop_reason_ = "early_stopping"
                if int(self.verbose) > 0:
                    print(self._c(f"⚡ Early stop: no improvement for {stagnation} generations.", "1;93"))
                break
            if generation < int(self.generations):
                population = self._next_generation(ranked, rng, X_semantic, target_size=target_population)

        return self._finalize_evolution(ranked, hall, start)

    # ----------------------------- island evolution -----------------------------
    def _island_profile_list(self, count):
        default = ("accurate", "compact", "diverse", "robust")
        if self.island_profiles is None:
            return [default[i % len(default)] for i in range(count)]
        values = list(self.island_profiles)
        if not values:
            values = list(default)
        return [str(values[i % len(values)]) for i in range(count)]

    def _migrate(self, islands, ranked_islands, rng):
        count = len(islands)
        if count <= 1 or int(self.migration_size) <= 0:
            return 0
        size = min(int(self.migration_size), min(len(pop) for pop in islands))
        strategy = str(self.migration_strategy).lower()
        migrants = [[p.clone() for p in ranked[:size]] for ranked in ranked_islands]
        destinations = list(range(count))
        if strategy == "ring":
            destinations = [(i + 1) % count for i in range(count)]
        elif strategy == "random":
            destinations = list(rng.permutation(count))
            if any(i == d for i, d in enumerate(destinations)) and count > 1:
                destinations = destinations[1:] + destinations[:1]
        elif strategy == "best_to_worst":
            quality = [ranked[0].selection_fitness_ for ranked in ranked_islands]
            source_order = np.argsort(quality)
            dest_order = np.argsort(quality)[::-1]
            destinations = [0] * count
            for src, dst in zip(source_order, dest_order):
                destinations[int(src)] = int(dst)
        else:  # diversity_guided
            diversity = [len({p.structural_hash() for p in pop}) / len(pop) for pop in islands]
            destinations = list(np.argsort(diversity))
        for source, destination in enumerate(destinations):
            destination = int(destination)
            target_ranked = ranked_islands[destination]
            keep = [p.clone() for p in target_ranked[:-size]] if len(target_ranked) > size else []
            islands[destination] = keep + [p.clone() for p in migrants[source]]
        return count * size

    def _evolve_islands(self, X_train, y_train, X_val, y_val, sw_train, sw_val, checkpoint):
        n_islands = max(2, int(self.n_islands))
        root_rng = np.random.default_rng(self.random_state)
        profiles = self._island_profile_list(n_islands)
        per_island = max(6, int(np.ceil(int(self.population_size) / n_islands)))
        island_rngs = [np.random.default_rng(int(root_rng.integers(0, 2**31 - 1))) for _ in range(n_islands)]
        if checkpoint is not None and checkpoint.get("islands"):
            islands = [[p.clone() for p in pop] for pop in checkpoint["islands"]]
            hall = [p.clone() for p in checkpoint["hall"]]
            self.history_ = list(checkpoint.get("history", []))
            start_generation = int(checkpoint["generation"]) + 1
            stagnation = int(checkpoint.get("stagnation", 0))
            best_seen = float(checkpoint.get("best_seen", np.inf))
            self.evaluation_count_ = int(checkpoint.get("evaluation_count", 0))
            self._current_parsimony_coefficient = float(checkpoint.get("parsimony", self.parsimony_coefficient))
            self._operator_rates = np.asarray(checkpoint.get("operator_rates", self._operator_rates), dtype=float)
            self._function_weights = dict(checkpoint.get("function_weights", self._function_weights))
        else:
            islands = [self._initial_population(island_rngs[i], self.n_features_in_, per_island) for i in range(n_islands)]
            hall, self.history_ = [], []
            start_generation, stagnation, best_seen = 1, 0, np.inf
            self.evaluation_count_ = 0
            self._current_parsimony_coefficient = float(self.parsimony_coefficient)
        semantic_count = min(int(self.semantic_sample_size), X_train.shape[0])
        X_semantic = X_train[root_rng.choice(X_train.shape[0], semantic_count, replace=False)]
        start = perf_counter()
        self.stop_reason_ = "max_generations"
        self.migration_count_ = 0
        self.island_history_ = []
        self._print_start_banner(X_train, X_val)
        ranked_islands = islands

        for generation in range(start_generation, int(self.generations) + 1):
            generation_start = perf_counter()
            if str(self.subtree_cache_scope).lower() == "generation":
                self._subtree_cache_.clear()
            X_fit, y_fit, sw_fit, sample_count = self._sample_training_data(X_train, y_train, sw_train, root_rng)

            def evaluate_island(i):
                evaluated, hits = self._evaluate_population(
                    islands[i], X_fit, y_fit, X_val, y_val, sw_fit, sw_val,
                    island_rngs[i], parallel_jobs=1,
                )
                return self._rank_population(evaluated, profile=profiles[i]), hits

            outer_jobs = int(self.n_jobs) if self.island_parallel and int(self.n_jobs) != 1 else 1
            if outer_jobs != 1:
                results = Parallel(n_jobs=min(abs(outer_jobs), n_islands), prefer="threads") (
                    delayed(evaluate_island)(i) for i in range(n_islands)
                )
            else:
                results = [evaluate_island(i) for i in range(n_islands)]
            ranked_islands = [result[0] for result in results]
            cache_hits = sum(result[1] for result in results)
            self.evaluation_count_ += sum(len(pop) for pop in islands) * (2 if self.robustness_training and float(self.robustness_weight) > 0 else 1)

            if self.operator_adaptation and generation > 1 and generation % int(self.operator_adaptation_interval) == 0:
                self._adapt_operators([p for ranked in ranked_islands for p in ranked])
            if self.function_set_adaptation and generation > 1 and generation % int(self.function_adaptation_interval) == 0:
                self._adapt_functions([p for ranked in ranked_islands for p in ranked])

            island_rows = []
            all_ranked = []
            for i, ranked in enumerate(ranked_islands):
                all_ranked.extend(ranked)
                best_i = min(ranked, key=lambda p: (p.selection_fitness_, self._complexity(p), p.depth))
                diversity_i = len({p.structural_hash() for p in ranked}) / len(ranked)
                val_i = best_i.validation_fitness_ if best_i.validation_fitness_ is not None else best_i.raw_fitness_
                island_rows.append({
                    "generation": generation, "island": i + 1, "profile": profiles[i],
                    "valid": float(val_i), "fitness": float(best_i.selection_fitness_),
                    "nodes": int(best_i.size), "dag": int(best_i.dag_size), "depth": int(best_i.depth),
                    "diversity": float(diversity_i), "population": len(ranked), "stagnation": stagnation,
                })
            hall = self._update_hall_of_fame(hall, all_ranked)
            self.hall_of_fame_ = hall
            global_ranked = self._rank_population(all_ranked)
            best = min(global_ranked, key=lambda p: (p.selection_fitness_, self._complexity(p), p.depth))
            fitnesses = np.asarray([p.selection_fitness_ for p in global_ranked])
            diversity = len({p.structural_hash() for p in global_ranked}) / len(global_ranked)
            mean_nodes = float(np.mean([p.size for p in global_ranked]))
            elapsed = perf_counter() - start
            generation_time = perf_counter() - generation_start
            val_best = best.validation_fitness_ if best.validation_fitness_ is not None else best.raw_fitness_
            pareto_size = len(self._pareto_front(hall))
            row = {
                "generation": generation, "best_selection_fitness": float(best.selection_fitness_),
                "best_training_fitness": float(best.raw_fitness_), "best_validation_fitness": float(val_best),
                "mean_selection_fitness": float(np.mean(fitnesses)), "median_selection_fitness": float(np.median(fitnesses)),
                "best_nodes": int(best.size), "best_dag_nodes": int(best.dag_size),
                "dag_compression_ratio": float(best.dag_compression_ratio), "mean_nodes": mean_nodes,
                "best_depth": int(best.depth), "diversity": float(diversity), "pareto_size": pareto_size,
                "cache_hits": int(cache_hits), "cache_hit_rate": float(self._subtree_cache_.statistics()["hit_rate"]),
                "training_samples_used": int(sample_count), "population_size": int(len(global_ranked)),
                "evaluation_count": int(self.evaluation_count_), "n_islands": n_islands,
                "migrations": int(self.migration_count_), "generation_time_seconds": float(generation_time),
                "elapsed_seconds": float(elapsed), "stagnation": int(stagnation),
                "evaluation_backend": self.evaluation_backend_,
                "execution_engine": self.execution_engine_, "evolution_model": "islands",
            }
            self.history_.append(row)
            self.island_history_.extend(island_rows)
            self._format_island_progress(generation, island_rows, elapsed, self.migration_count_)

            if best.selection_fitness_ < best_seen - float(self.min_delta):
                best_seen, stagnation = float(best.selection_fitness_), 0
            else:
                stagnation += 1
            self._update_parsimony(mean_nodes, fitnesses)

            if int(self.migration_interval) > 0 and generation % int(self.migration_interval) == 0 and generation < int(self.generations):
                self.migration_count_ += self._migrate(islands, ranked_islands, root_rng)
                ranked_islands = [self._rank_population(pop, profile=profiles[i]) for i, pop in enumerate(islands)]
            if int(self.checkpoint_interval) > 0 and generation % int(self.checkpoint_interval) == 0:
                self._save_evolution_checkpoint(generation, global_ranked, hall, root_rng, stagnation, best_seen, islands=islands)
            if self._budget_exhausted(elapsed):
                break
            if int(self.patience) > 0 and stagnation >= int(self.patience):
                self.stop_reason_ = "early_stopping"
                break
            if generation < int(self.generations):
                next_islands = []
                for i, ranked in enumerate(ranked_islands):
                    island_diversity = len({p.structural_hash() for p in ranked}) / len(ranked)
                    target = self._population_target(len(ranked), stagnation, island_diversity)
                    next_islands.append(self._next_generation(ranked, island_rngs[i], X_semantic, target_size=target))
                islands = next_islands

        self.island_populations_ = [] if self.low_memory else [[p.clone() for p in ranked] for ranked in ranked_islands]
        global_ranked = self._rank_population([p for ranked in ranked_islands for p in ranked])
        return self._finalize_evolution(global_ranked, hall, start)

    # ----------------------------- entry point -----------------------------
    def _evolve(self, X_train, y_train, X_val, y_val, sample_weight_train, sample_weight_val):
        self._validate_common_parameters()
        checkpoint = self._load_evolution_checkpoint() if self.resume_from_checkpoint else None
        if str(self.subtree_cache_scope).lower() == "generation" and hasattr(self, "_subtree_cache_"):
            self._subtree_cache_.clear()
        if str(self.evolution_model).lower() in {"islands", "island"} or int(self.n_islands) > 1:
            return self._evolve_islands(X_train, y_train, X_val, y_val, sample_weight_train, sample_weight_val, checkpoint)
        return self._evolve_panmictic(X_train, y_train, X_val, y_val, sample_weight_train, sample_weight_val, checkpoint)

    # ----------------------------- public helpers -----------------------------
    def raw_score(self, X):
        check_is_fitted(self, "best_program_")
        return self._execute_program(self.best_program_, self._prepare_X_predict(X))

    def score_samples(self, X):
        return self.raw_score(X)

    def get_expression(self):
        check_is_fitted(self, "best_program_")
        return self.best_program_.to_string(getattr(self, "feature_names_in_", None))

    def get_raw_expression(self):
        check_is_fitted(self, "best_program_")
        return self.best_raw_program_.to_string(getattr(self, "feature_names_in_", None))

    def get_simplified_expression(self):
        check_is_fitted(self, "best_program_")
        return simplify_program(self.best_program_).to_string(getattr(self, "feature_names_in_", None))

    def get_expression_stats(self):
        check_is_fitted(self, "best_program_")
        simplified = simplify_program(self.best_program_)
        result = {
            "nodes": self.best_program_.size, "depth": self.best_program_.depth,
            "simplified_nodes": simplified.size, "simplified_depth": simplified.depth,
            "pareto_models": len(getattr(self, "pareto_front_", [])),
            "expression": self.get_expression(),
        }
        result.update(self.best_program_.dag_stats())
        return result

    def select_from_pareto(self, max_nodes=None, max_depth=None, max_loss=None):
        check_is_fitted(self, "pareto_front_")
        candidates = list(self.pareto_front_)
        if max_nodes is not None:
            candidates = [p for p in candidates if self._complexity(p) <= int(max_nodes)]
        if max_depth is not None:
            candidates = [p for p in candidates if p.depth <= int(max_depth)]
        if max_loss is not None:
            candidates = [p for p in candidates if p.selection_fitness_ <= float(max_loss)]
        if not candidates:
            raise ValueError("No Pareto model satisfies the requested constraints")
        self.best_raw_program_ = min(candidates, key=lambda p: (p.selection_fitness_, self._complexity(p), p.depth)).clone()
        self.best_program_ = simplify_program(self.best_raw_program_) if self.simplify_expression else self.best_raw_program_.clone()
        self.best_expression_ = self.get_expression()
        self.dag_statistics_ = self.best_program_.dag_stats()
        return self

    def continue_evolution(self, X, y, additional_generations=20, sample_weight=None):
        self.set_params(generations=int(getattr(self, "n_generations_", 0)) + int(additional_generations), warm_start=True)
        return self.fit(X, y, sample_weight=sample_weight)

    def branch(self, **parameter_changes):
        """Create an independent warm-start branch from the current state."""
        check_is_fitted(self, "best_program_")
        branch = copy.deepcopy(self)
        branch.set_params(**parameter_changes)
        branch.warm_start = True
        branch.resume_from_checkpoint = False
        return branch

    def save_checkpoint(self, path):
        check_is_fitted(self, "best_program_")
        joblib.dump(self, path)

    @classmethod
    def load_checkpoint(cls, path):
        return cls.load_model(path)

    def save_history_csv(self, path):
        save_history_csv(self.history_, path)

    def save_report_json(self, path):
        report = {
            "version": __version__, "estimator": self.__class__.__name__,
            "parameters": self.get_params(deep=False), "n_features_in": int(self.n_features_in_),
            "n_generations": int(self.n_generations_), "run_time_seconds": float(self.run_time_seconds_),
            "stop_reason": getattr(self, "stop_reason_", "complete"),
            "evaluation_backend": getattr(self, "evaluation_backend_", self.evaluation_backend),
            "execution_engine": getattr(self, "execution_engine_", "unknown"),
            "evolution_model": self.evolution_model, "evaluation_count": int(getattr(self, "evaluation_count_", 0)),
            "expression": self.get_expression(), "expression_stats": self.get_expression_stats(),
            "cache_statistics": getattr(self, "cache_statistics_", {}),
            "feature_usage": getattr(self, "feature_usage_", {}),
            "function_usage": getattr(self, "function_usage_", {}),
            "pareto_front": [
                {"loss": p.selection_fitness_, "nodes": p.size, "dag_nodes": p.dag_size, "depth": p.depth, "expression": p.to_string()}
                for p in self.pareto_front_
            ],
            "history": self.history_,
            "island_history": getattr(self, "island_history_", []),
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "numba_available": NUMBA_AVAILABLE,
            },
        }
        save_json(report, path)

    def save_expression(self, path):
        Path(path).write_text(self.get_expression() + "\n", encoding="utf-8")

    def save_model(self, path):
        joblib.dump(self, path)

    @classmethod
    def load_model(cls, path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Stored object is {type(model).__name__}, expected {cls.__name__}")
        return model

    def profile_prediction(self, X, repeats=100, warmup=5):
        check_is_fitted(self, "best_program_")
        X = self._prepare_X_predict(X)
        for _ in range(max(0, int(warmup))):
            self.predict(X) if hasattr(self, "predict") else self.raw_score(X)
        tracemalloc.start()
        start = perf_counter()
        for _ in range(max(1, int(repeats))):
            self.predict(X) if hasattr(self, "predict") else self.raw_score(X)
        elapsed = perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        calls = max(1, int(repeats))
        samples = max(1, X.shape[0] * calls)
        return {
            "repeats": calls,
            "batch_size": int(X.shape[0]),
            "seconds_total": float(elapsed),
            "milliseconds_per_batch": float(elapsed * 1000 / calls),
            "microseconds_per_sample": float(elapsed * 1e6 / samples),
            "peak_working_memory_bytes": int(peak),
            "tree_operations": int(self.best_program_.size),
            "dag_operations": int(self.best_program_.dag_size),
            "dag_compression_ratio": float(self.best_program_.dag_compression_ratio),
        }

    def export_python(self, path):
        from .exporter import export_model
        return export_model(self, path, language="python")

    def export_c(self, path):
        from .exporter import export_model
        return export_model(self, path, language="c")

    def export_cpp(self, path):
        from .exporter import export_model
        return export_model(self, path, language="cpp")

    def export_java(self, path):
        from .exporter import export_model
        return export_model(self, path, language="java")

    def export_kotlin(self, path):
        from .exporter import export_model
        return export_model(self, path, language="kotlin")

    def export_javascript(self, path):
        from .exporter import export_model
        return export_model(self, path, language="javascript")
