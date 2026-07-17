"""Symbolic regressor for FastSymbolicGP V0.7.0."""
from __future__ import annotations

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .base import BaseFastSymbolicGP
from .metrics import fit_linear_scaler, regression_loss
from .validation import check_X_y_finite


class FastSymbolicRegressor(RegressorMixin, BaseFastSymbolicGP):
    def __init__(
        self,
        population_size=500,
        generations=50,
        tournament_size=20,
        function_set=("add", "sub", "mul", "div"),
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
        fitness_metric="rmse",
        metric=None,
        fitness_scaling="linear",
        validation_fraction=0.20,
        validation_metric=None,
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
        super().__init__(
            population_size=population_size, generations=generations,
            tournament_size=tournament_size, function_set=function_set,
            init_depth=init_depth, max_depth=max_depth, max_nodes=max_nodes,
            p_crossover=p_crossover, p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation, p_point_mutation=p_point_mutation,
            crossover_rate=crossover_rate, subtree_mutation_rate=subtree_mutation_rate,
            hoist_mutation_rate=hoist_mutation_rate, point_mutation_rate=point_mutation_rate,
            mutation_depth=mutation_depth, const_range=const_range,
            const_probability=const_probability, max_samples=max_samples,
            parsimony_coefficient=parsimony_coefficient, parsimony=parsimony,
            parsimony_target_nodes=parsimony_target_nodes,
            parsimony_growth_rate=parsimony_growth_rate,
            validation_fraction=validation_fraction, selection_metric=selection_metric,
            validation_gap_penalty=validation_gap_penalty,
            final_selection=final_selection, selection_tolerance=selection_tolerance,
            optimization=optimization, pareto_algorithm=pareto_algorithm,
            patience=patience, min_delta=min_delta, elitism=elitism,
            hall_of_fame_size=hall_of_fame_size,
            duplicate_elimination=duplicate_elimination,
            semantic_duplicate_elimination=semantic_duplicate_elimination,
            semantic_sample_size=semantic_sample_size,
            max_duplicate_attempts=max_duplicate_attempts,
            simplify_expression=simplify_expression,
            operator_adaptation=operator_adaptation,
            operator_adaptation_interval=operator_adaptation_interval,
            operator_min_probability=operator_min_probability,
            reject_oversized_offspring=reject_oversized_offspring,
            tarpeian_rate=tarpeian_rate,
            evaluation_cache=evaluation_cache, subtree_cache=subtree_cache,
            subtree_cache_scope=subtree_cache_scope,
            subtree_cache_max_mb=subtree_cache_max_mb,
            dag_execution=dag_execution, complexity_measure=complexity_measure,
            evaluation_backend=evaluation_backend, batch_size=batch_size,
            n_jobs=n_jobs, thread_limit=thread_limit,
            evolution_model=evolution_model, n_islands=n_islands,
            migration_interval=migration_interval, migration_size=migration_size,
            migration_strategy=migration_strategy, island_profiles=island_profiles,
            island_parallel=island_parallel, adaptive_population=adaptive_population,
            population_min=population_min, population_max=population_max,
            function_set_adaptation=function_set_adaptation,
            function_adaptation_interval=function_adaptation_interval,
            time_budget=time_budget, evaluation_budget=evaluation_budget,
            memory_budget_mb=memory_budget_mb,
            missing_value_strategy=missing_value_strategy,
            missing_value_constant=missing_value_constant,
            robustness_training=robustness_training,
            robustness_method=robustness_method,
            robustness_weight=robustness_weight,
            robustness_noise=robustness_noise,
            feature_dropout_rate=feature_dropout_rate,
            preset=preset, checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            resume_from_checkpoint=resume_from_checkpoint,
            random_state=random_state, verbose=verbose, display=display,
            dashboard_interval=dashboard_interval, use_color=use_color,
            warm_start=warm_start, low_memory=low_memory, **kwargs,
        )
        self.fitness_metric = fitness_metric
        self.metric = metric
        self.fitness_scaling = fitness_scaling
        self.validation_metric = validation_metric

    def _score_program(self, program, raw_train, y_train, raw_val, y_val, sw_train, sw_val):
        metric = self.fitness_metric if self.metric is None else self.metric
        if self.fitness_scaling in {"linear", "auto"}:
            scale, intercept = fit_linear_scaler(raw_train, y_train, sw_train)
        elif self.fitness_scaling in {None, "none", "identity"}:
            scale, intercept = 1.0, 0.0
        else:
            raise ValueError("fitness_scaling must be linear or none")
        program.scale_, program.intercept_ = scale, intercept
        program.raw_fitness_ = regression_loss(y_train, scale * raw_train + intercept, metric, sw_train)
        if raw_val is not None:
            val_metric = metric if self.validation_metric is None else self.validation_metric
            program.validation_fitness_ = regression_loss(y_val, scale * raw_val + intercept, val_metric, sw_val)
        else:
            program.validation_fitness_ = program.raw_fitness_
        return program

    def _robustness_loss(self, program, raw, y, sample_weight):
        metric = self.fitness_metric if self.metric is None else self.metric
        return regression_loss(y, program.scale_ * raw + program.intercept_, metric, sample_weight)

    def fit(self, X, y, sample_weight=None):
        feature_names = getattr(X, "columns", None)
        X, y = check_X_y_finite(
            X, y, dtype=np.float64, ensure_2d=True, y_numeric=True,
            allow_nan=self._allow_nan(),
        )
        self._apply_preset(X, y)
        X = self._prepare_X_fit(X)
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if float(self.validation_fraction) > 0:
            idx = np.arange(X.shape[0])
            train_idx, val_idx = train_test_split(
                idx, test_size=float(self.validation_fraction), random_state=self.random_state,
            )
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            sw_train = None if weights is None else weights[train_idx]
            sw_val = None if weights is None else weights[val_idx]
        else:
            X_train, y_train, sw_train = X, y, weights
            X_val = y_val = sw_val = None
        self._evolve(
            np.ascontiguousarray(X_train), y_train,
            None if X_val is None else np.ascontiguousarray(X_val), y_val,
            sw_train, sw_val,
        )
        raw_train = self._execute_program(self.best_program_, X_train)
        self.scale_, self.intercept_ = fit_linear_scaler(raw_train, y_train, sw_train)
        self.best_program_.scale_, self.best_program_.intercept_ = self.scale_, self.intercept_
        return self

    def predict(self, X):
        return self.scale_ * self.raw_score(X) + self.intercept_

    def score(self, X, y, sample_weight=None):
        return r2_score(y, self.predict(X), sample_weight=sample_weight)
