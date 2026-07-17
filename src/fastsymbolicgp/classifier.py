"""Symbolic binary and multiclass classifiers for FastSymbolicGP V0.7.0."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import tracemalloc
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from .base import BaseFastSymbolicGP
from .calibration import (
    TemperatureScaler, apply_calibrator, expected_calibration_error,
    fit_calibrator, multiclass_brier_score,
)
from .history import save_json
from .metrics import (
    classification_loss, fit_logistic_scaler, optimize_threshold,
    probability_from_raw, resolve_class_weight,
)
from .validation import check_X_y_finite


def _project_simplex(v):
    v = np.asarray(v, dtype=float)
    if np.all(v >= 0) and abs(v.sum() - 1.0) < 1e-12:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


class FastSymbolicClassifier(ClassifierMixin, BaseFastSymbolicGP):
    """Fast symbolic binary and multiclass classifier.

    Multiclass strategies
    ---------------------
    ``ovr`` evolves one calibrated symbolic decision system per class.
    ``shared_softmax`` selects a common symbolic feature backbone from all
    class-specific Pareto elites and trains a joint multinomial softmax head.
    """

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
        fitness_metric="log_loss",
        metric=None,
        link_function="sigmoid",
        fitness_scaling="elite_logistic",
        fitness_scaling_top_k=12,
        fitness_scaling_interval=1,
        fitness_scaling_regularization=1e-4,
        fitness_scaling_warm_start=True,
        fitness_scaling_fallback="rank",
        class_weight=None,
        validation_fraction=0.20,
        validation_metric=None,
        selection_metric="validation",
        validation_gap_penalty=0.25,
        final_selection="smallest_within_tolerance",
        selection_tolerance=0.02,
        optimization="scalar",
        pareto_algorithm="nsga2",
        probability_calibration="auto",
        calibration_cv=3,
        probability_clip=1e-6,
        threshold_strategy="mcc",
        threshold_validation_fraction=None,
        multiclass_strategy="ovr",
        multiclass_calibration="temperature",
        shared_n_components=12,
        shared_max_correlation=0.95,
        shared_regularization=1.0,
        prediction_mode="best",
        ensemble_size=5,
        ensemble_weighting="optimized",
        max_ensemble_correlation=0.97,
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
        self.link_function = link_function
        self.fitness_scaling = fitness_scaling
        self.fitness_scaling_top_k = fitness_scaling_top_k
        self.fitness_scaling_interval = fitness_scaling_interval
        self.fitness_scaling_regularization = fitness_scaling_regularization
        self.fitness_scaling_warm_start = fitness_scaling_warm_start
        self.fitness_scaling_fallback = fitness_scaling_fallback
        self.class_weight = class_weight
        self.validation_metric = validation_metric
        self.probability_calibration = probability_calibration
        self.calibration_cv = calibration_cv
        self.probability_clip = probability_clip
        self.threshold_strategy = threshold_strategy
        self.threshold_validation_fraction = threshold_validation_fraction
        self.multiclass_strategy = multiclass_strategy
        self.multiclass_calibration = multiclass_calibration
        self.shared_n_components = shared_n_components
        self.shared_max_correlation = shared_max_correlation
        self.shared_regularization = shared_regularization
        self.prediction_mode = prediction_mode
        self.ensemble_size = ensemble_size
        self.ensemble_weighting = ensemble_weighting
        self.max_ensemble_correlation = max_ensemble_correlation

    # ----------------------------- fitness -----------------------------
    def _score_program(self, program, raw_train, y_train, raw_val, y_val, sw_train, sw_val):
        metric = self.fitness_metric if self.metric is None else self.metric
        if self.fitness_scaling in {"elite_logistic", "logistic", "auto"}:
            scale, intercept = fit_logistic_scaler(
                raw_train, y_train, sw_train,
                regularization=float(self.fitness_scaling_regularization), max_iter=15,
            )
        elif self.fitness_scaling in {None, "none", "identity"}:
            scale, intercept = 1.0, 0.0
        else:
            raise ValueError(f"Unsupported fitness_scaling: {self.fitness_scaling}")
        program.scale_, program.intercept_ = scale, intercept
        train_proba = probability_from_raw(raw_train, scale, intercept)
        program.raw_fitness_ = classification_loss(y_train, train_proba, metric, sw_train)
        if raw_val is not None:
            val_proba = probability_from_raw(raw_val, scale, intercept)
            val_metric = metric if self.validation_metric is None else self.validation_metric
            program.validation_fitness_ = classification_loss(y_val, val_proba, val_metric, sw_val)
        else:
            program.validation_fitness_ = program.raw_fitness_
        return program

    def _robustness_loss(self, program, raw, y, sample_weight):
        metric = self.fitness_metric if self.metric is None else self.metric
        proba = probability_from_raw(raw, program.scale_, program.intercept_)
        return classification_loss(y, proba, metric, sample_weight)

    # ----------------------------- fitting -----------------------------
    def fit(self, X, y, sample_weight=None):
        if str(self.link_function).lower() != "sigmoid":
            raise ValueError("FastSymbolicClassifier supports link_function='sigmoid'")
        feature_names = getattr(X, "columns", None)
        X, y = check_X_y_finite(X, y, dtype=np.float64, ensure_2d=True, allow_nan=self._allow_nan())
        self._apply_preset(X, y)
        X = self._prepare_X_fit(X)
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] < 2:
            raise ValueError("At least two target classes are required")
        if self.classes_.shape[0] > 2:
            return self._fit_multiclass(X, y, sample_weight)
        self._is_multiclass_ = False
        self.multiclass_strategy_ = None
        return self._fit_binary(X, y, sample_weight)

    def _binary_child_params(self, index):
        params = self.get_params(deep=False).copy()
        params["random_state"] = None if self.random_state is None else int(self.random_state) + index * 1009
        params["resume_from_checkpoint"] = False
        params["preset"] = None
        params["multiclass_strategy"] = "ovr"
        params["multiclass_calibration"] = "none"
        if self.checkpoint_path:
            root = Path(self.checkpoint_path)
            params["checkpoint_path"] = str(root / f"class_{index}")
        return params

    def _fit_multiclass(self, X, y, sample_weight=None):
        strategy = str(self.multiclass_strategy).lower()
        if strategy in {"auto"}:
            strategy = "shared_softmax" if len(self.classes_) <= 20 else "ovr"
        if strategy in {"one_vs_rest", "one-vs-rest"}:
            strategy = "ovr"
        if strategy not in {"ovr", "shared_softmax", "shared", "softmax"}:
            raise ValueError("multiclass_strategy must be ovr or shared_softmax")
        self._is_multiclass_ = True
        self.multiclass_strategy_ = "shared_softmax" if strategy in {"shared_softmax", "shared", "softmax"} else "ovr"

        # Multiclass backbones use all available data. Temperature scaling is
        # deliberately conservative (T >= 0.75) because flexible calibration
        # on the small validation sets common in symbolic benchmarks can create
        # severely overconfident probabilities.
        X_model, y_model = X, y
        sw_model = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        calibration_source = "training_predictions_conservative_single_parameter"

        self.estimators_ = []
        self.history_by_class_ = {}
        self.expressions_ = []
        total_runtime = 0.0
        if int(self.verbose) > 0:
            print("\n╔" + "═" * 104 + "╗")
            title = f" FASTSYMBOLICGP 0.7.0 // {self.multiclass_strategy_.upper()} // {len(self.classes_)} CLASS SYMBOLIC SYSTEM "
            print("║" + title.center(104) + "║")
            print("╚" + "═" * 104 + "╝")
        for index, class_label in enumerate(self.classes_):
            if int(self.verbose) > 0:
                print(f"\n▶ BACKBONE CLASS {index + 1}/{len(self.classes_)}: {class_label!r} versus rest")
            estimator = FastSymbolicClassifier(**self._binary_child_params(index))
            target = (y_model == class_label).astype(int)
            estimator.fit(X_model, target, sample_weight=sw_model)
            self.estimators_.append(estimator)
            self.history_by_class_[str(class_label)] = estimator.history_
            self.expressions_.append(estimator.get_expression())
            total_runtime += float(estimator.run_time_seconds_)

        self.run_time_seconds_ = total_runtime
        per_class_generations = [int(e.n_generations_) for e in self.estimators_]
        self.n_generations_total_ = int(sum(per_class_generations))
        self.n_generations_mean_per_class_ = float(np.mean(per_class_generations))
        self.n_generations_max_per_class_ = int(max(per_class_generations))
        # n_generations_ is the wall-clock-equivalent maximum, not the misleading sum.
        self.n_generations_ = self.n_generations_max_per_class_
        self.history_ = []
        for class_label, estimator in zip(self.classes_, self.estimators_):
            for row in estimator.history_:
                self.history_.append({"class": str(class_label), **row})

        if self.multiclass_strategy_ == "shared_softmax":
            self._fit_shared_softmax_head(X_model, y_model, sw_model)
        else:
            self.shared_programs_ = []
            self.softmax_model_ = None

        X_reference = X_model
        y_reference = y_model
        sw_reference = sw_model
        base = self._multiclass_base_probability(X_reference)
        calibration = str(self.multiclass_calibration).lower()
        self.multiclass_calibrator_ = None
        identity_loss = float(log_loss(y_reference, base, labels=self.classes_, sample_weight=sw_reference))
        if calibration in {"temperature", "auto"}:
            candidate = TemperatureScaler(min_temperature=0.75, max_temperature=4.0).fit(base, y_reference, sample_weight=sw_reference)
            scaled = candidate.predict_proba(base)
            scaled_loss = float(log_loss(y_reference, scaled, labels=self.classes_, sample_weight=sw_reference))
            # Retain the identity map unless the conservative temperature
            # materially improves the reference log loss.
            required_gain = max(1e-4, identity_loss * 0.01)
            if scaled_loss < identity_loss - required_gain:
                self.multiclass_calibrator_ = candidate
        elif calibration not in {"none", "identity", "off"}:
            raise ValueError("multiclass_calibration must be none, temperature, or auto")
        calibrated = self._apply_multiclass_calibration(base)
        self.multiclass_calibration_report_ = {
            "requested_method": calibration,
            "selected_method": "temperature" if self.multiclass_calibrator_ is not None else "none",
            "temperature": None if self.multiclass_calibrator_ is None else float(self.multiclass_calibrator_.temperature_),
            "identity_log_loss": identity_loss,
            "log_loss": float(log_loss(y_reference, calibrated, labels=self.classes_, sample_weight=sw_reference)),
            "brier_score": multiclass_brier_score(y_reference, calibrated, self.classes_),
            "expected_calibration_error": expected_calibration_error(y_reference, calibrated),
            "fit_source": calibration_source,
            "samples": int(len(y_reference)),
        }
        self._aggregate_multiclass_usage()
        if int(self.verbose) > 0:
            stats = self.get_expression_stats()
            print(
                f"\n✓ Multiclass system ready: strategy={self.multiclass_strategy_}, "
                f"DAG nodes={stats['dag_nodes_total']}, calibration={self.multiclass_calibration_report_['selected_method']}"
            )
        return self

    def _fit_shared_softmax_head(self, X, y, sample_weight=None):
        candidates = []
        for estimator in self.estimators_:
            candidates.extend(estimator.hall_of_fame_)
        candidates = sorted(candidates, key=lambda p: (p.selection_fitness_, p.dag_size, p.depth))
        selected, outputs = [], []
        for candidate in candidates:
            output = candidate.execute(np.ascontiguousarray(X), backend=getattr(self.estimators_[0], "evaluation_backend_", "numpy"))
            if np.std(output) < 1e-12:
                continue
            if outputs:
                correlations = [abs(float(np.corrcoef(output, old)[0, 1])) if np.std(old) > 1e-12 else 1.0 for old in outputs]
                if max(correlations) > float(self.shared_max_correlation):
                    continue
            selected.append(candidate.clone())
            outputs.append(output)
            if len(selected) >= int(self.shared_n_components):
                break
        if not selected:
            selected = [est.best_program_.clone() for est in self.estimators_]
            outputs = [program.execute(np.ascontiguousarray(X), backend=getattr(self.estimators_[0], "evaluation_backend_", "numpy")) for program in selected]
        Z = np.column_stack(outputs)
        self.shared_feature_mean_ = np.mean(Z, axis=0)
        self.shared_feature_scale_ = np.std(Z, axis=0)
        self.shared_feature_scale_ = np.where(self.shared_feature_scale_ > 1e-12, self.shared_feature_scale_, 1.0)
        Zs = (Z - self.shared_feature_mean_) / self.shared_feature_scale_
        self.softmax_model_ = LogisticRegression(
            C=float(self.shared_regularization), solver="lbfgs", max_iter=1000,
            random_state=self.random_state,
        )
        self.softmax_model_.fit(Zs, y, sample_weight=sample_weight)
        self.shared_programs_ = selected
        self.shared_expressions_ = [p.to_string(getattr(self, "feature_names_in_", None)) for p in selected]

    def _fit_binary(self, X, y, sample_weight=None):
        y_binary = (y == self.classes_[1]).astype(int)
        full_weights = resolve_class_weight(y_binary, self.class_weight)
        if sample_weight is not None:
            full_weights *= np.asarray(sample_weight, dtype=float)
        if float(self.validation_fraction) > 0:
            indices = np.arange(X.shape[0])
            train_idx, val_idx = train_test_split(
                indices, test_size=float(self.validation_fraction),
                random_state=self.random_state, stratify=y_binary,
            )
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_binary[train_idx], y_binary[val_idx]
            sw_train, sw_val = full_weights[train_idx], full_weights[val_idx]
        else:
            X_train, y_train, sw_train = X, y_binary, full_weights
            X_val = y_val = sw_val = None

        self._evolve(
            np.ascontiguousarray(X_train), y_train,
            None if X_val is None else np.ascontiguousarray(X_val), y_val,
            sw_train, sw_val,
        )
        raw_train = self._execute_program(self.best_program_, X_train)
        self.scale_, self.intercept_ = fit_logistic_scaler(
            raw_train, y_train, sw_train,
            regularization=float(self.fitness_scaling_regularization), max_iter=30,
        )
        self.best_program_.scale_, self.best_program_.intercept_ = self.scale_, self.intercept_

        mode = str(self.prediction_mode).lower()
        if mode in {"ensemble", "symbolic_ensemble", "elite_ensemble"}:
            self._fit_symbolic_ensemble(X_train, y_train, X_val, y_val, sw_train, sw_val)
        elif mode == "best":
            self.ensemble_programs_ = [self.best_program_]
            self.ensemble_weights_ = np.asarray([1.0])
            self.ensemble_report_ = {"size": 1, "weights": [1.0], "nodes": [self.best_program_.size], "dag_nodes": [self.best_program_.dag_size]}
        else:
            raise ValueError("prediction_mode must be best or symbolic_ensemble")

        self.calibration_model_ = None
        self.calibration_report_ = {"selected_method": "none"}
        self.decision_threshold_ = 0.5
        self.threshold_score_ = None
        if X_val is not None:
            base_val = self._base_probability(X_val)
            self.calibration_model_, self.calibration_report_, _ = fit_calibrator(
                base_val, y_val, method=self.probability_calibration,
                sample_weight=sw_val, cv=int(self.calibration_cv), random_state=self.random_state,
            )
            calibrated = np.clip(apply_calibrator(self.calibration_model_, base_val), float(self.probability_clip), 1 - float(self.probability_clip))
            strategy = str(self.threshold_strategy).lower()
            if strategy not in {"fixed", "0.5", "none"}:
                self.decision_threshold_, self.threshold_score_ = optimize_threshold(y_val, calibrated, metric=strategy, sample_weight=sw_val)
            self.calibration_report_["validation_log_loss"] = float(log_loss(y_val, calibrated, sample_weight=sw_val, labels=[0, 1]))
            self.calibration_report_["expected_calibration_error"] = expected_calibration_error(y_val, np.column_stack((1-calibrated, calibrated)))
            self.calibration_report_["decision_threshold"] = float(self.decision_threshold_)
        return self

    # ----------------------------- ensemble -----------------------------
    def _fit_symbolic_ensemble(self, X_train, y_train, X_val, y_val, sw_train, sw_val):
        reference_X = X_val if X_val is not None else X_train
        reference_y = y_val if y_val is not None else y_train
        reference_w = sw_val if X_val is not None else sw_train
        selected, predictions = [], []
        candidates = sorted(self.hall_of_fame_, key=lambda p: (p.selection_fitness_, p.dag_size, p.depth))
        for candidate in candidates:
            program = candidate.clone()
            raw_train = self._execute_program(program, X_train)
            program.scale_, program.intercept_ = fit_logistic_scaler(
                raw_train, y_train, sw_train,
                regularization=float(self.fitness_scaling_regularization), max_iter=30,
            )
            p = probability_from_raw(self._execute_program(program, reference_X), program.scale_, program.intercept_)
            if np.std(p) < 1e-12:
                continue
            if predictions:
                correlations = [abs(float(np.corrcoef(p, old)[0, 1])) if np.std(old) > 1e-12 else 1.0 for old in predictions]
                if max(correlations) > float(self.max_ensemble_correlation):
                    continue
            selected.append(program); predictions.append(p)
            if len(selected) >= int(self.ensemble_size):
                break
        if len(selected) < int(self.ensemble_size):
            used = {p.structural_hash() for p in selected}
            remaining = []
            for candidate in candidates:
                if candidate.structural_hash() in used:
                    continue
                program = candidate.clone()
                raw_train = self._execute_program(program, X_train)
                program.scale_, program.intercept_ = fit_logistic_scaler(raw_train, y_train, sw_train, regularization=float(self.fitness_scaling_regularization), max_iter=30)
                p = probability_from_raw(self._execute_program(program, reference_X), program.scale_, program.intercept_)
                if np.std(p) < 1e-12:
                    continue
                max_corr = max([abs(float(np.corrcoef(p, old)[0, 1])) for old in predictions], default=0.0)
                remaining.append((max_corr, program.selection_fitness_, program, p))
            for _, _, program, p in sorted(remaining, key=lambda row: (row[0], row[1])):
                selected.append(program); predictions.append(p)
                if len(selected) >= int(self.ensemble_size):
                    break
        if not selected:
            selected = [self.best_program_.clone()]
            predictions = [probability_from_raw(self._execute_program(selected[0], reference_X), selected[0].scale_, selected[0].intercept_)]
        matrix = np.column_stack(predictions)
        weighting = str(self.ensemble_weighting).lower()
        if weighting == "uniform" or matrix.shape[1] == 1:
            weights = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
        elif weighting in {"validation", "score"}:
            losses = np.asarray([log_loss(reference_y, np.clip(matrix[:, j], 1e-6, 1 - 1e-6), sample_weight=reference_w, labels=[0, 1]) for j in range(matrix.shape[1])])
            quality = np.exp(-(losses - losses.min()) / max(losses.std(), 1e-6))
            weights = quality / quality.sum()
        elif weighting in {"optimized", "nonnegative"}:
            weights = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
            yy = np.asarray(reference_y, dtype=float)
            ww = np.ones_like(yy) if reference_w is None else np.asarray(reference_w, dtype=float)
            for step in range(250):
                p = np.clip(matrix @ weights, 1e-6, 1 - 1e-6)
                gradient = matrix.T @ (ww * (p - yy) / np.maximum(p * (1 - p), 1e-6)) / max(ww.sum(), 1.0)
                weights = _project_simplex(weights - 0.03 / np.sqrt(step + 1) * gradient)
        else:
            raise ValueError("ensemble_weighting must be uniform, validation, or optimized")
        self.ensemble_programs_ = selected
        self.ensemble_weights_ = np.asarray(weights, dtype=float)
        self.ensemble_report_ = {
            "size": len(selected), "weights": self.ensemble_weights_.tolist(),
            "nodes": [p.size for p in selected], "dag_nodes": [p.dag_size for p in selected],
            "depths": [p.depth for p in selected],
            "expressions": [p.to_string(getattr(self, "feature_names_in_", None)) for p in selected],
        }

    def _base_probability(self, X):
        X = self._prepare_X_predict(X)
        members = []
        for program in self.ensemble_programs_:
            raw = self._execute_program(program, X)
            members.append(probability_from_raw(raw, program.scale_, program.intercept_))
        return np.clip(np.column_stack(members) @ self.ensemble_weights_, float(self.probability_clip), 1 - float(self.probability_clip))

    # ----------------------------- multiclass probability -----------------------------
    def _shared_features(self, X):
        X = self._prepare_X_predict(X)
        backend = getattr(self.estimators_[0], "evaluation_backend_", "numpy")
        Z = np.column_stack([p.execute(X, backend=backend) for p in self.shared_programs_])
        return (Z - self.shared_feature_mean_) / self.shared_feature_scale_

    def _multiclass_base_probability(self, X):
        if self.multiclass_strategy_ == "shared_softmax":
            return self.softmax_model_.predict_proba(self._shared_features(X))
        scores = np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])
        denom = scores.sum(axis=1, keepdims=True)
        zero = denom[:, 0] <= 1e-15
        scores[zero] = 1.0
        return scores / scores.sum(axis=1, keepdims=True)

    def _apply_multiclass_calibration(self, probabilities):
        if getattr(self, "multiclass_calibrator_", None) is None:
            return probabilities
        return self.multiclass_calibrator_.predict_proba(probabilities)

    # ----------------------------- inference -----------------------------
    def predict_proba(self, X):
        if getattr(self, "_is_multiclass_", False):
            check_is_fitted(self, "estimators_")
            return self._apply_multiclass_calibration(self._multiclass_base_probability(X))
        check_is_fitted(self, "best_program_")
        p1 = np.clip(apply_calibrator(self.calibration_model_, self._base_probability(X)), float(self.probability_clip), 1 - float(self.probability_clip))
        return np.column_stack((1.0 - p1, p1))

    def predict(self, X):
        proba = self.predict_proba(X)
        if getattr(self, "_is_multiclass_", False):
            return self.classes_[np.argmax(proba, axis=1)]
        encoded = (proba[:, 1] >= float(self.decision_threshold_)).astype(int)
        return self.classes_[encoded]

    def raw_score(self, X):
        if getattr(self, "_is_multiclass_", False):
            if self.multiclass_strategy_ == "shared_softmax":
                return self.softmax_model_.decision_function(self._shared_features(X))
            return np.column_stack([est.raw_score(X) for est in self.estimators_])
        return super().raw_score(X)

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def refit_threshold(self, X, y, metric="mcc", sample_weight=None):
        if getattr(self, "_is_multiclass_", False):
            raise ValueError("A single threshold is not defined for multiclass models")
        y = np.asarray(y)
        y_binary = (y == self.classes_[1]).astype(int)
        self.decision_threshold_, self.threshold_score_ = optimize_threshold(y_binary, self.predict_proba(X)[:, 1], metric=metric, sample_weight=sample_weight)
        return self

    # ----------------------------- interpretability -----------------------------
    def _aggregate_multiclass_usage(self):
        usage = {}
        programs = self.shared_programs_ if self.multiclass_strategy_ == "shared_softmax" else [e.best_program_ for e in self.estimators_]
        names = getattr(self, "feature_names_in_", None)
        for program in programs:
            for index, count in program.feature_counts().items():
                key = str(names[index]) if names is not None else f"X{index}"
                usage[key] = usage.get(key, 0) + count
        total = max(sum(usage.values()), 1)
        self.feature_usage_ = {key: {"count": int(value), "fraction": float(value / total)} for key, value in sorted(usage.items(), key=lambda item: -item[1])}

    def explain_instance(self, x, baseline=None, top_k=10):
        """Return a model-agnostic local feature-occlusion explanation."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features")
        if baseline is None:
            baseline = getattr(self, "missing_fill_values_", None)
            if baseline is None:
                baseline = np.zeros(self.n_features_in_, dtype=float)
        baseline = np.asarray(baseline, dtype=float)
        original = self.predict_proba(x)[0]
        predicted_index = int(np.argmax(original))
        contributions = []
        names = getattr(self, "feature_names_in_", None)
        for j in range(self.n_features_in_):
            changed = x.copy()
            changed[0, j] = baseline[j]
            probability = self.predict_proba(changed)[0, predicted_index]
            contributions.append({
                "feature": str(names[j]) if names is not None else f"X{j}",
                "index": j, "value": float(x[0, j]), "baseline": float(baseline[j]),
                "contribution": float(original[predicted_index] - probability),
            })
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        report = {
            "prediction": self.classes_[predicted_index].item() if hasattr(self.classes_[predicted_index], "item") else self.classes_[predicted_index],
            "probabilities": original.tolist(),
            "top_contributions": contributions[: int(top_k)],
        }
        if not getattr(self, "_is_multiclass_", False):
            report["ensemble_votes"] = [
                float(probability_from_raw(self._execute_program(p, self._prepare_X_predict(x)), p.scale_, p.intercept_)[0])
                for p in self.ensemble_programs_
            ]
        return report

    def counterfactual(self, x, target_class=None, feature_ranges=None, max_changed_features=3, grid_points=11):
        """Greedy transparent counterfactual search over feature ranges."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        current = self.predict(x)[0]
        if target_class is None:
            if len(self.classes_) != 2:
                raise ValueError("target_class is required for multiclass counterfactuals")
            target_class = self.classes_[0] if current == self.classes_[1] else self.classes_[1]
        target_index = int(np.where(self.classes_ == target_class)[0][0])
        if feature_ranges is None:
            center = x[0]
            scale = np.maximum(np.abs(center), 1.0)
            lows, highs = center - 2 * scale, center + 2 * scale
        else:
            ranges = np.asarray(feature_ranges, dtype=float)
            lows, highs = ranges[:, 0], ranges[:, 1]
        candidate = x.copy()
        changes = []
        for _ in range(int(max_changed_features)):
            best = None
            base_score = float(self.predict_proba(candidate)[0, target_index])
            for j in range(self.n_features_in_):
                for value in np.linspace(lows[j], highs[j], int(grid_points)):
                    trial = candidate.copy(); trial[0, j] = value
                    score = float(self.predict_proba(trial)[0, target_index])
                    improvement = score - base_score
                    if best is None or improvement > best[0]:
                        best = (improvement, j, float(value), score)
            if best is None or best[0] <= 0:
                break
            _, j, value, score = best
            old = float(candidate[0, j]); candidate[0, j] = value
            changes.append({"feature": int(j), "from": old, "to": value, "target_probability": score})
            if self.predict(candidate)[0] == target_class:
                break
        return {
            "original_class": current.item() if hasattr(current, "item") else current,
            "target_class": target_class.item() if hasattr(target_class, "item") else target_class,
            "achieved": bool(self.predict(candidate)[0] == target_class),
            "changes": changes,
            "counterfactual": candidate[0].tolist(),
            "probabilities": self.predict_proba(candidate)[0].tolist(),
        }

    def distill(self, X, max_nodes=40, generations=120, population_size=64, random_state=None, **kwargs):
        from .distillation import DistilledSymbolicClassifier
        distilled = DistilledSymbolicClassifier(
            max_nodes=max_nodes, generations=generations, population_size=population_size,
            random_state=self.random_state if random_state is None else random_state,
            model_params=kwargs,
        )
        return distilled.fit_from_teacher(self, X)

    # ----------------------------- reporting -----------------------------
    def get_expression(self):
        if getattr(self, "_is_multiclass_", False):
            if self.multiclass_strategy_ == "shared_softmax":
                return {
                    "shared_backbone": list(self.shared_expressions_),
                    "softmax_coefficients": self.softmax_model_.coef_.tolist(),
                    "softmax_intercepts": self.softmax_model_.intercept_.tolist(),
                }
            return {str(c): e.get_expression() for c, e in zip(self.classes_, self.estimators_)}
        return super().get_expression()

    def get_raw_expression(self):
        if getattr(self, "_is_multiclass_", False):
            return {str(c): e.get_raw_expression() for c, e in zip(self.classes_, self.estimators_)}
        return super().get_raw_expression()

    def get_expression_stats(self):
        if getattr(self, "_is_multiclass_", False):
            if self.multiclass_strategy_ == "shared_softmax":
                programs = self.shared_programs_
                per_model = [{"nodes": p.size, "dag_nodes": p.dag_size, "depth": p.depth} for p in programs]
            else:
                programs = [e.best_program_ for e in self.estimators_]
                per_model = [e.get_expression_stats() for e in self.estimators_]
            nodes = [p.size for p in programs]
            dag_nodes = [p.dag_size for p in programs]
            depths = [p.depth for p in programs]
            return {
                "n_classes": len(self.classes_), "multiclass_strategy": self.multiclass_strategy_,
                "models": len(programs),
                "nodes_total": int(sum(nodes)), "nodes_mean_per_model": float(np.mean(nodes)), "nodes_max_per_model": int(max(nodes)),
                "dag_nodes_total": int(sum(dag_nodes)), "dag_nodes_mean_per_model": float(np.mean(dag_nodes)), "dag_nodes_max_per_model": int(max(dag_nodes)),
                "depth_mean_per_model": float(np.mean(depths)), "depth_max_per_model": int(max(depths)),
                "generations_total": int(self.n_generations_total_),
                "generations_mean_per_class": float(self.n_generations_mean_per_class_),
                "generations_max_per_class": int(self.n_generations_max_per_class_),
                "per_model": per_model,
            }
        result = super().get_expression_stats()
        result.update({
            "prediction_mode": self.prediction_mode,
            "ensemble_size": len(getattr(self, "ensemble_programs_", [self.best_program_])),
            "decision_threshold": float(getattr(self, "decision_threshold_", 0.5)),
        })
        return result


    def profile_prediction(self, X, repeats=100, warmup=5):
        """Measure inference latency and structural cost for binary or multiclass models."""
        check_is_fitted(self, "classes_")
        X = self._prepare_X_predict(X)
        for _ in range(max(0, int(warmup))):
            self.predict(X)
        tracemalloc.start()
        start = perf_counter()
        calls = max(1, int(repeats))
        for _ in range(calls):
            self.predict(X)
        elapsed = perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples = max(1, X.shape[0] * calls)
        stats = self.get_expression_stats()
        if getattr(self, "_is_multiclass_", False):
            tree_ops = int(stats["nodes_total"])
            dag_ops = int(stats["dag_nodes_total"])
            compression = 0.0 if tree_ops == 0 else (1.0 - dag_ops / tree_ops)
        else:
            tree_ops = int(stats["nodes"])
            dag_ops = int(stats["dag_nodes"])
            compression = float(stats.get("compression_ratio", stats.get("dag_compression_ratio", 0.0)))
        return {
            "repeats": calls, "batch_size": int(X.shape[0]),
            "seconds_total": float(elapsed),
            "milliseconds_per_batch": float(elapsed * 1000 / calls),
            "microseconds_per_sample": float(elapsed * 1e6 / samples),
            "peak_working_memory_bytes": int(peak),
            "tree_operations": tree_ops, "dag_operations": dag_ops,
            "dag_compression_ratio": float(compression),
            "execution_engine": getattr(self, "execution_engine_",
                getattr(self.estimators_[0], "execution_engine_", "unknown") if getattr(self, "_is_multiclass_", False) else "unknown"),
        }

    def save_report_json(self, path):
        if not getattr(self, "_is_multiclass_", False):
            super().save_report_json(path)
            import json
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            payload["calibration"] = getattr(self, "calibration_report_", {})
            payload["ensemble"] = getattr(self, "ensemble_report_", {"size": 1, "weights": [1.0]})
            save_json(payload, path)
            return
        payload = {
            "version": "0.7.0", "estimator": self.__class__.__name__,
            "multiclass_strategy": self.multiclass_strategy_,
            "evaluation_backend": getattr(self.estimators_[0], "evaluation_backend_", self.evaluation_backend),
            "execution_engine": getattr(self.estimators_[0], "execution_engine_", "unknown"),
            "classes": [str(c) for c in self.classes_],
            "run_time_seconds": self.run_time_seconds_,
            "generations_total": self.n_generations_total_,
            "generations_mean_per_class": self.n_generations_mean_per_class_,
            "generations_max_per_class": self.n_generations_max_per_class_,
            "calibration": self.multiclass_calibration_report_,
            "expressions": self.get_expression(), "expression_stats": self.get_expression_stats(),
            "feature_usage": getattr(self, "feature_usage_", {}),
            "per_class": [
                {"class": str(c), "calibration": e.calibration_report_, "ensemble": getattr(e, "ensemble_report_", {}), "history": e.history_}
                for c, e in zip(self.classes_, self.estimators_)
            ],
        }
        save_json(payload, path)
