"""Supervised symbolic feature construction for FastSymbolicGP V0.7.0."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .classifier import FastSymbolicClassifier
from .regressor import FastSymbolicRegressor
from .validation import check_X_y_finite, check_array_finite


class FastSymbolicTransformer(TransformerMixin, BaseEstimator):
    """Evolve a compact, diverse bank of supervised symbolic features.

    Parameters
    ----------
    component_selection : {"pareto", "diversity_validation", "mrmr"}
        Greedy selection criterion for the final feature bank.
    include_original_features : bool
        Concatenate the original matrix to symbolic features during transform.
    standardize_outputs : bool
        Standardize symbolic components using training statistics.
    """

    def __init__(
        self,
        n_components=10,
        task="auto",
        component_selection="mrmr",
        max_correlation=0.90,
        redundancy_penalty=0.10,
        include_original_features=False,
        standardize_outputs=True,
        model_params=None,
        random_state=None,
        verbose=1,
    ):
        self.n_components = n_components
        self.task = task
        self.component_selection = component_selection
        self.max_correlation = max_correlation
        self.redundancy_penalty = redundancy_penalty
        self.include_original_features = include_original_features
        self.standardize_outputs = standardize_outputs
        self.model_params = model_params
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        feature_names = getattr(X, "columns", None)
        X, y = check_X_y_finite(X, y, dtype=np.float64, ensure_2d=True, allow_nan=True)
        self.n_features_in_ = X.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        target_type = type_of_target(y)
        task = str(self.task).lower()
        if task == "auto":
            task = "classification" if target_type in {"binary", "multiclass"} else "regression"
        params = dict(self.model_params or {})
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbose", self.verbose)
        params.setdefault("hall_of_fame_size", max(30, int(self.n_components) * 6))
        params.setdefault("low_memory", False)
        if np.isnan(X).any():
            params.setdefault("missing_value_strategy", "median")
        if task == "classification":
            params.setdefault("probability_calibration", "none")
            params.setdefault("prediction_mode", "best")
            self.estimator_ = FastSymbolicClassifier(**params).fit(X, y, sample_weight=sample_weight)
        elif task == "regression":
            self.estimator_ = FastSymbolicRegressor(**params).fit(X, y, sample_weight=sample_weight)
        else:
            raise ValueError("task must be auto, classification, or regression")
        self.task_ = task

        candidate_programs = []
        if getattr(self.estimator_, "_is_multiclass_", False):
            for estimator in self.estimator_.estimators_:
                candidate_programs.extend(estimator.hall_of_fame_)
        else:
            candidate_programs.extend(self.estimator_.hall_of_fame_)
        unique = {}
        for program in candidate_programs:
            key = program.structural_hash()
            if key not in unique or program.selection_fitness_ < unique[key].selection_fitness_:
                unique[key] = program
        candidate_programs = sorted(unique.values(), key=lambda p: (p.selection_fitness_, p.dag_size, p.depth))
        backend = getattr(self.estimator_, "evaluation_backend_", "numpy")
        outputs, programs = [], []
        Xc = np.ascontiguousarray(np.nan_to_num(X, nan=0.0))
        for program in candidate_programs:
            values = program.execute(Xc, backend=backend)
            if np.std(values) < 1e-12 or not np.isfinite(values).all():
                continue
            outputs.append(values)
            programs.append(program)
        if not programs:
            if getattr(self.estimator_, "_is_multiclass_", False):
                programs = [e.best_program_.clone() for e in self.estimator_.estimators_]
            else:
                programs = [self.estimator_.best_program_.clone()]
            outputs = [program.execute(Xc, backend=backend) for program in programs]

        matrix = np.column_stack(outputs)
        relevance = self._relevance(matrix, y, task)
        selected_indices = self._select_components(matrix, relevance, programs)
        self.programs_ = [programs[i].clone() for i in selected_indices]
        selected_matrix = matrix[:, selected_indices]
        self.component_relevance_ = relevance[selected_indices]
        self.component_correlation_ = np.corrcoef(selected_matrix, rowvar=False) if selected_matrix.shape[1] > 1 else np.asarray([[1.0]])
        self.symbolic_mean_ = selected_matrix.mean(axis=0)
        self.symbolic_scale_ = selected_matrix.std(axis=0)
        self.symbolic_scale_ = np.where(self.symbolic_scale_ > 1e-12, self.symbolic_scale_, 1.0)
        self.n_components_ = len(self.programs_)
        symbolic_names = [f"symbolic_{i:03d}" for i in range(self.n_components_)]
        if self.include_original_features:
            original = list(getattr(self, "feature_names_in_", [f"X{i}" for i in range(self.n_features_in_)]))
            self.feature_names_out_ = np.asarray(original + symbolic_names, dtype=object)
        else:
            self.feature_names_out_ = np.asarray(symbolic_names, dtype=object)
        self.component_report_ = [
            {
                "name": symbolic_names[i], "relevance": float(self.component_relevance_[i]),
                "nodes": int(program.size), "dag_nodes": int(program.dag_size),
                "depth": int(program.depth), "expression": program.to_string(getattr(self, "feature_names_in_", None)),
            }
            for i, program in enumerate(self.programs_)
        ]
        return self

    def _relevance(self, matrix, y, task):
        try:
            if task == "classification":
                return np.asarray(mutual_info_classif(matrix, y, random_state=self.random_state), dtype=float)
            return np.asarray(mutual_info_regression(matrix, y, random_state=self.random_state), dtype=float)
        except Exception:
            values = []
            y_numeric = np.asarray(y, dtype=float)
            for j in range(matrix.shape[1]):
                corr = np.corrcoef(matrix[:, j], y_numeric)[0, 1]
                values.append(0.0 if not np.isfinite(corr) else abs(float(corr)))
            return np.asarray(values)

    def _select_components(self, matrix, relevance, programs):
        method = str(self.component_selection).lower()
        remaining = set(range(matrix.shape[1]))
        selected = []
        while remaining and len(selected) < int(self.n_components):
            best_index, best_score = None, -np.inf
            for index in remaining:
                if selected:
                    correlations = [abs(float(np.corrcoef(matrix[:, index], matrix[:, old])[0, 1])) for old in selected]
                    max_corr = max(0.0 if not np.isfinite(c) else c for c in correlations)
                    if max_corr > float(self.max_correlation):
                        continue
                    redundancy = float(np.mean(correlations))
                else:
                    redundancy = 0.0
                if method == "pareto":
                    score = -float(programs[index].selection_fitness_) - 1e-3 * programs[index].dag_size
                elif method in {"diversity_validation", "diversity"}:
                    score = -float(programs[index].selection_fitness_) - float(self.redundancy_penalty) * redundancy
                elif method == "mrmr":
                    score = float(relevance[index]) - float(self.redundancy_penalty) * redundancy
                else:
                    raise ValueError("component_selection must be pareto, diversity_validation, or mrmr")
                if score > best_score:
                    best_index, best_score = index, score
            if best_index is None:
                # Relax only when the diversity filter blocks every candidate.
                best_index = max(remaining, key=lambda i: float(relevance[i]))
            selected.append(best_index)
            remaining.remove(best_index)
        return selected

    def transform(self, X):
        check_is_fitted(self, "programs_")
        X = check_array_finite(X, dtype=np.float64, ensure_2d=True, allow_nan=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        Xc = np.ascontiguousarray(np.nan_to_num(X, nan=0.0))
        backend = getattr(self.estimator_, "evaluation_backend_", "numpy")
        symbolic = np.column_stack([p.execute(Xc, backend=backend) for p in self.programs_])
        if self.standardize_outputs:
            symbolic = (symbolic - self.symbolic_mean_) / self.symbolic_scale_
        return np.column_stack((Xc, symbolic)) if self.include_original_features else symbolic

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def get_expressions(self):
        check_is_fitted(self, "programs_")
        names = getattr(self, "feature_names_in_", None)
        return [p.to_string(names) for p in self.programs_]

    def get_component_report(self):
        check_is_fitted(self, "component_report_")
        return list(self.component_report_)
