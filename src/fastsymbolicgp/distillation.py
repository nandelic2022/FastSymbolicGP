"""Symbolic model distillation utilities."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from .regressor import FastSymbolicRegressor


class DistilledSymbolicClassifier(ClassifierMixin, BaseEstimator):
    """Compress a symbolic ensemble or multiclass system into small equations."""

    def __init__(
        self,
        population_size=64,
        generations=120,
        max_nodes=40,
        temperature=2.0,
        random_state=None,
        model_params=None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_nodes = max_nodes
        self.temperature = temperature
        self.random_state = random_state
        self.model_params = model_params

    def fit(self, X, y):
        """Fit a compact symbolic classifier to smoothed hard labels."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        encoded = np.searchsorted(classes, y)
        probabilities = np.full((len(y), len(classes)), 0.02 / max(1, len(classes) - 1), dtype=float)
        probabilities[np.arange(len(y)), encoded] = 0.98

        class _LabelTeacher:
            def __init__(self, classes_, probabilities_):
                self.classes_ = classes_
                self._probabilities = probabilities_
            def predict_proba(self, X_):
                return self._probabilities
            def get_expression_stats(self):
                return {"source": "smoothed_hard_labels"}

        return self.fit_from_teacher(_LabelTeacher(classes, probabilities), X)

    def fit_from_teacher(self, teacher, X):
        probabilities = np.clip(teacher.predict_proba(X), 1e-7, 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        self.classes_ = np.asarray(teacher.classes_)
        self.n_features_in_ = X.shape[1]
        self.teacher_expression_stats_ = teacher.get_expression_stats()
        params = dict(self.model_params or {})
        params.setdefault("population_size", int(self.population_size))
        params.setdefault("generations", int(self.generations))
        params.setdefault("max_nodes", int(self.max_nodes))
        params.setdefault("max_depth", max(4, int(np.ceil(np.log2(max(2, int(self.max_nodes)))) + 2)))
        params.setdefault("selection_tolerance", 0.03)
        params.setdefault("final_selection", "smallest_within_tolerance")
        params.setdefault("optimization", "nsga2")
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbose", 0)
        self.regressors_ = []
        if len(self.classes_) == 2:
            p = probabilities[:, 1]
            target = np.log(p / (1.0 - p)) / max(float(self.temperature), 1e-9)
            self.regressors_.append(FastSymbolicRegressor(**params).fit(X, target))
        else:
            logits = np.log(probabilities)
            logits -= logits.mean(axis=1, keepdims=True)
            logits /= max(float(self.temperature), 1e-9)
            for index in range(len(self.classes_)):
                child = dict(params)
                child["random_state"] = None if self.random_state is None else int(self.random_state) + index * 4099
                self.regressors_.append(FastSymbolicRegressor(**child).fit(X, logits[:, index]))
        self.expression_stats_ = {
            "models": len(self.regressors_),
            "nodes_total": int(sum(r.best_program_.size for r in self.regressors_)),
            "dag_nodes_total": int(sum(r.best_program_.dag_size for r in self.regressors_)),
            "depth_max": int(max(r.best_program_.depth for r in self.regressors_)),
        }
        return self

    @staticmethod
    def _softmax(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(np.clip(logits, -50.0, 50.0))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        check_is_fitted(self, "regressors_")
        if len(self.classes_) == 2:
            logit = self.regressors_[0].predict(X) * float(self.temperature)
            p1 = 1.0 / (1.0 + np.exp(-np.clip(logit, -35.0, 35.0)))
            return np.column_stack((1.0 - p1, p1))
        logits = np.column_stack([reg.predict(X) for reg in self.regressors_]) * float(self.temperature)
        return self._softmax(logits)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_expression(self):
        check_is_fitted(self, "regressors_")
        if len(self.classes_) == 2:
            return self.regressors_[0].get_expression()
        return {str(c): reg.get_expression() for c, reg in zip(self.classes_, self.regressors_)}

    def get_expression_stats(self):
        check_is_fitted(self, "expression_stats_")
        return dict(self.expression_stats_)
