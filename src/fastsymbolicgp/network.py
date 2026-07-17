"""Experimental multilayer symbolic network estimators."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .transformer import FastSymbolicTransformer


class FastSymbolicNetworkClassifier(ClassifierMixin, BaseEstimator):
    """Experimental multilayer symbolic feature network.

    Each layer evolves a symbolic representation. A trainable logistic bridge
    converts that representation into class scores that are fed, together with
    the symbolic outputs, to the next layer. The final softmax head is trained
    numerically while every hidden representation remains exportable as
    equations.
    """

    def __init__(
        self,
        symbolic_layers=(16, 8),
        transformer_params=None,
        bridge_regularization=1.0,
        output_regularization=1.0,
        inherit_original_features=False,
        random_state=None,
        verbose=1,
    ):
        self.symbolic_layers = symbolic_layers
        self.transformer_params = transformer_params
        self.bridge_regularization = bridge_regularization
        self.output_regularization = output_regularization
        self.inherit_original_features = inherit_original_features
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.transformers_ = []
        self.scalers_ = []
        self.bridges_ = []
        current = X
        for index, width in enumerate(tuple(self.symbolic_layers)):
            params = dict(self.transformer_params or {})
            params.setdefault("random_state", None if self.random_state is None else int(self.random_state) + index * 10007)
            params.setdefault("verbose", max(0, int(self.verbose) - 1))
            transformer = FastSymbolicTransformer(
                n_components=int(width), task="classification",
                component_selection="mrmr", standardize_outputs=False,
                model_params=params,
                random_state=None if self.random_state is None else int(self.random_state) + index * 10007,
                verbose=max(0, int(self.verbose) - 1),
            )
            symbolic = transformer.fit_transform(current, y, sample_weight=sample_weight)
            scaler = StandardScaler().fit(symbolic)
            symbolic = scaler.transform(symbolic)
            self.transformers_.append(transformer)
            self.scalers_.append(scaler)
            if index < len(tuple(self.symbolic_layers)) - 1:
                bridge = LogisticRegression(
                    C=float(self.bridge_regularization), solver="lbfgs",
                    max_iter=1000, random_state=self.random_state,
                ).fit(symbolic, y, sample_weight=sample_weight)
                scores = bridge.predict_proba(symbolic)
                self.bridges_.append(bridge)
                pieces = [symbolic, scores]
                if self.inherit_original_features:
                    pieces.insert(0, current)
                current = np.column_stack(pieces)
            else:
                current = symbolic
        self.output_model_ = LogisticRegression(
            C=float(self.output_regularization), solver="lbfgs",
            max_iter=1000, random_state=self.random_state,
        ).fit(current, y, sample_weight=sample_weight)
        self.layer_expressions_ = [transformer.get_expressions() for transformer in self.transformers_]
        self.network_stats_ = {
            "layers": len(self.transformers_),
            "symbolic_widths": [len(x) for x in self.layer_expressions_],
            "equations_total": sum(len(x) for x in self.layer_expressions_),
            "dag_nodes_total": int(sum(program.dag_size for transformer in self.transformers_ for program in transformer.programs_)),
        }
        return self

    def _forward(self, X):
        current = np.asarray(X, dtype=float)
        for index, (transformer, scaler) in enumerate(zip(self.transformers_, self.scalers_)):
            symbolic = scaler.transform(transformer.transform(current))
            if index < len(self.bridges_):
                scores = self.bridges_[index].predict_proba(symbolic)
                pieces = [symbolic, scores]
                if self.inherit_original_features:
                    pieces.insert(0, current)
                current = np.column_stack(pieces)
            else:
                current = symbolic
        return current

    def predict_proba(self, X):
        check_is_fitted(self, "output_model_")
        return self.output_model_.predict_proba(self._forward(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_layer_expressions(self):
        check_is_fitted(self, "layer_expressions_")
        return [list(layer) for layer in self.layer_expressions_]

    def get_network_stats(self):
        check_is_fitted(self, "network_stats_")
        return dict(self.network_stats_)
