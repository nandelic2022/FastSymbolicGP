import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from fastsymbolicgp.estimators.classifier import FastSymbolicClassifier


class FastSymbolicEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble of independent FastSymbolicClassifier models.
    """

    def __init__(
        self,
        n_estimators=5,
        population_size=200,
        generations=20,
        max_depth=4,
        max_nodes=63,
        function_set="default",
        parsimony=0.001,
        subsample=0.8,
        threshold_metric="balanced_accuracy",
        backend="auto",
        random_state=None,
        verbose=1,
    ):
        self.n_estimators = n_estimators
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.function_set = function_set
        self.parsimony = parsimony
        self.subsample = subsample
        self.threshold_metric = threshold_metric
        self.backend = backend
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.models_ = []

        for i in range(self.n_estimators):
            if self.verbose:
                print(f"[FastSymbolicEnsembleClassifier] Training estimator {i + 1}/{self.n_estimators}")

            model = FastSymbolicClassifier(
                population_size=self.population_size,
                generations=self.generations,
                max_depth=self.max_depth,
                max_nodes=self.max_nodes,
                function_set=self.function_set,
                parsimony=self.parsimony,
                subsample=self.subsample,
                threshold_metric=self.threshold_metric,
                backend=self.backend,
                random_state=int(rng.integers(0, 2**31 - 1)),
                verbose=self.verbose,
            )
            model.fit(X, y)
            self.models_.append(model)

        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        probas = [model.predict_proba(X) for model in self.models_]
        return np.mean(probas, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_expressions(self, feature_names=None):
        return [m.get_expression(feature_names=feature_names) for m in self.models_]
