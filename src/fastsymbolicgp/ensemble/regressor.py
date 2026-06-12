import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

from fastsymbolicgp.estimators.regressor import FastSymbolicRegressor


class FastSymbolicEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble of independent FastSymbolicRegressor models.
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
        fitness_metric="r2",
        linear_scaling=True,
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
        self.fitness_metric = fitness_metric
        self.linear_scaling = linear_scaling
        self.backend = backend
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.models_ = []

        for i in range(self.n_estimators):
            if self.verbose:
                print(f"[FastSymbolicEnsembleRegressor] Training estimator {i + 1}/{self.n_estimators}")

            model = FastSymbolicRegressor(
                population_size=self.population_size,
                generations=self.generations,
                max_depth=self.max_depth,
                max_nodes=self.max_nodes,
                function_set=self.function_set,
                parsimony=self.parsimony,
                subsample=self.subsample,
                fitness_metric=self.fitness_metric,
                linear_scaling=self.linear_scaling,
                backend=self.backend,
                random_state=int(rng.integers(0, 2**31 - 1)),
                verbose=self.verbose,
            )
            model.fit(X, y)
            self.models_.append(model)

        return self

    def predict(self, X):
        preds = [model.predict(X) for model in self.models_]
        return np.mean(preds, axis=0)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def get_expressions(self, feature_names=None):
        return [m.get_expression(feature_names=feature_names) for m in self.models_]
