import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from fastsymbolicgp.estimators.classifier import FastSymbolicClassifier


class FastSymbolicMultiClassifier(BaseEstimator, ClassifierMixin):
    """
    One-vs-rest multiclass symbolic classifier.

    Internally trains one FastSymbolicClassifier per class.
    """

    def __init__(
        self,
        population_size=200,
        generations=20,
        max_depth=4,
        max_nodes=63,
        init_method="grow",
        function_set="default",
        elite_fraction=0.10,
        tournament_size=5,
        crossover_rate=0.45,
        subtree_mutation_rate=0.25,
        point_mutation_rate=0.20,
        hoist_mutation_rate=0.05,
        const_range=(-1.0, 1.0),
        parsimony=0.001,
        subsample=1.0,
        threshold_metric="balanced_accuracy",
        backend="auto",
        cache_programs=True,
        random_state=None,
        verbose=1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.init_method = init_method
        self.function_set = function_set
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.subtree_mutation_rate = subtree_mutation_rate
        self.point_mutation_rate = point_mutation_rate
        self.hoist_mutation_rate = hoist_mutation_rate
        self.const_range = const_range
        self.parsimony = parsimony
        self.subsample = subsample
        self.threshold_metric = threshold_metric
        self.backend = backend
        self.cache_programs = cache_programs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.models_ = []

        rng = np.random.default_rng(self.random_state)

        for i, cls in enumerate(self.classes_):
            if self.verbose:
                print(f"[FastSymbolicMultiClassifier] Training one-vs-rest model for class {cls}")

            y_binary = (y == cls).astype(np.int64)

            model = FastSymbolicClassifier(
                population_size=self.population_size,
                generations=self.generations,
                max_depth=self.max_depth,
                max_nodes=self.max_nodes,
                init_method=self.init_method,
                function_set=self.function_set,
                elite_fraction=self.elite_fraction,
                tournament_size=self.tournament_size,
                crossover_rate=self.crossover_rate,
                subtree_mutation_rate=self.subtree_mutation_rate,
                point_mutation_rate=self.point_mutation_rate,
                hoist_mutation_rate=self.hoist_mutation_rate,
                const_range=self.const_range,
                parsimony=self.parsimony,
                subsample=self.subsample,
                threshold_metric=self.threshold_metric,
                backend=self.backend,
                cache_programs=self.cache_programs,
                random_state=int(rng.integers(0, 2**31 - 1)),
                verbose=self.verbose,
            )

            model.fit(X, y_binary)
            self.models_.append(model)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = []
        for model in self.models_:
            scores.append(model.decision_function(X))
        return np.column_stack(scores)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(scores, -60.0, 60.0))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X):
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_expressions(self, feature_names=None):
        return {
            str(cls): model.get_expression(feature_names=feature_names)
            for cls, model in zip(self.classes_, self.models_)
        }
