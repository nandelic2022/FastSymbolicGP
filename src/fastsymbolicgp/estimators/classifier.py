import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from fastsymbolicgp.estimators.base import BaseSymbolicEvolution
from fastsymbolicgp.core.evaluator import evaluate_program
from fastsymbolicgp.core.fitness import safe_corr
from fastsymbolicgp.core.thresholds import optimize_binary_threshold


class FastSymbolicClassifier(BaseSymbolicEvolution, BaseEstimator, ClassifierMixin):
    """
    Fast symbolic binary classifier.

    The symbolic program produces a continuous score.
    A threshold and direction are optimized after training.
    """

    def __init__(
        self,
        population_size=300,
        generations=30,
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
        super().__init__(
            population_size=population_size,
            generations=generations,
            max_depth=max_depth,
            max_nodes=max_nodes,
            init_method=init_method,
            function_set=function_set,
            elite_fraction=elite_fraction,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            subtree_mutation_rate=subtree_mutation_rate,
            point_mutation_rate=point_mutation_rate,
            hoist_mutation_rate=hoist_mutation_rate,
            const_range=const_range,
            parsimony=parsimony,
            subsample=subsample,
            backend=backend,
            cache_programs=cache_programs,
            random_state=random_state,
            verbose=verbose,
        )
        self.threshold_metric = threshold_metric

    def _prepare_y(self, y):
        y = np.asarray(y)
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError(
                "FastSymbolicClassifier supports binary classification only. "
                "Use FastSymbolicMultiClassifier for multiclass tasks."
            )

        self.classes_ = classes
        y_binary = np.zeros(y.shape[0], dtype=np.int64)
        y_binary[y == classes[1]] = 1
        y_signed = np.where(y_binary == 1, 1.0, -1.0).astype(np.float64)
        return y_binary, y_signed

    def _fitness_from_scores(self, scores, y_target):
        return abs(safe_corr(scores, y_target))

    def fit(self, X, y):
        X = self._prepare_X(X)
        y_binary, y_signed = self._prepare_y(y)
        self.n_features_in_ = X.shape[1]

        self._evolve(X, y_signed)

        train_scores = evaluate_program(
            self.best_program_.ops,
            self.best_program_.args,
            X,
            backend=self.backend,
        )

        self.threshold_, self.direction_, self.train_threshold_score_ = optimize_binary_threshold(
            train_scores,
            y_binary,
            metric=self.threshold_metric,
        )

        if self.verbose:
            print(f"[{self.__class__.__name__}] Training complete")
            print(f"Expression: {self.get_expression()}")
            print(f"Threshold: {self.threshold_:.6f}")
            print(f"Direction: {self.direction_}")
            print(f"Train threshold score: {self.train_threshold_score_:.6f}")

        return self

    def decision_function(self, X):
        X = self._prepare_X(X)
        raw = evaluate_program(self.best_program_.ops, self.best_program_.args, X, backend=self.backend)
        return self.direction_ * (raw - self.threshold_)

    def predict(self, X):
        aligned = self.decision_function(X)
        y_binary = (aligned >= 0.0).astype(np.int64)
        return self.classes_[y_binary]

    def predict_proba(self, X):
        aligned = self.decision_function(X)
        aligned = np.clip(aligned, -60.0, 60.0)
        p1 = 1.0 / (1.0 + np.exp(-aligned))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
