import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

from fastsymbolicgp.estimators.base import BaseSymbolicEvolution
from fastsymbolicgp.core.evaluator import evaluate_program
from fastsymbolicgp.core.fitness import (
    safe_corr,
    fit_linear_scaling,
    r2_score_fast,
    rmse_fast,
    mae_fast,
)


class FastSymbolicRegressor(BaseSymbolicEvolution, BaseEstimator, RegressorMixin):
    """
    Fast symbolic regressor.

    The symbolic expression produces a raw score.
    Linear scaling y ~= a * score + b is fitted automatically.
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
        fitness_metric="r2",
        linear_scaling=True,
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
        self.fitness_metric = fitness_metric
        self.linear_scaling = linear_scaling

    def _prepare_y_reg(self, y):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        return y

    def _fitness_from_scores(self, scores, y_target):
        if self.linear_scaling:
            a, b = fit_linear_scaling(scores, y_target)
            pred = a * scores + b
        else:
            pred = scores

        if self.fitness_metric == "r2":
            return r2_score_fast(y_target, pred)

        if self.fitness_metric == "corr":
            return abs(safe_corr(pred, y_target))

        if self.fitness_metric == "neg_rmse":
            return -rmse_fast(y_target, pred)

        if self.fitness_metric == "neg_mae":
            return -mae_fast(y_target, pred)

        raise ValueError("fitness_metric must be one of: 'r2', 'corr', 'neg_rmse', 'neg_mae'.")

    def fit(self, X, y):
        X = self._prepare_X(X)
        y = self._prepare_y_reg(y)

        self.n_features_in_ = X.shape[1]
        self._evolve(X, y)

        train_scores = evaluate_program(
            self.best_program_.ops,
            self.best_program_.args,
            X,
            backend=self.backend,
        )

        if self.linear_scaling:
            self.scale_, self.intercept_ = fit_linear_scaling(train_scores, y)
        else:
            self.scale_, self.intercept_ = 1.0, 0.0

        if self.verbose:
            print(f"[{self.__class__.__name__}] Training complete")
            print(f"Expression: {self.get_expression()}")
            print(f"Scale: {self.scale_:.6f}")
            print(f"Intercept: {self.intercept_:.6f}")

        return self

    def predict(self, X):
        X = self._prepare_X(X)
        scores = evaluate_program(self.best_program_.ops, self.best_program_.args, X, backend=self.backend)
        return self.scale_ * scores + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))
