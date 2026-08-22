import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from fastsymbolicgp import FastSymbolicClassifier, FastSymbolicRegressor
from fastsymbolicgp.backend import (
    NUMBA_AVAILABLE,
    execute_population_postfix,
    score_population_classification_postfix,
    score_population_regression_postfix,
)
from fastsymbolicgp.metrics import (
    classification_loss,
    fit_linear_scaler,
    fit_logistic_scaler_approx,
    probability_from_raw,
    regression_loss,
)
from fastsymbolicgp.program import Node, SymbolicProgram


pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba acceleration is unavailable")


def _program_bank_classifier():
    X, y = load_breast_cancer(return_X_y=True)
    model = FastSymbolicClassifier(
        population_size=12, generations=1, tournament_size=3,
        init_depth=(1, 3), max_depth=5, max_nodes=47,
        validation_fraction=0.0, probability_calibration="none", patience=0,
        fitness_scaling="elite_logistic", fitness_scaling_top_k=0,
        evaluation_backend="population_numba", fitness_engine="materialized",
        dag_execution="off", n_jobs=2, random_state=19, verbose=0,
    ).fit(X[:180], y[:180])
    return X, y, [program.clone() for program in model._population[:8]]


def _program_bank_regressor():
    X, y = load_diabetes(return_X_y=True)
    model = FastSymbolicRegressor(
        population_size=12, generations=1, tournament_size=3,
        init_depth=(1, 3), max_depth=5, max_nodes=47,
        validation_fraction=0.0, patience=0,
        evaluation_backend="population_numba", fitness_engine="materialized",
        dag_execution="off", n_jobs=2, random_state=23, verbose=0,
    ).fit(X[:180], y[:180])
    return X, y, [program.clone() for program in model._population[:8]]


def test_fused_regression_matches_materialized_scoring():
    X, y, programs = _program_bank_regressor()
    X_train, y_train = X[:220], y[:220]
    X_val, y_val = X[220:280], y[220:280]
    weights_train = np.linspace(0.5, 1.5, len(y_train))
    weights_val = np.linspace(1.5, 0.5, len(y_val))

    raw_train = execute_population_postfix(X_train, programs, n_jobs=2)
    raw_val = execute_population_postfix(X_val, programs, n_jobs=2)
    reference = []
    for index in range(len(programs)):
        scale, intercept = fit_linear_scaler(raw_train[index], y_train, weights_train)
        reference.append((
            scale,
            intercept,
            regression_loss(y_train, scale * raw_train[index] + intercept, "rmse", weights_train),
            regression_loss(y_val, scale * raw_val[index] + intercept, "r2", weights_val),
        ))
    reference = np.asarray(reference)

    fused = score_population_regression_postfix(
        X_train, y_train, X_val, y_val, programs,
        metric="rmse", validation_metric="r2", scaling="linear",
        sample_weight_train=weights_train, sample_weight_val=weights_val,
        n_jobs=2,
    )
    observed = np.column_stack(fused)
    assert np.allclose(observed, reference, rtol=1e-9, atol=1e-9)


def test_fused_classifier_matches_approximate_materialized_scoring():
    X, y, programs = _program_bank_classifier()
    X_train, y_train = X[:260], y[:260]
    X_val, y_val = X[260:330], y[260:330]
    weights_train = np.linspace(0.75, 1.25, len(y_train))
    weights_val = np.linspace(1.25, 0.75, len(y_val))

    raw_train = execute_population_postfix(X_train, programs, n_jobs=2)
    raw_val = execute_population_postfix(X_val, programs, n_jobs=2)
    reference = []
    for index in range(len(programs)):
        scale, intercept = fit_logistic_scaler_approx(
            raw_train[index], y_train, weights_train, regularization=1e-4,
        )
        reference.append((
            scale,
            intercept,
            classification_loss(
                y_train, probability_from_raw(raw_train[index], scale, intercept),
                "log_loss", weights_train,
            ),
            classification_loss(
                y_val, probability_from_raw(raw_val[index], scale, intercept),
                "balanced_accuracy", weights_val,
            ),
        ))
    reference = np.asarray(reference)

    fused = score_population_classification_postfix(
        X_train, y_train, X_val, y_val, programs,
        metric="log_loss", validation_metric="balanced_accuracy",
        scaling="approx_logistic", regularization=1e-4,
        sample_weight_train=weights_train, sample_weight_val=weights_val,
        n_jobs=2,
    )
    observed = np.column_stack(fused)
    assert np.allclose(observed, reference, rtol=1e-9, atol=1e-9)


def test_fitness_cache_skips_unchanged_elites():
    X, y = load_breast_cancer(return_X_y=True)
    population_size = 20
    model = FastSymbolicClassifier(
        population_size=population_size, generations=3, tournament_size=4,
        init_depth=(1, 3), max_depth=5, max_nodes=47, elitism=4,
        probability_calibration="none", patience=0,
        fitness_scaling="elite_logistic", fitness_scaling_top_k=0,
        evaluation_backend="population_numba", fitness_engine="fused",
        evaluation_cache=True, fitness_cache_max_entries=1000,
        dag_execution="off", n_jobs=2, random_state=31, verbose=0,
    ).fit(X[:260], y[:260])

    assert model.history_[0]["fitness_cache_hits"] == 0
    assert any(row["fitness_cache_hits"] >= 4 for row in model.history_[1:])
    assert any(row["actual_program_evaluations"] < population_size for row in model.history_[1:])
    assert model.evaluation_count_ < population_size * model.n_generations_


def test_execution_hash_does_not_round_constants():
    first = SymbolicProgram(Node(kind="constant", value=1.0000000000001))
    second = SymbolicProgram(Node(kind="constant", value=1.0000000000002))
    assert first.execution_hash() != second.execution_hash()
