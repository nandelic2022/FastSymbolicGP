import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from fastsymbolicgp import FastSymbolicClassifier
from fastsymbolicgp.backend import NUMBA_AVAILABLE, execute_population_postfix


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba acceleration is unavailable")
def test_population_numba_matches_individual_numba():
    X, y = load_breast_cancer(return_X_y=True)
    model = FastSymbolicClassifier(
        population_size=12, generations=1, tournament_size=3,
        init_depth=(1, 3), max_depth=5, max_nodes=47,
        probability_calibration="none", patience=0,
        evaluation_backend="population_numba", dag_execution="off",
        semantic_duplicate_elimination=False, batch_size="full",
        n_jobs=2, random_state=9, verbose=0,
    ).fit(X[:180], y[:180])
    programs = model._population[:8]
    reference = np.vstack([
        program.execute(np.ascontiguousarray(X[180:210]), backend="numba")
        for program in programs
    ])
    packed = execute_population_postfix(X[180:210], programs, n_jobs=2)
    assert packed.shape == reference.shape
    assert np.allclose(packed, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba acceleration is unavailable")
def test_elite_logistic_refines_only_top_k():
    X, y = load_breast_cancer(return_X_y=True)
    model = FastSymbolicClassifier(
        population_size=16, generations=2, tournament_size=4,
        init_depth=(1, 3), max_depth=5, max_nodes=47,
        probability_calibration="none", patience=0,
        evaluation_backend="population_numba", dag_execution="off",
        semantic_duplicate_elimination=False, batch_size=5,
        fitness_scaling="elite_logistic", fitness_scaling_top_k=3,
        fitness_scaling_interval=1, n_jobs=2,
        random_state=4, verbose=0,
    ).fit(X[:220], y[:220])
    assert model.execution_engine_ == "population_numba"
    assert model.history_[-1]["scaling_exact_programs"] == 3
    assert model.history_[-1]["scaling_approx_programs"] == 13
    assert all(
        program.metadata_.get("evaluation_engine") == "population_numba"
        for program in model._population
    )


def test_progressive_sampling_increases_training_fraction():
    X, y = load_breast_cancer(return_X_y=True)
    model = FastSymbolicClassifier(
        population_size=10, generations=3, tournament_size=3,
        init_depth=(1, 2), max_depth=4, max_nodes=31,
        probability_calibration="none", patience=0,
        evaluation_backend="numpy", dag_execution="off",
        semantic_duplicate_elimination=False,
        max_samples=(0.25, 1.0), random_state=3, verbose=0,
    ).fit(X[:240], y[:240])
    counts = [row["training_samples_used"] for row in model.history_]
    assert counts[0] < counts[-1]
    assert counts == sorted(counts)


def test_turbo_preset_selects_performance_options():
    X, y = load_breast_cancer(return_X_y=True)
    model = FastSymbolicClassifier(preset="turbo", verbose=0)
    model._apply_preset(X, y)
    assert model.evaluation_backend == "population_numba"
    assert model.n_jobs == -1
    assert model.semantic_duplicate_elimination is False
    assert model.dag_execution == "off"
    assert isinstance(model.max_samples, tuple)
