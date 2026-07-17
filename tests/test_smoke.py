import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from fastsymbolicgp import (
    FastSymbolicClassifier, FastSymbolicRegressor,
    FastSymbolicTransformer, __version__,
)


def small_classifier(**kwargs):
    params = dict(
        population_size=14, generations=3, tournament_size=4,
        init_depth=(1, 3), max_depth=6, max_nodes=63,
        patience=0, random_state=1, verbose=0,
        probability_calibration="none", evaluation_backend="numpy",
    )
    params.update(kwargs)
    return FastSymbolicClassifier(**params)


def test_version_and_clone():
    assert __version__ == "0.7.0"
    assert isinstance(clone(small_classifier()), FastSymbolicClassifier)


def test_binary_classifier_and_ensemble():
    X, y = load_breast_cancer(return_X_y=True)
    model = small_classifier(
        prediction_mode="symbolic_ensemble", ensemble_size=3,
        optimization="nsga2",
    ).fit(X[:220], y[:220])
    p = model.predict_proba(X[220:230])
    assert p.shape == (10, 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert 1 <= len(model.ensemble_programs_) <= 3
    assert model.get_expression_stats()["nodes"] <= 63
    assert len(model.pareto_front_) >= 1


def test_multiclass_ovr():
    X, y = load_iris(return_X_y=True)
    model = small_classifier(population_size=10, generations=2).fit(X, y)
    p = model.predict_proba(X[:12])
    assert p.shape == (12, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert len(model.estimators_) == 3
    assert len(model.get_expression()) == 3


def test_regressor_smoke():
    X, y = load_diabetes(return_X_y=True)
    model = FastSymbolicRegressor(
        population_size=14, generations=3, tournament_size=4,
        init_depth=(1, 3), max_depth=6, max_nodes=63,
        patience=0, random_state=1, verbose=0,
        evaluation_backend="numpy",
    ).fit(X[:180], y[:180])
    pred = model.predict(X[180:190])
    assert pred.shape == (10,)
    assert np.isfinite(pred).all()


def test_numba_numpy_equivalence():
    X, y = load_breast_cancer(return_X_y=True)
    model = small_classifier(evaluation_backend="numpy").fit(X[:200], y[:200])
    p = model.best_program_
    a = p.execute(np.ascontiguousarray(X[200:240]), backend="numpy")
    b = p.execute(np.ascontiguousarray(X[200:240]), backend="numba")
    assert np.allclose(a, b, rtol=1e-10, atol=1e-10)


def test_checkpoint_resume(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    path = tmp_path / "checkpoint"
    first = small_classifier(generations=2, checkpoint_path=str(path), checkpoint_interval=1).fit(X[:220], y[:220])
    assert first.n_generations_ == 2
    resumed = small_classifier(
        generations=4, checkpoint_path=str(path), checkpoint_interval=1,
        resume_from_checkpoint=True,
    ).fit(X[:220], y[:220])
    assert resumed.n_generations_ == 4
    assert resumed.history_[-1]["generation"] == 4


def test_symbolic_transformer():
    X, y = load_breast_cancer(return_X_y=True)
    transformer = FastSymbolicTransformer(
        n_components=3, random_state=2, verbose=0,
        model_params={
            "population_size": 14, "generations": 3, "tournament_size": 4,
            "init_depth": (1, 3), "max_depth": 6, "max_nodes": 63,
            "patience": 0, "evaluation_backend": "numpy",
        },
    ).fit(X[:220], y[:220])
    Z = transformer.transform(X[220:230])
    assert Z.shape[0] == 10
    assert 1 <= Z.shape[1] <= 3
    assert len(transformer.get_expressions()) == Z.shape[1]
