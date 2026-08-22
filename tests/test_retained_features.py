import importlib.util
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris

from fastsymbolicgp import (
    FastSymbolicClassifier,
    FastSymbolicNetworkClassifier,
    FastSymbolicTransformer,
)


def tiny(**kwargs):
    params = dict(
        population_size=12, generations=2, tournament_size=3,
        init_depth=(1, 3), max_depth=5, max_nodes=47,
        patience=0, random_state=7, verbose=0,
        probability_calibration="none", evaluation_backend="numpy",
        subtree_cache=True,
    )
    params.update(kwargs)
    return FastSymbolicClassifier(**params)


def test_island_dag_cache_and_profile():
    X, y = load_breast_cancer(return_X_y=True)
    model = tiny(
        population_size=16, evolution_model="islands", n_islands=2,
        migration_interval=1, migration_size=1, n_jobs=1,
        optimization="nsga2",
    ).fit(X[:220], y[:220])
    assert len(model.island_populations_) == 2
    assert model.migration_count_ >= 2
    assert model.get_expression_stats()["dag_nodes"] <= model.get_expression_stats()["nodes"]
    assert "hit_rate" in model.cache_statistics_
    profile = model.profile_prediction(X[220:230], repeats=2, warmup=0)
    assert profile["microseconds_per_sample"] >= 0


def test_shared_softmax_reporting_and_temperature():
    X, y = load_iris(return_X_y=True)
    model = tiny(
        population_size=9, generations=2,
        multiclass_strategy="shared_softmax", shared_n_components=4,
        multiclass_calibration="temperature",
    ).fit(X, y)
    p = model.predict_proba(X[:8])
    assert p.shape == (8, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    stats = model.get_expression_stats()
    assert stats["generations_max_per_class"] <= 2
    assert stats["generations_total"] <= 6
    assert stats["dag_nodes_total"] > 0
    assert model.multiclass_calibration_report_["selected_method"] == "temperature"


def test_native_missing_values():
    X, y = load_breast_cancer(return_X_y=True)
    X = X[:180].copy(); y = y[:180]
    X[::7, 2] = np.nan
    model = tiny(
        function_set=("add", "sub", "mul", "div", "is_missing", "coalesce"),
        missing_value_strategy="native",
    ).fit(X, y)
    p = model.predict_proba(X[:5])
    assert np.isfinite(p).all()


def test_python_export_consistency(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    model = tiny(prediction_mode="symbolic_ensemble", ensemble_size=2).fit(X[:220], y[:220])
    path = tmp_path / "exported_model.py"
    model.export_python(path)
    spec = importlib.util.spec_from_file_location("exported_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = model.predict_proba(X[220:230])
    exported = module.predict_proba(X[220:230])
    assert np.allclose(actual, exported, atol=1e-9, rtol=1e-9)


def test_distillation_and_branch():
    X, y = load_breast_cancer(return_X_y=True)
    teacher = tiny(prediction_mode="symbolic_ensemble", ensemble_size=2).fit(X[:220], y[:220])
    distilled = teacher.distill(
        X[:220], max_nodes=23, generations=2, population_size=12,
        verbose=0, evaluation_backend="numpy", patience=0,
    )
    assert distilled.predict_proba(X[220:225]).shape == (5, 2)
    assert distilled.get_expression_stats()["nodes_total"] <= 23
    branch = teacher.branch(parsimony_target_nodes=10)
    assert branch.parsimony_target_nodes == 10
    assert branch.get_expression() == teacher.get_expression()


def test_transformer_v2_and_symbolic_network():
    X, y = load_iris(return_X_y=True)
    transformer = FastSymbolicTransformer(
        n_components=3, component_selection="mrmr",
        include_original_features=True, random_state=3, verbose=0,
        model_params=dict(
            population_size=8, generations=1, tournament_size=3,
            init_depth=(1,2), max_depth=4, max_nodes=31,
            patience=0, probability_calibration="none",
            evaluation_backend="numpy", verbose=0,
        ),
    ).fit(X, y)
    Z = transformer.transform(X[:4])
    assert Z.shape[1] == X.shape[1] + transformer.n_components_
    network = FastSymbolicNetworkClassifier(
        symbolic_layers=(2,), random_state=2, verbose=0,
        transformer_params=dict(
            population_size=8, generations=1, tournament_size=3,
            init_depth=(1,2), max_depth=4, max_nodes=31,
            patience=0, evaluation_backend="numpy", verbose=0,
        ),
    ).fit(X, y)
    assert network.predict_proba(X[:5]).shape == (5, 3)
    assert network.get_network_stats()["layers"] == 1


def test_resource_budget_stop():
    X, y = load_breast_cancer(return_X_y=True)
    model = tiny(generations=50, time_budget=0.001).fit(X[:180], y[:180])
    assert model.stop_reason_ == "time_budget"
    assert model.n_generations_ < 50


def test_shared_softmax_python_export_consistency(tmp_path):
    X, y = load_iris(return_X_y=True)
    model = tiny(
        population_size=9, generations=2,
        multiclass_strategy="shared_softmax", shared_n_components=4,
        multiclass_calibration="temperature",
    ).fit(X, y)
    path = tmp_path / "shared_export.py"
    model.export_python(path)
    spec = importlib.util.spec_from_file_location("shared_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = model.predict_proba(X[:12])
    exported = module.predict_proba(X[:12])
    assert np.allclose(actual, exported, atol=1e-10, rtol=1e-10)
