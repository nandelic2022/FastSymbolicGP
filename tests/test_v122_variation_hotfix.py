import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from fastsymbolicgp import FastSymbolicRegressor
from fastsymbolicgp.program import Program, Node, subtree_crossover


def _data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.20, random_state=42)[:2] + train_test_split(X, y, test_size=0.20, random_state=42)[2:]


def test_paths_are_immutable_cached_tuple():
    p = Program(Node("func", "add", (Node("feature", 0), Node("feature", 1))))
    a = p.paths()
    b = p.paths()
    assert isinstance(a, tuple)
    assert a is b
    assert len(a) == p.size


def test_shared_crossover_does_not_change_parents():
    rng = np.random.default_rng(7)
    a = Program(Node("func", "add", (Node("feature", 0), Node("const", 1.0))))
    b = Program(Node("func", "mul", (Node("feature", 1), Node("const", 2.0))))
    sa, sb = a.to_string(), b.to_string()
    _ = subtree_crossover(a, b, rng)
    assert a.to_string() == sa
    assert b.to_string() == sb


def test_dataframe_fit_has_no_feature_name_warning():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.20, random_state=42)
    model = FastSymbolicRegressor(
        population_size=30, generations=2,
        function_set=("add", "sub", "mul", "div"),
        evaluation_backend="numba", fitness_engine="fused",
        optimization="scalar", random_state=42, verbose=0,
        constant_optimization="off", uncertainty="none",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Xtr, ytr)
    assert not any("valid feature names" in str(w.message) for w in caught)


def test_fast_variation_benchmark_smoke():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.20, random_state=42)

    # Warm JIT so compilation is not charged to the timed benchmark.
    FastSymbolicRegressor(
        population_size=20, generations=1,
        evaluation_backend="numba", fitness_engine="fused",
        optimization="scalar", random_state=1, verbose=0,
        constant_optimization="off", uncertainty="none",
    ).fit(Xtr, ytr)

    model = FastSymbolicRegressor(
        population_size=200, generations=10,
        function_set=("add", "sub", "mul", "div"),
        max_depth=8, max_nodes=60,
        fitness_metric="rmse", fitness_scaling="linear",
        validation_fraction=0.20,
        evaluation_backend="numba", fitness_engine="fused",
        optimization="scalar",
        semantic_diversity=False, age_management="off", memetic_search="off",
        n_jobs=-1, random_state=42, verbose=0,
        patience=1000, constant_optimization="off", uncertainty="none",
    )
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0

    assert model.resolved_config_["evaluation_backend"] == "numba"
    assert model.resolved_config_["fitness_engine"] == "fused"
    assert model.profile_["nsga2_calls"] == 0
    assert model.profile_["variation_engine"] in {"flat_genome_inheritance_v1","numba_rootless_genome_v2"}
    assert model.profile_["tournament_sampling"] in {"without_replacement_exact_fast","batched_exact_distribution"}
    assert model.profile_["actual_program_evaluations"] > 0
    # Deliberately generous: catches catastrophic regressions, not hardware differences.
    assert elapsed < 15.0
    assert model.profile_["stage_seconds"]["variation"] < 12.0
