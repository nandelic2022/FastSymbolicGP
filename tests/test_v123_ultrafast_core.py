import time
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from fastsymbolicgp import FastSymbolicRegressor
from fastsymbolicgp.backend import score_population_regression_postfix
from fastsymbolicgp.program import Program, Node


def _split():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.20, random_state=42)


def test_single_pass_rmse_matches_closed_form_reference():
    Xtr, Xte, ytr, yte = _split()
    p = Program(Node("func", "add", (Node("feature", 2), Node("func", "mul", (Node("const", 0.5), Node("feature", 3))))))
    scale, intercept, tr, va = score_population_regression_postfix(
        Xtr.to_numpy(), ytr.to_numpy(), Xte.to_numpy(), yte.to_numpy(), [p],
        metric="rmse", validation_metric="rmse", scaling="linear", n_jobs=1,
    )
    raw = p.root.evaluate(Xtr.to_numpy())
    w = np.ones(len(raw))
    mx = np.average(raw, weights=w); my = np.average(ytr.to_numpy(), weights=w)
    denom = np.sum(w * (raw-mx)**2)
    a = 0.0 if denom <= 1e-18 else np.sum(w*(raw-mx)*(ytr.to_numpy()-my))/denom
    b = my-a*mx
    ref = np.sqrt(np.mean((a*raw+b-ytr.to_numpy())**2))
    assert np.isclose(scale[0], a, rtol=1e-10, atol=1e-10)
    assert np.isclose(intercept[0], b, rtol=1e-10, atol=1e-10)
    assert np.isclose(tr[0], ref, rtol=1e-10, atol=1e-10)


def test_quality_preserving_fast_tournament_profile():
    Xtr, _, ytr, _ = _split()
    m = FastSymbolicRegressor(
        population_size=120, generations=5, function_set=("add","sub","mul","div"),
        max_depth=8, max_nodes=60, evaluation_backend="numba", fitness_engine="fused",
        optimization="scalar", semantic_diversity=False, age_management="off",
        memetic_search="off", n_jobs=1, random_state=42, verbose=0,
        constant_optimization="off", uncertainty="none", patience=1000,
    ).fit(Xtr, ytr)
    assert m.profile_["variation_engine"] in {"flat_genome_inheritance_v1","numba_rootless_genome_v2"}
    assert m.profile_["tournament_sampling"] in {"without_replacement_exact_fast","batched_exact_distribution"}
    assert m.execution_plan_["fused_training_passes"] == 1
    assert m.execution_plan_["fused_single_pass_squared_error"] is True


def test_diabetes_speed_quality_smoke():
    Xtr, Xte, ytr, yte = _split()
    # warm JIT
    FastSymbolicRegressor(population_size=20, generations=1, evaluation_backend="numba",
        fitness_engine="fused", optimization="scalar", random_state=1, verbose=0,
        constant_optimization="off", uncertainty="none").fit(Xtr, ytr)
    m = FastSymbolicRegressor(
        population_size=200, generations=15, function_set=("add","sub","mul","div"),
        max_depth=8, max_nodes=60, evaluation_backend="numba", fitness_engine="fused",
        optimization="scalar", semantic_diversity=False, age_management="off",
        memetic_search="off", n_jobs=-1, random_state=42, verbose=0,
        constant_optimization="champion", patience=1000,
    )
    t0=time.perf_counter(); m.fit(Xtr,ytr); elapsed=time.perf_counter()-t0
    score=r2_score(yte,m.predict(Xte))
    baseline=r2_score(yte,LinearRegression().fit(Xtr,ytr).predict(Xte))
    assert elapsed < 20.0
    assert score > 0.20
    assert m.profile_["actual_program_evaluations"] > 0
    assert baseline > 0.40
