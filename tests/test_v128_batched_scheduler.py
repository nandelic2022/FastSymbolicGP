import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from fastsymbolicgp import FastSymbolicRegressor
from fastsymbolicgp.base import _tournament_min_cdf, _BatchedTournamentScheduler


def _data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.20, random_state=42)


def _model(**extra):
    cfg = dict(
        population_size=120, generations=6,
        function_set=("add", "sub", "mul", "div"),
        max_depth=8, max_nodes=60,
        fitness_metric="rmse", fitness_scaling="linear",
        validation_fraction=0.20,
        evaluation_backend="numba", fitness_engine="fused",
        optimization="scalar", semantic_diversity=False,
        age_management="off", memetic_search="off",
        n_jobs=1, random_state=42, verbose=0,
        patience=1000, constant_optimization="off", uncertainty="none",
    )
    cfg.update(extra)
    return FastSymbolicRegressor(**cfg)


def test_tournament_cdf_is_valid_exact_distribution():
    n, k = 500, 20
    cdf = _tournament_min_cdf(n, k)
    assert len(cdf) == n - k + 1
    assert np.all(np.diff(cdf) >= 0)
    assert abs(float(cdf[-1]) - 1.0) < 1e-15
    # P(winner == 0) = k/n for best-of-k without replacement.
    assert abs(float(cdf[0]) - k / n) < 1e-15


def test_batched_scheduler_is_reproducible():
    a = _BatchedTournamentScheduler(np.random.default_rng(123), 500, 20, 128)
    b = _BatchedTournamentScheduler(np.random.default_rng(123), 500, 20, 128)
    xa = [a.next() for _ in range(1000)]
    xb = [b.next() for _ in range(1000)]
    assert xa == xb


def test_repeated_fit_same_seed_is_identical():
    Xtr, _, ytr, _ = _data()
    a = _model().fit(Xtr, ytr)
    b = _model().fit(Xtr, ytr)
    assert a.get_raw_expression() == b.get_raw_expression()
    assert np.allclose(a.predict(Xtr), b.predict(Xtr), rtol=0, atol=0)
    assert a.profile_["tournament_sampling"] == "batched_exact_distribution"
    assert a.profile_["tournament_scheduler"] == "vectorized_cdf_v1"


def test_checkpoint_persists_selection_rng_state(tmp_path):
    Xtr, _, ytr, _ = _data()
    path = tmp_path / "resume.joblib"
    m = _model(generations=3, checkpoint_path=str(path), checkpoint_interval=1).fit(Xtr, ytr)
    assert path.exists()
    saved = FastSymbolicRegressor.load_checkpoint(path)
    state = saved.evolution_state_
    assert state.get("selection_rng_state") is not None
