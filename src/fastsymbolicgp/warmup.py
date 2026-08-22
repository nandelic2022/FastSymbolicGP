"""Optional Numba/JIT warm-up helpers for FastSymbolicGP V1.0.0."""
from __future__ import annotations

import time
import numpy as np


def warmup(task: str = "all") -> dict:
    """Compile common FastSymbolicGP kernels before a timed experiment.

    Parameters
    ----------
    task : {"all", "classification", "regression"}
        Which estimator kernels to exercise.
    """
    task = str(task).lower()
    if task not in {"all", "classification", "regression"}:
        raise ValueError("task must be all, classification, or regression")
    result = {}
    start = time.perf_counter()
    rng = np.random.default_rng(12345)
    X = np.ascontiguousarray(rng.normal(size=(48, 6)), dtype=np.float64)

    if task in {"all", "classification"}:
        from .classifier import FastSymbolicClassifier
        y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(int)
        model = FastSymbolicClassifier(
            population_size=8, generations=1, tournament_size=2,
            validation_fraction=0.2, fitness_scaling="fixed_sigmoid",
            prediction_mode="best", n_islands=1, evolution_model="panmictic",
            evaluation_backend="population_numba", fitness_engine="fused",
            display="off", verbose=0, preset=None, random_state=12345,
        )
        t0 = time.perf_counter(); model.fit(X, y)
        result["classification_seconds"] = time.perf_counter() - t0

    if task in {"all", "regression"}:
        from .regressor import FastSymbolicRegressor
        y = 1.5 * X[:, 0] - 0.7 * X[:, 2] + 0.1
        model = FastSymbolicRegressor(
            population_size=8, generations=1, tournament_size=2,
            validation_fraction=0.2, n_islands=1, evolution_model="panmictic",
            evaluation_backend="population_numba", fitness_engine="fused",
            display="off", verbose=0, preset=None, random_state=12345,
        )
        t0 = time.perf_counter(); model.fit(X, y)
        result["regression_seconds"] = time.perf_counter() - t0

    result["total_seconds"] = time.perf_counter() - start
    result["status"] = "ok"
    return result
