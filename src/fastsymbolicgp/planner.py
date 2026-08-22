"""Execution planning for FastSymbolicGP V1.0.0.

The planner is intentionally conservative: it only mutates execution settings
when the user requested ``preset='auto'``. For named presets and custom
configurations it records recommendations but preserves the resolved settings.
"""
from __future__ import annotations

import os


def build_execution_plan(estimator, X) -> dict:
    n_samples, n_features = map(int, X.shape)
    population = int(getattr(estimator, "population_size", 1))
    depth = int(getattr(estimator, "max_depth", 1))
    cpus = max(1, int(os.cpu_count() or 1))
    work = int(n_samples * max(population, 1) * max(1, min(depth, 16)))

    if n_samples <= 5_000 and n_features <= 128:
        workload = "small"
        recommended_islands = 1
        recommended_parallel = False
        recommended_dashboard_interval = max(5, int(getattr(estimator, "dashboard_interval", 1)))
    elif n_samples <= 100_000 and n_features <= 512:
        workload = "medium"
        recommended_islands = min(2, cpus)
        recommended_parallel = recommended_islands > 1
        recommended_dashboard_interval = max(2, int(getattr(estimator, "dashboard_interval", 1)))
    else:
        workload = "large"
        recommended_islands = min(4, cpus)
        recommended_parallel = recommended_islands > 1
        recommended_dashboard_interval = max(5, int(getattr(estimator, "dashboard_interval", 1)))

    requested_preset = str(getattr(estimator, "preset", "none") or "none").lower()
    planner_mode = str(getattr(estimator, "execution_planner", "auto") or "auto").lower()
    applied = False
    if planner_mode not in {"off", "none", "disabled"} and requested_preset == "auto":
        estimator.n_islands = recommended_islands
        estimator.island_parallel = recommended_parallel
        estimator.evolution_model = "islands" if recommended_islands > 1 else "panmictic"
        estimator.dashboard_interval = recommended_dashboard_interval
        applied = True

    plan = {
        "workload": workload,
        "n_samples": n_samples,
        "n_features": n_features,
        "estimated_symbolic_work": work,
        "cpu_count": cpus,
        "recommended_n_islands": recommended_islands,
        "recommended_island_parallel": recommended_parallel,
        "recommended_dashboard_interval": recommended_dashboard_interval,
        "applied": applied,
        "resolved_n_islands": int(getattr(estimator, "n_islands", 1)),
        "resolved_island_parallel": bool(getattr(estimator, "island_parallel", False)),
        "resolved_evolution_model": str(getattr(estimator, "evolution_model", "panmictic")),
        "resolved_evaluation_backend": str(getattr(estimator, "evaluation_backend", "auto")),
        "resolved_fitness_engine": str(getattr(estimator, "fitness_engine", "auto")),
    }
    estimator.execution_plan_ = plan
    return plan
