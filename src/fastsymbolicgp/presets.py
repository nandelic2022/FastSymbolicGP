"""Task-aware parameter presets for FastSymbolicGP V1.0.0."""
from __future__ import annotations

import os
import inspect


PRESETS = {
    # Reproduces the conservative V0.7.0 search behaviour while using the new
    # packed execution engine when Numba is available.
    "compatibility": {
        "evaluation_backend": "auto", "dag_execution": "auto",
        "fitness_engine": "materialized", "evaluation_cache": False,
        "fitness_scaling": "exact_logistic", "n_jobs": -1,
        "semantic_duplicate_elimination": True,
    },
    # Strong default: population Numba plus elite-only iterative scaling.
    "fast": {
        "population_size": 48, "generations": 80, "max_depth": 10,
        "max_nodes": 127, "ensemble_size": 3, "n_islands": 1,
        "parsimony_target_nodes": 32, "evaluation_backend": "auto",
        "fitness_engine": "fused", "evaluation_cache": True,
        "dag_execution": "auto", "n_jobs": -1,
        "fitness_scaling": "elite_logistic", "fitness_scaling_top_k": 12,
        "fitness_scaling_interval": 2,
        "semantic_duplicate_elimination": False,
    },
    "turbo": {
        "population_size": 48, "generations": 60, "max_depth": 9,
        "max_nodes": 95, "ensemble_size": 1, "prediction_mode": "best",
        "parsimony_target_nodes": 24, "evaluation_backend": "population_numba",
        "fitness_engine": "fused", "evaluation_cache": True,
        "dag_execution": "off", "subtree_cache": False, "n_jobs": -1,
        "fitness_scaling": "elite_logistic", "fitness_scaling_top_k": 8,
        "fitness_scaling_interval": 4, "max_samples": (0.20, 1.0),
        "semantic_duplicate_elimination": False, "low_memory": True,
    },
    "balanced": {
        "population_size": 64, "generations": 120, "max_depth": 14,
        "max_nodes": 255, "ensemble_size": 5, "n_islands": 2,
        "parsimony_target_nodes": 45, "evaluation_backend": "auto",
        "fitness_engine": "fused", "evaluation_cache": True,
        "fitness_scaling": "elite_logistic", "fitness_scaling_interval": 2,
        "n_jobs": -1,
    },
    "accurate": {
        "population_size": 96, "generations": 200, "max_depth": 16,
        "max_nodes": 383, "ensemble_size": 9, "n_islands": 4,
        "parsimony_target_nodes": 65, "evaluation_backend": "auto",
        "fitness_engine": "fused", "evaluation_cache": True,
        "fitness_scaling": "elite_logistic", "fitness_scaling_top_k": 20,
        "fitness_scaling_interval": 1, "n_jobs": -1,
        "constant_optimization": "champion", "constant_optimization_rounds": 2,
    },
    "interpretable": {
        "population_size": 64, "generations": 140, "max_depth": 9,
        "max_nodes": 95, "ensemble_size": 1, "prediction_mode": "best",
        "parsimony_target_nodes": 24, "selection_tolerance": 0.03,
        "evaluation_backend": "auto", "fitness_engine": "fused",
        "evaluation_cache": True, "n_jobs": -1,
    },
    "large_dataset": {
        "population_size": 48, "generations": 100, "max_depth": 10,
        "max_nodes": 127, "max_samples": (0.15, 0.60), "ensemble_size": 3,
        "parsimony_target_nodes": 32, "evaluation_backend": "population_numba",
        "fitness_engine": "fused", "evaluation_cache": True,
        "dag_execution": "off", "subtree_cache": False,
        "fitness_scaling": "elite_logistic", "fitness_scaling_top_k": 8,
        "fitness_scaling_interval": 4, "semantic_duplicate_elimination": False,
        "n_jobs": -1, "low_memory": True,
    },
    "superfast": {
        "population_size": 64, "generations": 80, "max_depth": 10,
        "max_nodes": 127, "ensemble_size": 3, "n_islands": 1,
        "max_samples": (0.12, 1.0), "parsimony_target_nodes": 30,
        "evaluation_backend": "population_numba", "fitness_engine": "fused",
        "evaluation_cache": True, "fitness_cache_max_entries": 200000,
        "dag_execution": "off", "subtree_cache": False, "n_jobs": -1,
        "fitness_scaling": "elite_logistic", "fitness_scaling_top_k": 8,
        "fitness_scaling_interval": 3, "semantic_duplicate_elimination": False,
        "low_memory": True,
    },
    "publication": {
        "population_size": 96, "generations": 200, "max_depth": 14,
        "max_nodes": 255, "max_samples": 1.0, "ensemble_size": 5,
        "n_islands": 1, "parsimony_target_nodes": 45,
        "evaluation_backend": "population_numba", "fitness_engine": "fused",
        "evaluation_cache": True, "fitness_cache_max_entries": 250000,
        "dag_execution": "off", "subtree_cache": False, "n_jobs": -1,
        "fitness_scaling": "elite_logistic", "fitness_scaling_top_k": 16,
        "fitness_scaling_interval": 1, "semantic_duplicate_elimination": True,
        "constant_optimization": "champion", "constant_optimization_rounds": 3,
    },
    "mobile": {
        "population_size": 40, "generations": 100, "max_depth": 8,
        "max_nodes": 63, "ensemble_size": 1, "prediction_mode": "best",
        "function_set": ("add", "sub", "mul", "div", "abs"),
        "parsimony_target_nodes": 18, "evaluation_backend": "auto",
        "fitness_engine": "fused", "evaluation_cache": True,
        "n_jobs": -1,
    },
}


def resolve_auto_preset(X, y=None) -> str:
    n_samples, n_features = X.shape
    if n_samples >= 100_000 or n_features >= 1_000:
        return "large_dataset"
    if n_samples <= 2_000 and n_features <= 100:
        return "balanced"
    return "fast"


def _constructor_defaults(estimator) -> dict:
    defaults = {}
    try:
        signature = inspect.signature(estimator.__class__.__init__)
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.default is inspect._empty:
                continue
            defaults[name] = parameter.default
    except (TypeError, ValueError):
        pass
    return defaults


def _same_value(a, b) -> bool:
    try:
        result = a == b
        if hasattr(result, "all"):
            return bool(result.all())
        return bool(result)
    except Exception:
        return False


def apply_preset(estimator, X, y=None) -> dict:
    requested = str(getattr(estimator, "preset", "none") or "none").lower()
    if requested in {"none", "off", "custom"}:
        estimator.preset_ = "custom"
        estimator.preset_parameters_ = {}
        estimator.preset_skipped_parameters_ = {}
        return {}
    selected = resolve_auto_preset(X, y) if requested == "auto" else requested
    if selected not in PRESETS:
        raise ValueError(f"Unknown preset {requested!r}. Available: auto, {', '.join(sorted(PRESETS))}")

    requested_changes = dict(PRESETS[selected])
    if requested_changes.get("n_islands", 1) > 1:
        requested_changes["n_islands"] = min(requested_changes["n_islands"], max(1, os.cpu_count() or 1))

    policy = str(getattr(estimator, "preset_override_policy", "defaults_only") or "defaults_only").lower()
    if policy not in {"defaults_only", "always"}:
        raise ValueError("preset_override_policy must be defaults_only or always")
    defaults = _constructor_defaults(estimator)
    applied, skipped = {}, {}
    for name, value in requested_changes.items():
        if not hasattr(estimator, name):
            continue
        current = getattr(estimator, name)
        should_apply = policy == "always" or name not in defaults or _same_value(current, defaults[name])
        if should_apply:
            setattr(estimator, name, value)
            applied[name] = value
        else:
            skipped[name] = current

    estimator.preset_ = selected
    estimator.preset_parameters_ = applied
    estimator.preset_skipped_parameters_ = skipped
    return applied
