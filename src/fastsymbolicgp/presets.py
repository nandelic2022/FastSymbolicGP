"""Task-aware parameter presets for FastSymbolicGP V0.7.0."""
from __future__ import annotations

import os


PRESETS = {
    "fast": {
        "population_size": 32, "generations": 60, "max_depth": 10,
        "max_nodes": 127, "ensemble_size": 3, "n_islands": 1,
        "parsimony_target_nodes": 32,
    },
    "balanced": {
        "population_size": 64, "generations": 120, "max_depth": 14,
        "max_nodes": 255, "ensemble_size": 5, "n_islands": 2,
        "parsimony_target_nodes": 45,
    },
    "accurate": {
        "population_size": 96, "generations": 200, "max_depth": 16,
        "max_nodes": 383, "ensemble_size": 9, "n_islands": 4,
        "parsimony_target_nodes": 65,
    },
    "interpretable": {
        "population_size": 64, "generations": 140, "max_depth": 9,
        "max_nodes": 95, "ensemble_size": 1, "prediction_mode": "best",
        "parsimony_target_nodes": 24, "selection_tolerance": 0.03,
    },
    "large_dataset": {
        "population_size": 48, "generations": 100, "max_depth": 10,
        "max_nodes": 127, "max_samples": 0.35, "ensemble_size": 3,
        "parsimony_target_nodes": 32,
    },
    "mobile": {
        "population_size": 40, "generations": 100, "max_depth": 8,
        "max_nodes": 63, "ensemble_size": 1, "prediction_mode": "best",
        "function_set": ("add", "sub", "mul", "div", "abs"),
        "parsimony_target_nodes": 18,
    },
}


def resolve_auto_preset(X, y=None) -> str:
    n_samples, n_features = X.shape
    if n_samples >= 100_000 or n_features >= 1_000:
        return "large_dataset"
    if n_samples <= 2_000 and n_features <= 100:
        return "balanced"
    return "fast"


def apply_preset(estimator, X, y=None) -> dict:
    requested = str(getattr(estimator, "preset", "none") or "none").lower()
    if requested in {"none", "off", "custom"}:
        estimator.preset_ = "custom"
        return {}
    selected = resolve_auto_preset(X, y) if requested == "auto" else requested
    if selected not in PRESETS:
        raise ValueError(f"Unknown preset {requested!r}. Available: auto, {', '.join(sorted(PRESETS))}")
    changes = dict(PRESETS[selected])
    # Use available CPU cores conservatively for the island preset.
    if changes.get("n_islands", 1) > 1:
        changes["n_islands"] = min(changes["n_islands"], max(1, os.cpu_count() or 1))
    for name, value in changes.items():
        if hasattr(estimator, name):
            setattr(estimator, name, value)
    estimator.preset_ = selected
    estimator.preset_parameters_ = changes
    return changes
