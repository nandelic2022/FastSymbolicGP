"""Cross-run interpretability and stability helpers."""
from __future__ import annotations

from collections import Counter


def feature_stability(models) -> dict:
    """Return the fraction of fitted models using each feature."""
    models = list(models)
    if not models:
        return {}
    present = Counter()
    total_counts = Counter()
    for model in models:
        usage = getattr(model, "feature_usage_", {})
        for feature, stats in usage.items():
            present[feature] += 1
            total_counts[feature] += int(stats.get("count", 1)) if isinstance(stats, dict) else 1
    n = len(models)
    return {
        feature: {
            "model_fraction": present[feature] / n,
            "models": present[feature],
            "total_occurrences": total_counts[feature],
        }
        for feature in sorted(present, key=lambda key: (-present[key], key))
    }


def subexpression_stability(models, top_k=50) -> list[dict]:
    counts = Counter()
    for model in models:
        programs = []
        if getattr(model, "_is_multiclass_", False):
            programs = getattr(model, "shared_programs_", []) or [e.best_program_ for e in model.estimators_]
        else:
            programs = [model.best_program_]
        seen = set()
        for program in programs:
            for _, node in program.root.iter_paths():
                text = node.to_string(getattr(model, "feature_names_in_", None))
                seen.add(text)
        counts.update(seen)
    n = max(1, len(list(models)))
    return [
        {"subexpression": text, "models": count, "model_fraction": count / n}
        for text, count in counts.most_common(int(top_k))
    ]
