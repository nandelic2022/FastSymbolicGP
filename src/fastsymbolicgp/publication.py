"""Publication-grade statistical, robustness, stability and certification utilities.

The module deliberately keeps SciPy optional. Rank summaries and paired effect
sizes are always available; exact/asymptotic p-values and the Nemenyi critical
difference are added when SciPy is installed.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ._version import __version__


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def paired_rank_biserial(x, y, *, higher_is_better=True) -> float:
    """Paired rank-biserial correlation in [-1, 1]. Positive favors x."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    d = x - y if higher_is_better else y - x
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return 0.0
    ranks = _rankdata(np.abs(d))
    pos = float(np.sum(ranks[d > 0])); neg = float(np.sum(ranks[d < 0]))
    den = pos + neg
    return 0.0 if den <= 0 else float((pos - neg) / den)


def holm_adjust(p_values):
    """Holm family-wise-error correction preserving original order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.where(np.isfinite(p))[0]
    if not len(finite):
        return out.tolist()
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for j, idx in enumerate(order):
        running = max(running, (m - j) * float(p[idx]))
        out[idx] = min(1.0, running)
    return out.tolist()


def _scipy_stats():
    try:
        from scipy import stats
        return stats
    except Exception:
        return None


def compare_algorithms(rows, metric: str, *, higher_is_better=True, alpha=0.05,
                       dataset_key="dataset", algorithm_key="algorithm",
                       block_key="split") -> dict:
    """Paired multi-algorithm analysis on common dataset/split blocks.

    Only blocks for which every algorithm has a finite value are included,
    preventing unpaired failures from contaminating statistical tests.
    """
    rows = list(rows)
    metric_rows = []
    for r in rows:
        try:
            value = float(r[metric])
        except Exception:
            continue
        if np.isfinite(value) and algorithm_key in r:
            metric_rows.append((r, value))
    algorithms = sorted({str(r[algorithm_key]) for r, _ in metric_rows})
    by_block = {}
    for r, value in metric_rows:
        block = (str(r.get(dataset_key, "")), int(r.get(block_key, 0)))
        by_block.setdefault(block, {})[str(r[algorithm_key])] = value
    common = [(b, vals) for b, vals in sorted(by_block.items()) if all(a in vals for a in algorithms)]
    matrix = np.asarray([[vals[a] for a in algorithms] for _, vals in common], dtype=float) if common else np.empty((0, len(algorithms)))
    oriented = -matrix if higher_is_better else matrix
    ranks = np.asarray([_rankdata(row) for row in oriented]) if len(matrix) else np.empty_like(matrix)
    avg_ranks = ranks.mean(axis=0) if len(ranks) else np.full(len(algorithms), np.nan)

    stats = _scipy_stats()
    friedman_stat = friedman_p = None
    if stats is not None and matrix.shape[0] >= 2 and matrix.shape[1] >= 3:
        try:
            fr = stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
            friedman_stat, friedman_p = float(fr.statistic), float(fr.pvalue)
        except Exception:
            pass

    pairs = []
    raw_ps = []
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            x, y = matrix[:, i], matrix[:, j]
            p = None
            if len(x) >= 2 and np.allclose(x, y):
                p = 1.0
            elif stats is not None and len(x) >= 2:
                try:
                    res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
                    p = float(res.pvalue)
                except Exception:
                    p = None
            effect = paired_rank_biserial(x, y, higher_is_better=higher_is_better) if len(x) else 0.0
            mean_delta = float(np.mean(x - y)) if len(x) else float("nan")
            pairs.append({"algorithm_a": algorithms[i], "algorithm_b": algorithms[j], "n_blocks": int(len(x)), "p_value": p, "rank_biserial": effect, "mean_delta_a_minus_b": mean_delta})
            raw_ps.append(np.nan if p is None else p)
    adj = holm_adjust(raw_ps)
    for row, p_adj in zip(pairs, adj):
        row["p_holm"] = None if not np.isfinite(p_adj) else float(p_adj)
        row["significant_holm"] = bool(np.isfinite(p_adj) and p_adj < alpha)

    cd = None
    if stats is not None and len(algorithms) >= 2 and len(common) >= 2:
        try:
            # q_alpha from studentized range, converted for average-rank CD.
            q = float(stats.studentized_range.ppf(1.0 - alpha, len(algorithms), np.inf) / math.sqrt(2.0))
            cd = q * math.sqrt(len(algorithms) * (len(algorithms) + 1) / (6.0 * len(common)))
        except Exception:
            pass

    return {
        "metric": metric,
        "higher_is_better": bool(higher_is_better),
        "alpha": float(alpha),
        "algorithms": algorithms,
        "n_common_blocks": int(len(common)),
        "blocks": [list(b) for b, _ in common],
        "average_ranks": {a: float(r) for a, r in zip(algorithms, avg_ranks)},
        "friedman": {"statistic": friedman_stat, "p_value": friedman_p, "available": friedman_p is not None},
        "nemenyi_critical_difference": None if cd is None else float(cd),
        "pairwise_wilcoxon": pairs,
        "scipy_available": bool(stats is not None),
    }


def stability_report(models, X=None) -> dict:
    """Cross-run structural and optional prediction stability summary."""
    models = list(models)
    if not models:
        return {"n_models": 0, "expressions": {}, "feature_frequency": {}, "prediction": None}
    exprs = [m.get_expression() for m in models]
    expression_counts = Counter(exprs)
    feature_counts = Counter()
    for m in models:
        p = getattr(m, "best_program_", None)
        if p is not None:
            for idx in set(p.root.feature_indices()):
                names = getattr(m, "_symbolic_feature_names_", None)
                name = str(names[idx]) if names is not None else f"X{idx}"
                feature_counts[name] += 1
    pred = None
    if X is not None:
        values = np.column_stack([np.asarray(m.predict(X), dtype=float) for m in models])
        pred = {
            "mean_pairwise_correlation": 1.0 if values.shape[1] == 1 else float(np.nanmean(np.corrcoef(values, rowvar=False)[np.triu_indices(values.shape[1], 1)])),
            "mean_prediction_std": float(np.mean(np.std(values, axis=1))),
            "max_prediction_std": float(np.max(np.std(values, axis=1))),
        }
    return {
        "n_models": len(models),
        "unique_expressions": len(expression_counts),
        "expression_frequency": dict(expression_counts),
        "feature_frequency": {k: {"models": int(v), "fraction": float(v / len(models))} for k, v in sorted(feature_counts.items())},
        "prediction": pred,
    }


def robustness_report(model, X, *, noise_levels=(0.001, 0.01, 0.05), repeats=5, random_state=0) -> dict:
    """Prediction sensitivity to relative Gaussian input perturbations."""
    X = np.asarray(X, dtype=float)
    base = np.asarray(model.predict(X), dtype=float)
    scale = np.nanstd(X, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    rng = np.random.default_rng(random_state)
    rows = []
    for level in noise_levels:
        deltas = []
        for _ in range(int(repeats)):
            noisy = X + rng.normal(0.0, float(level), size=X.shape) * scale
            pred = np.asarray(model.predict(noisy), dtype=float)
            deltas.append(pred - base)
        d = np.concatenate([np.ravel(v) for v in deltas])
        rows.append({"noise_level": float(level), "repeats": int(repeats), "mae_prediction_change": float(np.mean(np.abs(d))), "rmse_prediction_change": float(np.sqrt(np.mean(d*d))), "max_abs_prediction_change": float(np.max(np.abs(d)))})
    return {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1]), "rows": rows}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scientific_certificate(model, X=None) -> dict:
    """Machine-readable certification snapshot for a fitted model."""
    manifest = model.model_manifest()
    constraint = model.get_constraint_report(X) if X is not None else dict(getattr(model, "constraint_report_", {}))
    try:
        unit = model.get_unit_report()
    except Exception as exc:
        unit = {"enabled": False, "passed": False, "error": str(exc)}
    complexity = model.get_expression_stats()
    checks = {
        "fitted": hasattr(model, "best_program_"),
        "finite_expression_complexity": all(np.isfinite(float(v)) for v in complexity.values() if isinstance(v, (int, float))),
        "constraints_passed": bool(constraint.get("passed", True)),
        "units_passed": bool(unit.get("passed", True)),
        "run_fingerprint_present": bool(getattr(model, "run_fingerprint_", None)),
    }
    return {
        "format": "FastSymbolicGP-scientific-certificate-v1",
        "version": __version__,
        "estimator": model.__class__.__name__,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "manifest": manifest,
        "constraint_report": constraint,
        "unit_report": unit,
        "complexity": complexity,
        "profile": dict(getattr(model, "profile_", {})),
    }


def write_reproducibility_bundle_v2(model, path, X=None, y=None) -> str:
    """Create a self-describing reproducibility bundle with content hashes."""
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(model.get_params(deep=False), indent=2, default=str), encoding="utf-8")
    (path / "resolved_config.json").write_text(json.dumps(model.get_resolved_config(), indent=2, default=str), encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps(model.model_manifest(), indent=2, default=str), encoding="utf-8")
    (path / "scientific_certificate.json").write_text(json.dumps(scientific_certificate(model, X), indent=2, default=str), encoding="utf-8")
    (path / "expression.txt").write_text(model.get_expression() + "\n", encoding="utf-8")
    try:
        (path / "expression.tex").write_text(model.to_latex() + "\n", encoding="utf-8")
    except Exception:
        pass
    try:
        model.save_model_json(path / "model.safe.json")
    except Exception:
        pass
    try:
        model.history_dataframe().to_csv(path / "history.csv", index=False)
    except Exception:
        (path / "history.json").write_text(json.dumps(getattr(model, "history_", []), indent=2, default=str), encoding="utf-8")
    if X is not None:
        np.save(path / "X_reference.npy", np.asarray(X))
        if y is not None:
            np.save(path / "y_reference.npy", np.asarray(y))
        try:
            (path / "robustness.json").write_text(json.dumps(robustness_report(model, X, repeats=3), indent=2), encoding="utf-8")
        except Exception:
            pass
    packages = {}
    for name in ("numpy", "sklearn", "scipy", "pandas", "numba", "sympy"):
        try:
            mod = __import__(name); packages[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            packages[name] = None
    env = {"fastsymbolicgp": __version__, "python": platform.python_version(), "python_implementation": platform.python_implementation(), "platform": platform.platform(), "executable": sys.executable, "packages": packages}
    (path / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    hashes = {p.name: _sha256(p) for p in sorted(path.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"}
    (path / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)
