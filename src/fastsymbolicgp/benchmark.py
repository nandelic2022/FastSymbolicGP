"""Reproducible publication benchmark utilities for FastSymbolicGP V1.2.4."""
from __future__ import annotations

import csv
import json
import platform
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_diabetes, make_friedman1
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, log_loss, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ._version import __version__
from .classifier import FastSymbolicClassifier
from .regressor import FastSymbolicRegressor
from .publication import compare_algorithms

CLASSIFICATION_DATASETS = {"breast_cancer": load_breast_cancer, "iris": load_iris, "wine": load_wine}
REGRESSION_DATASETS = {"diabetes": load_diabetes}


def _nguyen(name, n=400, seed=42):
    rng = np.random.default_rng(seed)
    name = str(name).lower()
    if name in {"nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6"}:
        x = rng.uniform(-1.0, 1.0, size=n); X = x[:, None]
        if name == "nguyen1": y = x**3 + x**2 + x
        elif name == "nguyen2": y = x**4 + x**3 + x**2 + x
        elif name == "nguyen3": y = x**5 + x**4 + x**3 + x**2 + x
        elif name == "nguyen4": y = x**6 + x**5 + x**4 + x**3 + x**2 + x
        elif name == "nguyen5": y = np.sin(x**2) * np.cos(x) - 1.0
        else: y = np.sin(x) + np.sin(x + x**2)
        return X, y
    if name == "nguyen7":
        x = rng.uniform(0.0, 2.0, size=n); return x[:, None], np.log(x + 1.0) + np.log(x**2 + 1.0)
    if name == "nguyen8":
        x = rng.uniform(0.0, 4.0, size=n); return x[:, None], np.sqrt(x)
    if name == "nguyen9":
        X = rng.uniform(-1.0, 1.0, size=(n, 2)); x, z = X[:, 0], X[:, 1]; return X, np.sin(x) + np.sin(z**2)
    if name == "nguyen10":
        X = rng.uniform(-1.0, 1.0, size=(n, 2)); x, z = X[:, 0], X[:, 1]; return X, 2.0 * np.sin(x) * np.cos(z)
    raise KeyError(name)


def _friedman1(n=500, seed=42):
    return make_friedman1(n_samples=int(n), n_features=10, noise=0.0, random_state=int(seed))


SYMBOLIC_REGRESSION_DATASETS = {**{f"nguyen{i}": (lambda i=i: _nguyen(f"nguyen{i}")) for i in range(1, 11)}, "friedman1": _friedman1}
REGRESSION_DATASETS.update(SYMBOLIC_REGRESSION_DATASETS)


def _load_dataset(registry, name):
    loader = registry[name]
    try:
        return loader(return_X_y=True)
    except TypeError:
        return loader()


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8"); return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def _read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            out = dict(r)
            for key, value in list(out.items()):
                if key in {"split", "repeat", "fold"}:
                    try: out[key] = int(value)
                    except Exception: pass
                elif key not in {"task", "dataset", "algorithm", "status", "expression"}:
                    try: out[key] = float(value)
                    except Exception: pass
            rows.append(out)
        return rows


def _factory(task, name, seed, population_size, generations, params=None):
    name = str(name).lower(); extra = dict(params or {})
    if name in {"fastsymbolicgp", "fsgp"}:
        base = dict(population_size=population_size, generations=generations, random_state=seed, verbose=0); base.update(extra)
        return FastSymbolicClassifier(**base) if task == "classification" else FastSymbolicRegressor(uncertainty="off", **base)
    if task == "classification":
        if name in {"logistic", "logistic_regression"}: return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        if name in {"random_forest", "rf"}: return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed)
        if name in {"hist_gradient_boosting", "hgb"}: return HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    else:
        if name in {"linear", "linear_regression"}: return LinearRegression()
        if name in {"random_forest", "rf"}: return RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
        if name in {"hist_gradient_boosting", "hgb"}: return HistGradientBoostingRegressor(max_iter=200, random_state=seed)
    raise ValueError(f"Unknown {task} algorithm: {name}")


def _metric_directions(task):
    if task == "classification":
        return {"accuracy": True, "balanced_accuracy": True, "f1": True, "mcc": True, "roc_auc": True, "log_loss": False, "fit_seconds": False}
    return {"r2": True, "mae": False, "rmse": False, "nrmse": False, "fit_seconds": False}


def analyze_benchmark_results(raw_results, task, output_dir=None, alpha=0.05):
    """Create paired statistical analyses for every available benchmark metric."""
    raw_results = [r for r in raw_results if str(r.get("status", "ok")) == "ok"]
    analyses = {}
    for metric, higher in _metric_directions(task).items():
        if sum(metric in r and isinstance(r.get(metric), (int, float, np.integer, np.floating)) for r in raw_results) == 0:
            continue
        analyses[metric] = compare_algorithms(raw_results, metric, higher_is_better=higher, alpha=alpha)
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "statistical_comparison.json").write_text(json.dumps(analyses, indent=2, default=str), encoding="utf-8")
        rank_rows = []
        pair_rows = []
        for metric, report in analyses.items():
            for alg, rank in report["average_ranks"].items():
                rank_rows.append({"metric": metric, "algorithm": alg, "average_rank": rank, "n_common_blocks": report["n_common_blocks"], "critical_difference": report["nemenyi_critical_difference"]})
            for row in report["pairwise_wilcoxon"]:
                pair_rows.append({"metric": metric, **row})
        _write_csv(out / "average_ranks.csv", rank_rows)
        _write_csv(out / "pairwise_wilcoxon_holm.csv", pair_rows)
    return analyses


def run_publication_benchmark(output_dir, task="classification", datasets=None, algorithms=None, runs=1, folds=5,
                              population_size=32, generations=60, seed=42, fastsymbolic_params=None,
                              resume=True, alpha=0.05):
    task = str(task).lower(); registry = CLASSIFICATION_DATASETS if task == "classification" else REGRESSION_DATASETS
    datasets = tuple(datasets or registry.keys())
    algorithms = tuple(algorithms or (("fastsymbolicgp", "logistic", "random_forest", "hist_gradient_boosting") if task == "classification" else ("fastsymbolicgp", "linear", "random_forest", "hist_gradient_boosting")))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_results.csv"; fail_path = out / "failures.csv"
    raw = _read_csv(raw_path) if resume else []
    failures = _read_csv(fail_path) if resume else []
    done = {(str(r.get("dataset")), str(r.get("algorithm")), int(r.get("split", -1))) for r in raw if str(r.get("status", "ok")) == "ok"}

    for ds in datasets:
        if ds not in registry: raise ValueError(f"Unknown dataset {ds!r} for task {task}")
        X, y = _load_dataset(registry, ds)
        splitter = RepeatedStratifiedKFold(n_splits=int(folds), n_repeats=int(runs), random_state=seed) if task == "classification" else RepeatedKFold(n_splits=int(folds), n_repeats=int(runs), random_state=seed)
        for split, (tr, te) in enumerate(splitter.split(X, y), start=1):
            repeat = (split - 1) // int(folds) + 1; fold = (split - 1) % int(folds) + 1
            for alg in algorithms:
                key = (str(ds), str(alg), int(split))
                if key in done:
                    continue
                try:
                    m = _factory(task, alg, seed + split * 1009, int(population_size), int(generations), fastsymbolic_params)
                    t = perf_counter(); m.fit(X[tr], y[tr]); elapsed = perf_counter() - t
                    t = perf_counter(); p = m.predict(X[te]); predict_seconds = perf_counter() - t
                    row = {"task": task, "dataset": ds, "algorithm": alg, "split": split, "repeat": repeat, "fold": fold, "fit_seconds": elapsed, "predict_seconds": predict_seconds, "status": "ok", "train_n": len(tr), "test_n": len(te)}
                    if task == "classification":
                        row.update(accuracy=float(accuracy_score(y[te], p)), balanced_accuracy=float(balanced_accuracy_score(y[te], p)), f1=float(f1_score(y[te], p, average="binary" if len(np.unique(y)) == 2 else "macro", zero_division=0)), mcc=float(matthews_corrcoef(y[te], p)))
                        if hasattr(m, "predict_proba"):
                            P = m.predict_proba(X[te]); row["log_loss"] = float(log_loss(y[te], P, labels=np.unique(y))); row["roc_auc"] = float(roc_auc_score(y[te], P[:, 1] if P.shape[1] == 2 else P, multi_class="ovr", average="macro"))
                    else:
                        rmse = float(np.sqrt(mean_squared_error(y[te], p))); denom = float(np.std(y[te])); row.update(r2=float(r2_score(y[te], p)), mae=float(mean_absolute_error(y[te], p)), rmse=rmse, nrmse=float(rmse / denom) if denom > 1e-15 else float("nan"))
                    if hasattr(m, "get_expression_stats"):
                        row.update({f"model_{k}": v for k, v in m.get_expression_stats().items() if isinstance(v, (int, float))})
                        try: row["expression"] = m.get_expression()
                        except Exception: pass
                    raw.append(row); done.add(key); _write_csv(raw_path, raw)
                except Exception as exc:
                    failures.append({"task": task, "dataset": ds, "algorithm": alg, "split": split, "repeat": repeat, "fold": fold, "error": repr(exc), "traceback": traceback.format_exc()}); _write_csv(fail_path, failures)

    summary = []
    for ds in datasets:
        for alg in algorithms:
            ss = [r for r in raw if r.get("dataset") == ds and r.get("algorithm") == alg and str(r.get("status", "ok")) == "ok"]
            if not ss: continue
            row = {"task": task, "dataset": ds, "algorithm": alg, "n": len(ss)}
            for key in sorted({k for r in ss for k in r if isinstance(r.get(k), (int, float, np.integer, np.floating)) and k not in {"split", "repeat", "fold", "train_n", "test_n"}}):
                vals = np.asarray([r[key] for r in ss if key in r and np.isfinite(float(r[key]))], float)
                if len(vals): row[key + "_mean"] = float(vals.mean()); row[key + "_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0)
            summary.append(row)
    _write_csv(raw_path, raw); _write_csv(out / "summary_results.csv", summary); _write_csv(fail_path, failures)
    analyses = analyze_benchmark_results(raw, task, out, alpha=alpha)
    environment = {"fastsymbolicgp": __version__, "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "task": task, "datasets": list(datasets), "algorithms": list(algorithms), "runs": int(runs), "folds": int(folds), "seed": int(seed), "resume": bool(resume), "alpha": float(alpha)}
    (out / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return {"raw_results": raw, "summary": summary, "failures": failures, "statistical_comparison": analyses}


def benchmark_regressor(model, X_train, y_train, X_test, y_test):
    t = perf_counter(); model.fit(X_train, y_train); fit = perf_counter() - t; t = perf_counter(); p = model.predict(X_test); pred = perf_counter() - t
    return {"r2": float(r2_score(y_test, p)), "mae": float(mean_absolute_error(y_test, p)), "rmse": float(np.sqrt(mean_squared_error(y_test, p))), "fit_seconds": fit, "predict_seconds": pred, "nodes": model.get_expression_stats()["nodes"], "expression": model.get_expression()}
