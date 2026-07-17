"""Reproducible publication benchmark utilities for FastSymbolicGP V0.7.0."""
from __future__ import annotations

import csv
import json
import platform
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ._version import __version__
from .classifier import FastSymbolicClassifier


DATASETS = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "wine": load_wine,
}


def _write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _factory(name, seed, population_size, generations, backend, fastsymbolic_params=None):
    name = name.lower()
    if name == "fastsymbolicgp":
        params = dict(
            population_size=population_size, generations=generations,
            tournament_size=max(3, min(10, population_size // 3)),
            init_depth=(2, 8), max_depth=14, max_nodes=255,
            class_weight="balanced", validation_fraction=0.20,
            selection_metric="combined", optimization="nsga2",
            final_selection="smallest_within_tolerance", selection_tolerance=0.02,
            prediction_mode="symbolic_ensemble", ensemble_size=5,
            probability_calibration="auto", multiclass_calibration="temperature",
            threshold_strategy="balanced_accuracy", parsimony="adaptive",
            parsimony_target_nodes=40, subtree_cache=True,
            evaluation_backend=backend, random_state=seed, verbose=0,
        )
        params.update(fastsymbolic_params or {})
        return FastSymbolicClassifier(**params)
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed)
    if name in {"hist_gradient_boosting", "hgb"}:
        return HistGradientBoostingClassifier(max_iter=250, random_state=seed)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=seed))
    if name == "gplearn":
        try:
            from gplearn.genetic import SymbolicClassifier
        except Exception as exc:
            raise RuntimeError("gplearn is not installed") from exc
        return SymbolicClassifier(population_size=population_size, generations=generations, random_state=seed, verbose=0)
    raise ValueError(f"Unknown algorithm: {name}")


def _model_stats(model):
    if not isinstance(model, FastSymbolicClassifier):
        return {}
    stats = model.get_expression_stats()
    if getattr(model, "_is_multiclass_", False):
        return {
            "strategy": model.multiclass_strategy_,
            "evaluation_backend": getattr(model.estimators_[0], "evaluation_backend_", model.evaluation_backend),
            "execution_engine": getattr(model.estimators_[0], "execution_engine_", "unknown"),
            "generations_total": stats["generations_total"],
            "generations_mean_per_class": stats["generations_mean_per_class"],
            "generations_max_per_class": stats["generations_max_per_class"],
            "nodes_total": stats["nodes_total"], "nodes_mean_per_model": stats["nodes_mean_per_model"],
            "nodes_max_per_model": stats["nodes_max_per_model"],
            "dag_nodes_total": stats["dag_nodes_total"], "dag_nodes_mean_per_model": stats["dag_nodes_mean_per_model"],
            "depth_mean_per_model": stats["depth_mean_per_model"], "depth_max_per_model": stats["depth_max_per_model"],
            "calibration_method": model.multiclass_calibration_report_["selected_method"],
            "calibration_ece": model.multiclass_calibration_report_["expected_calibration_error"],
        }
    return {
        "strategy": "binary", "evaluation_backend": model.evaluation_backend_,
        "execution_engine": getattr(model, "execution_engine_", "unknown"),
        "generations_total": model.n_generations_, "generations_mean_per_class": model.n_generations_,
        "generations_max_per_class": model.n_generations_,
        "nodes_total": stats["nodes"], "nodes_mean_per_model": stats["nodes"], "nodes_max_per_model": stats["nodes"],
        "dag_nodes_total": stats["dag_nodes"], "dag_nodes_mean_per_model": stats["dag_nodes"],
        "depth_mean_per_model": stats["depth"], "depth_max_per_model": stats["depth"],
        "cache_hit_rate": getattr(model, "cache_statistics_", {}).get("hit_rate", 0.0),
        "calibration_method": getattr(model, "calibration_report_", {}).get("selected_method", "none"),
    }


def _summary(rows, metrics):
    output = []
    groups = sorted({(row["dataset"], row["algorithm"]) for row in rows})
    for dataset, algorithm in groups:
        subset = [row for row in rows if row["dataset"] == dataset and row["algorithm"] == algorithm]
        row = {"dataset": dataset, "algorithm": algorithm, "n": len(subset)}
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in subset if item.get(metric, "") not in {"", None}], dtype=float)
            if values.size:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1) if len(values) > 1 else 0.0)
                row[f"{metric}_min"] = float(values.min())
                row[f"{metric}_max"] = float(values.max())
        output.append(row)
    return output


def _rank_table(summary_rows, metric="balanced_accuracy"):
    datasets = sorted({row["dataset"] for row in summary_rows})
    algorithms = sorted({row["algorithm"] for row in summary_rows})
    rank_rows = []
    per_algorithm = {algorithm: [] for algorithm in algorithms}
    for dataset in datasets:
        values = []
        for algorithm in algorithms:
            match = next((r for r in summary_rows if r["dataset"] == dataset and r["algorithm"] == algorithm), None)
            if match and f"{metric}_mean" in match:
                values.append((algorithm, float(match[f"{metric}_mean"])))
        ordered = sorted(values, key=lambda item: item[1], reverse=True)
        for rank, (algorithm, value) in enumerate(ordered, start=1):
            rank_rows.append({"dataset": dataset, "algorithm": algorithm, metric: value, "rank": rank})
            per_algorithm[algorithm].append(rank)
    averages = [{"algorithm": algorithm, "average_rank": float(np.mean(ranks)), "datasets": len(ranks)} for algorithm, ranks in per_algorithm.items() if ranks]
    averages.sort(key=lambda row: row["average_rank"])
    return rank_rows, averages


def _write_latex_tables(output_dir, summary_rows, rank_averages):
    latex = output_dir / "latex"; latex.mkdir(exist_ok=True)
    lines = ["\\begin{tabular}{llrrrr}", "\\hline", "Dataset & Algorithm & BACC & AUC & Log-loss & Time (s) \\\\", "\\hline"]
    for row in summary_rows:
        lines.append(
            f"{row['dataset'].replace('_','\\_')} & {row['algorithm'].replace('_','\\_')} & "
            f"{row.get('balanced_accuracy_mean', float('nan')):.4f} & {row.get('roc_auc_mean', float('nan')):.4f} & "
            f"{row.get('log_loss_mean', float('nan')):.4f} & {row.get('fit_seconds_mean', float('nan')):.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    (latex / "performance_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rank_lines = ["\\begin{tabular}{lr}", "\\hline", "Algorithm & Average rank \\\\", "\\hline"]
    for row in rank_averages:
        rank_lines.append(f"{row['algorithm'].replace('_','\\_')} & {row['average_rank']:.3f} \\\\ ")
    rank_lines.extend(["\\hline", "\\end{tabular}"])
    (latex / "rank_table.tex").write_text("\n".join(rank_lines) + "\n", encoding="utf-8")


def run_publication_benchmark(
    output_dir,
    datasets=("breast_cancer", "iris", "wine"),
    algorithms=("fastsymbolicgp", "logistic", "random_forest", "hist_gradient_boosting", "svm"),
    runs=3,
    folds=5,
    population_size=30,
    generations=120,
    backend="auto",
    seed=42,
    fastsymbolic_params=None,
):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "fold_splits"; split_dir.mkdir(exist_ok=True)
    expression_dir = output_dir / "expressions"; expression_dir.mkdir(exist_ok=True)
    report_dir = output_dir / "reports"; report_dir.mkdir(exist_ok=True)
    raw_rows, failures = [], []

    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        X, y = DATASETS[dataset_name](return_X_y=True)
        splitter = RepeatedStratifiedKFold(n_splits=int(folds), n_repeats=int(runs), random_state=int(seed))
        splits = list(splitter.split(X, y))
        for split_index, (train_idx, test_idx) in enumerate(splits, start=1):
            np.savez_compressed(split_dir / f"{dataset_name}_split_{split_index:03d}.npz", train_idx=train_idx, test_idx=test_idx)
            for algorithm in algorithms:
                model_seed = int(seed) + split_index * 1009
                try:
                    model = _factory(algorithm, model_seed, int(population_size), int(generations), backend, fastsymbolic_params)
                    start = perf_counter(); model.fit(X[train_idx], y[train_idx]); elapsed = perf_counter() - start
                    pred = model.predict(X[test_idx])
                    proba = model.predict_proba(X[test_idx]) if hasattr(model, "predict_proba") else None
                    binary = len(np.unique(y)) == 2
                    row = {
                        "dataset": dataset_name, "algorithm": algorithm,
                        "split": split_index, "repeat": (split_index - 1) // int(folds) + 1,
                        "fold": (split_index - 1) % int(folds) + 1,
                        "accuracy": float(accuracy_score(y[test_idx], pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                        "f1": float(f1_score(y[test_idx], pred, average="binary" if binary else "macro", zero_division=0)),
                        "mcc": float(matthews_corrcoef(y[test_idx], pred)),
                        "fit_seconds": float(elapsed), "status": "ok",
                    }
                    if proba is not None:
                        row["roc_auc"] = float(roc_auc_score(y[test_idx], proba[:, 1] if binary else proba, multi_class="ovr", average="macro"))
                        row["log_loss"] = float(log_loss(y[test_idx], proba, labels=np.unique(y)))
                    row.update(_model_stats(model))
                    raw_rows.append(row)
                    if isinstance(model, FastSymbolicClassifier):
                        model.save_report_json(report_dir / f"{dataset_name}_{algorithm}_{split_index:03d}.json")
                        (expression_dir / f"{dataset_name}_{algorithm}_{split_index:03d}.txt").write_text(str(model.get_expression()) + "\n", encoding="utf-8")
                    print(f"[{dataset_name}][{algorithm}] {split_index:02d}/{len(splits)} BACC={row['balanced_accuracy']:.4f} time={elapsed:.2f}s")
                except Exception as exc:
                    failures.append({
                        "dataset": dataset_name, "algorithm": algorithm, "split": split_index,
                        "error": repr(exc), "traceback": traceback.format_exc(),
                    })
                    print(f"[{dataset_name}][{algorithm}] FAILED: {exc}")

    metrics = ["accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc", "log_loss", "fit_seconds", "nodes_total", "dag_nodes_total"]
    summary_rows = _summary(raw_rows, metrics)
    rank_rows, rank_averages = _rank_table(summary_rows)
    _write_csv(output_dir / "raw_results.csv", raw_rows)
    _write_csv(output_dir / "summary_results.csv", summary_rows)
    _write_csv(output_dir / "per_dataset_ranks.csv", rank_rows)
    _write_csv(output_dir / "average_ranks.csv", rank_averages)
    _write_csv(output_dir / "failures.csv", failures)
    (output_dir / "summary_results.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    (output_dir / "statistics.json").write_text(json.dumps({"average_ranks": rank_averages}, indent=2), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps({
        "fastsymbolicgp": __version__, "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "datasets": list(datasets), "algorithms": list(algorithms), "runs": runs, "folds": folds,
        "population_size": population_size, "generations": generations, "backend": backend, "seed": seed,
    }, indent=2), encoding="utf-8")
    _write_latex_tables(output_dir, summary_rows, rank_averages)
    return {"raw_results": raw_rows, "summary": summary_rows, "failures": failures, "average_ranks": rank_averages}
