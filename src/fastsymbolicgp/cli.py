"""FastSymbolicGP command-line diagnostics and benchmark entry points."""
from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import sklearn

from ._version import __version__
from .backend import NUMBA_AVAILABLE
from .benchmark import run_publication_benchmark


def diagnose():
    try:
        import numba
        numba_version = numba.__version__
    except Exception:
        numba_version = None
    payload = {
        "fastsymbolicgp_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "numba": numba_version,
        "compiled_backend_available": NUMBA_AVAILABLE,
        "features": {
            "binary_classification": True,
            "multiclass_ovr": True,
            "shared_softmax_multiclass": True,
            "temperature_calibration": True,
            "regression": True,
            "symbolic_ensemble": True,
            "nsga2_pareto": True,
            "island_evolution": True,
            "dag_execution": True,
            "persistent_subtree_cache": True,
            "checkpoint_resume_and_branching": True,
            "symbolic_transformer_v2": True,
            "symbolic_distillation": True,
            "experimental_symbolic_network": True,
            "missing_value_operators": True,
            "robustness_fitness": True,
            "resource_budgets": True,
            "deployment_export": ["python", "c", "cpp", "java", "kotlin", "javascript"],
            "publication_benchmark_v3": True,
        },
        "status": "ok",
    }
    print(json.dumps(payload, indent=2))


def benchmark():
    parser = argparse.ArgumentParser(description="FastSymbolicGP V0.7.0 publication benchmark V3")
    parser.add_argument("--datasets", default="breast_cancer,iris,wine")
    parser.add_argument("--algorithms", default="fastsymbolicgp,logistic,random_forest,hist_gradient_boosting,svm")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--backend", choices=["auto", "numba", "numpy"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="benchmark_v070_results")
    parser.add_argument("--islands", type=int, default=1)
    parser.add_argument("--time-budget", type=float, default=None)
    args = parser.parse_args()
    run_publication_benchmark(
        output_dir=args.output_dir,
        datasets=tuple(x.strip() for x in args.datasets.split(",") if x.strip()),
        algorithms=tuple(x.strip() for x in args.algorithms.split(",") if x.strip()),
        runs=args.runs, folds=args.folds,
        population_size=args.population_size, generations=args.generations,
        backend=args.backend, seed=args.seed,
        fastsymbolic_params={
            "evolution_model": "islands" if args.islands > 1 else "panmictic",
            "n_islands": args.islands,
            "time_budget": args.time_budget,
        },
    )
    print(f"Saved publication outputs to: {args.output_dir}")


if __name__ == "__main__":
    diagnose()
