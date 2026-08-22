"""Compare V0.9.0 fused fitness against the V0.8-style materialized path."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

from sklearn.datasets import make_classification, make_regression

from fastsymbolicgp import FastSymbolicClassifier, FastSymbolicRegressor, __version__


def timed_fit(factory, X, y, parameters, repeats):
    # Compile both postfix and fused kernels before timed measurements.
    warm = dict(parameters)
    warm.update(population_size=8, generations=1)
    factory(**warm).fit(X[:400], y[:400])
    times = []
    model = None
    for _ in range(repeats):
        start = perf_counter()
        model = factory(**parameters).fit(X, y)
        times.append(perf_counter() - start)
    return {
        "times_seconds": times,
        "median_seconds": statistics.median(times),
        "mean_seconds": statistics.mean(times),
        "minimum_seconds": min(times),
        "score": float(model.score(X, y)),
        "nodes": int(model.best_program_.size),
        "program_evaluations": int(model.evaluation_count_),
        "fitness_cache_hits": int(sum(row.get("fitness_cache_hits", 0) for row in model.history_)),
        "execution_engine": model.execution_engine_,
        "fitness_engine": model.history_[-1].get("fitness_engine", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12_000)
    parser.add_argument("--features", type=int, default=30)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", default="benchmark_v090_speed.json")
    args = parser.parse_args()

    Xc, yc = make_classification(
        n_samples=args.samples, n_features=args.features,
        n_informative=max(2, int(args.features * 0.6)),
        n_redundant=max(0, int(args.features * 0.2)),
        weights=[0.62, 0.38], random_state=123,
    )
    Xr, yr = make_regression(
        n_samples=args.samples, n_features=args.features,
        n_informative=max(2, int(args.features * 0.6)), noise=5.0,
        random_state=123,
    )

    common = dict(
        population_size=args.population, generations=args.generations,
        tournament_size=8, init_depth=(1, 4), max_depth=7, max_nodes=95,
        patience=0, semantic_duplicate_elimination=False, dag_execution="off",
        evaluation_backend="population_numba", n_jobs=args.jobs,
        batch_size="full", random_state=7, verbose=0,
    )
    classifier = dict(
        probability_calibration="none", threshold_strategy="fixed",
        fitness_scaling="elite_logistic", fitness_scaling_top_k=8,
        fitness_scaling_interval=2,
    )
    regressor = dict(fitness_metric="rmse", fitness_scaling="linear")

    results = {"version": __version__, "configuration": vars(args), "modes": {}}
    for mode, cache in (("materialized", False), ("fused", True)):
        results["modes"][mode] = {
            "classification": timed_fit(
                FastSymbolicClassifier, Xc, yc,
                {**common, **classifier, "fitness_engine": mode, "evaluation_cache": cache},
                args.repeats,
            ),
            "regression": timed_fit(
                FastSymbolicRegressor, Xr, yr,
                {**common, **regressor, "fitness_engine": mode, "evaluation_cache": cache},
                args.repeats,
            ),
        }

    for task in ("classification", "regression"):
        old = results["modes"]["materialized"][task]["median_seconds"]
        new = results["modes"]["fused"][task]["median_seconds"]
        results.setdefault("speedup", {})[task] = old / new

    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
