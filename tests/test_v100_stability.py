import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from fastsymbolicgp import FastSymbolicClassifier, FastSymbolicRegressor, warmup


def test_named_preset_preserves_explicit_user_parameters():
    X, y = load_breast_cancer(return_X_y=True)
    model = FastSymbolicClassifier(
        preset="balanced", population_size=37, generations=3,
        n_islands=3, display="off", verbose=0, random_state=1,
    )
    model.fit(X[:180], y[:180])
    assert model.population_size == 37
    assert model.generations == 3
    assert model.n_islands == 3
    assert model.resolved_config_["population_size"] == 37
    assert "population_size" in model.preset_skipped_parameters_


def test_auto_planner_uses_single_island_for_small_dataset():
    X, y = load_breast_cancer(return_X_y=True)
    model = FastSymbolicClassifier(
        preset="auto", generations=2, population_size=20,
        display="off", verbose=0, random_state=2,
    )
    model.fit(X[:220], y[:220])
    assert model.execution_plan_["workload"] == "small"
    assert model.n_islands == 1
    assert model.evolution_model == "panmictic"


def test_exact_validation_champion_is_archived_every_generation():
    X, y = load_breast_cancer(return_X_y=True)
    model = FastSymbolicClassifier(
        population_size=20, generations=4, tournament_size=4,
        fitness_scaling="elite_logistic", fitness_scaling_interval=99,
        exact_champion=True, exact_champion_candidates=3,
        validation_fraction=0.25, n_islands=1,
        display="off", verbose=0, random_state=3,
    )
    model.fit(X[:260], y[:260])
    assert model.exact_validation_champion_ is not None
    assert model.best_validation_program_ is not None
    vals = [r["best_exact_validation_fitness"] for r in model.history_]
    assert all(v is not None and np.isfinite(v) for v in vals)
    # Global exact champion can only improve or stay unchanged.
    assert all(vals[i] <= vals[i-1] + 1e-12 for i in range(1, len(vals)))


def test_profile_summary_and_manifest_are_available():
    X, y = load_diabetes(return_X_y=True)
    model = FastSymbolicRegressor(
        population_size=16, generations=2, tournament_size=3,
        display="off", verbose=0, random_state=4,
    ).fit(X[:180], y[:180])
    assert model.profile_["actual_program_evaluations"] > 0
    assert model.profile_["programs_per_second"] > 0
    summary = model.summary(print_output=False)
    assert summary["version"] == "1.0.0"
    manifest = model.model_manifest()
    assert manifest["format"] == "FastSymbolicGP-model-manifest-v1"
    assert manifest["version"] == "1.0.0"


def test_flat_genome_and_cached_indexing():
    X, y = load_diabetes(return_X_y=True)
    model = FastSymbolicRegressor(
        population_size=12, generations=1, tournament_size=2,
        display="off", verbose=0, random_state=5,
    ).fit(X[:100], y[:100])
    genome = model.best_program_.flat_genome()
    assert genome["opcodes"].shape == genome["arguments"].shape
    assert genome["size"] == model.best_program_.size
    assert len(model.best_program_.paths()) == model.best_program_.size


def test_champion_constant_optimization_runs_without_regression():
    X, y = load_diabetes(return_X_y=True)
    model = FastSymbolicRegressor(
        population_size=18, generations=2, tournament_size=3,
        constant_probability=0.8, constant_optimization="champion",
        constant_optimization_rounds=1,
        display="off", verbose=0, random_state=7,
    ).fit(X[:180], y[:180])
    report = model.constant_optimization_report_
    assert report["enabled"] is True
    assert report["evaluations"] >= 0
