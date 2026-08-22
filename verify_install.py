import json
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris

from fastsymbolicgp import FastSymbolicClassifier, FastSymbolicTransformer, __version__
from fastsymbolicgp.backend import NUMBA_AVAILABLE

assert __version__ == "0.9.0", __version__

X, y = load_breast_cancer(return_X_y=True)
model = FastSymbolicClassifier(
    population_size=12, generations=2, tournament_size=3,
    init_depth=(1, 3), max_depth=5, max_nodes=47,
    probability_calibration="none", threshold_strategy="fixed",
    semantic_duplicate_elimination=False,
    evaluation_backend="auto", n_jobs=2,
    random_state=42, verbose=0,
).fit(X[:180], y[:180])
assert model.predict_proba(X[180:185]).shape == (5, 2)
if NUMBA_AVAILABLE:
    assert model.execution_engine_ == "population_numba", model.execution_engine_
    assert model.history_[-1].get("fitness_engine") == "fused", model.history_[-1]
assert "actual_program_evaluations" in model.history_[-1]
assert "fitness_cache_hits" in model.history_[-1]

Xi, yi = load_iris(return_X_y=True)
multi = FastSymbolicClassifier(
    population_size=9, generations=2, tournament_size=3,
    init_depth=(1, 3), max_depth=5, max_nodes=47,
    multiclass_strategy="shared_softmax", shared_n_components=4,
    multiclass_calibration="temperature", n_jobs=2,
    random_state=42, verbose=0,
).fit(Xi, yi)
assert np.allclose(multi.predict_proba(Xi[:5]).sum(axis=1), 1.0)

transformer = FastSymbolicTransformer(
    n_components=2, random_state=42, verbose=0,
    model_params={
        "population_size": 8, "generations": 1, "verbose": 0,
        "n_jobs": 2, "semantic_duplicate_elimination": False,
    },
).fit(Xi, yi)
assert transformer.transform(Xi[:3]).shape == (3, 2)

print(json.dumps({
    "version": __version__,
    "numba_available": NUMBA_AVAILABLE,
    "binary_engine": model.execution_engine_,
    "binary_fitness_engine": model.history_[-1].get("fitness_engine"),
    "binary_actual_program_evaluations": model.history_[-1].get("actual_program_evaluations"),
    "binary_fitness_cache_hits": model.history_[-1].get("fitness_cache_hits"),
    "binary_expression_stats": model.get_expression_stats(),
    "binary_scaling_exact_last_generation": model.history_[-1].get("scaling_exact_programs"),
    "multiclass_strategy": multi.multiclass_strategy_,
    "transformer_components": transformer.n_components_,
    "status": "ok",
}, indent=2))
