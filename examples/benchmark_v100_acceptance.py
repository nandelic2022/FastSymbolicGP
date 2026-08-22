from time import perf_counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from fastsymbolicgp import FastSymbolicClassifier, warmup

warmup("classification")
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
model = FastSymbolicClassifier(
    population_size=64, generations=120, tournament_size=10,
    max_depth=14, max_nodes=255, validation_fraction=0.20,
    n_islands=2, evolution_model="islands", island_parallel=True,
    evaluation_backend="population_numba", fitness_engine="fused",
    fitness_scaling="elite_logistic", fitness_scaling_top_k=12,
    fitness_scaling_interval=2, patience=0, prediction_mode="best",
    probability_calibration="none", threshold_strategy="fixed",
    preset=None, display="off", verbose=0, random_state=42,
)
t0 = perf_counter(); model.fit(Xtr, ytr); elapsed = perf_counter() - t0
print(f"V1.0.0 forced 120 generations: {elapsed:.4f} s")
print(f"Balanced accuracy: {balanced_accuracy_score(yte, model.predict(Xte)):.6f}")
print(model.profile_)
