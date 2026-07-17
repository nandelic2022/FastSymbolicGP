from sklearn.datasets import load_breast_cancer
from fastsymbolicgp import FastSymbolicClassifier

X, y = load_breast_cancer(return_X_y=True)
base = FastSymbolicClassifier(
    population_size=30,
    generations=40,
    checkpoint_path="checkpoints_v070/base",
    checkpoint_interval=5,
    optimization="nsga2",
    random_state=42,
    verbose=0,
).fit(X, y)

compact = base.branch(
    generations=80,
    parsimony_target_nodes=20,
    selection_tolerance=0.03,
)
compact.continue_evolution(X, y, additional_generations=40)
print("Base stats:", base.get_expression_stats())
print("Compact branch stats:", compact.get_expression_stats())
