from sklearn.datasets import load_diabetes
from fastsymbolicgp import FastSymbolicRegressor

X, y = load_diabetes(return_X_y=True)
model = FastSymbolicRegressor(
    population_size=96,
    generations=120,
    selection_method="epsilon_lexicase",
    optimization="pareto",
    constraints={"output_min": 0.0, "max_variables": 5},
    uncertainty="split_conformal",
    random_state=42,
)
model.fit(X, y)
print(model.get_expression())
print(model.get_expression_stats())
print(model.coverage_report(X, y, confidence=0.95))
model.export_reproducibility_bundle("v110_reproducibility_bundle")
