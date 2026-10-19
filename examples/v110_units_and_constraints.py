import numpy as np
from fastsymbolicgp import FastSymbolicRegressor

rng = np.random.default_rng(42)
X = rng.uniform(0.2, 3.0, size=(500, 2))
# x0: velocity [m/s], x1: time [s], target: distance [m]
y = X[:, 0] * X[:, 1]

model = FastSymbolicRegressor(
    feature_units={"X0": "m/s", "X1": "s"},
    target_unit="m",
    unit_policy="enforce",
    function_set=("add", "sub", "mul", "div"),
    constraints={"output_min": 0.0, "max_variables": 2},
    selection_method="epsilon_lexicase",
    population_size=100,
    generations=100,
    random_state=42,
)
model.fit(X, y)
print(model.get_expression())
print(model.explain())
