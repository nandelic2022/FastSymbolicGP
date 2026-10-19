"""CCPP-style symbolic regression example for FastSymbolicGP 1.1.0."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from fastsymbolicgp import FastSymbolicRegressor

# Expected columns: AT, V, AP, RH, PE
DATASET = "CCPP.csv"
df = pd.read_csv(DATASET)
X = df[["AT", "V", "AP", "RH"]]
y = df["PE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)

model = FastSymbolicRegressor(
    preset="balanced",
    selection_method="epsilon_lexicase",
    fitness_metric="rmse",
    fitness_scaling="linear",
    evaluation_backend="auto",
    fitness_engine="auto",
    uncertainty="split_conformal",
    constraints={"max_variables": 4},
    random_state=42,
    verbose=1,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("R2:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** .5)
print("Expression:", model.get_expression())
print("Resolved:", model.get_resolved_config())
print("Execution plan:", model.execution_plan_)
