from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from fastsymbolicgp import FastSymbolicRegressor


data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = FastSymbolicRegressor(
    population_size=300,
    generations=20,
    max_depth=4,
    function_set="default",
    subsample=0.8,
    random_state=42,
    verbose=1,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nBest expression:")
print(model.get_expression(feature_names=data.feature_names))

print("\nMetrics:")
print(f"R2:   {r2_score(y_test, pred):.6f}")
print(f"MSE:  {mean_squared_error(y_test, pred):.6f}")
print(f"MAE:  {mean_absolute_error(y_test, pred):.6f}")

model.save_elite_expressions("elite_regression.csv", feature_names=data.feature_names, n=20)
model.save_history("history_regression.csv")
print("\nSaved elite_regression.csv and history_regression.csv")
