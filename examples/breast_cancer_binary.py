from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

from fastsymbolicgp import FastSymbolicClassifier


data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = FastSymbolicClassifier(
    population_size=300,
    generations=20,
    max_depth=4,
    function_set="fast",
    subsample=0.75,
    random_state=42,
    verbose=1,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nBest expression:")
print(model.get_expression(feature_names=data.feature_names))

print("\nMetrics:")
print(f"Accuracy:          {accuracy_score(y_test, pred):.6f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, pred):.6f}")
print(f"F1-score:          {f1_score(y_test, pred):.6f}")
print(f"MCC:               {matthews_corrcoef(y_test, pred):.6f}")

model.save_elite_expressions("elite_binary.csv", feature_names=data.feature_names, n=20)
model.save_history("history_binary.csv")
print("\nSaved elite_binary.csv and history_binary.csv")
