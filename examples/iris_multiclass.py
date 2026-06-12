from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from fastsymbolicgp import FastSymbolicMultiClassifier


data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = FastSymbolicMultiClassifier(
    population_size=150,
    generations=12,
    max_depth=4,
    function_set="fast",
    subsample=0.8,
    random_state=42,
    verbose=1,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nMetrics:")
print(f"Accuracy:          {accuracy_score(y_test, pred):.6f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, pred):.6f}")
print("\nReport:")
print(classification_report(y_test, pred))

print("\nExpressions:")
for cls, expr in model.get_expressions(feature_names=data.feature_names).items():
    print(f"Class {cls}: {expr}")
