from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from fastsymbolicgp import FastSymbolicEnsembleClassifier


data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = FastSymbolicEnsembleClassifier(
    n_estimators=3,
    population_size=120,
    generations=8,
    function_set="fast",
    random_state=42,
    verbose=1,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Balanced accuracy:", balanced_accuracy_score(y_test, pred))
print("Expressions:")
for expr in model.get_expressions(feature_names=data.feature_names):
    print(expr)
