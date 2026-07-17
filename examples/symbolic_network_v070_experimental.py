from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from fastsymbolicgp import FastSymbolicNetworkClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = FastSymbolicNetworkClassifier(
    symbolic_layers=(8, 4),
    inherit_original_features=False,
    transformer_params={
        "population_size": 20,
        "generations": 30,
        "optimization": "nsga2",
        "verbose": 0,
    },
    random_state=42,
    verbose=1,
).fit(X_train, y_train)

print("Balanced accuracy:", balanced_accuracy_score(y_test, model.predict(X_test)))
print("Network statistics:", model.get_network_stats())
print("Layer expressions:", model.get_layer_expressions())
