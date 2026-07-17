from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from fastsymbolicgp import FastSymbolicTransformer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("symbolic", FastSymbolicTransformer(
        n_components=12,
        component_selection="mrmr",
        max_correlation=0.90,
        include_original_features=True,
        model_params={
            "population_size": 30,
            "generations": 50,
            "optimization": "nsga2",
            "subtree_cache": True,
            "verbose": 0,
        },
        random_state=42,
        verbose=1,
    )),
    ("classifier", LogisticRegression(max_iter=2000)),
])
pipeline.fit(X_train, y_train)
print("Balanced accuracy:", balanced_accuracy_score(y_test, pipeline.predict(X_test)))
print("Symbolic expressions:", pipeline.named_steps["symbolic"].get_expressions())
print("Transformer report:", pipeline.named_steps["symbolic"].get_component_report())
