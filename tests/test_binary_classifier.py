from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from fastsymbolicgp import FastSymbolicClassifier


def test_binary_classifier_runs():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
    )

    model = FastSymbolicClassifier(
        population_size=30,
        generations=2,
        max_depth=3,
        function_set="fast",
        random_state=42,
        verbose=0,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    assert len(pred) == len(y_test)
    assert proba.shape == (len(y_test), 2)
    assert isinstance(model.get_expression(), str)
