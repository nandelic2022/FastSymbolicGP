from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from fastsymbolicgp import FastSymbolicEnsembleClassifier, FastSymbolicEnsembleRegressor


def test_ensemble_classifier_runs():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
    )

    model = FastSymbolicEnsembleClassifier(
        n_estimators=2,
        population_size=15,
        generations=1,
        function_set="fast",
        random_state=42,
        verbose=0,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    assert len(pred) == len(y_test)
    assert len(model.get_expressions()) == 2


def test_ensemble_regressor_runs():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )

    model = FastSymbolicEnsembleRegressor(
        n_estimators=2,
        population_size=15,
        generations=1,
        function_set="fast",
        random_state=42,
        verbose=0,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    assert len(pred) == len(y_test)
    assert len(model.get_expressions()) == 2
