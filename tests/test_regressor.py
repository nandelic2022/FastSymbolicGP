from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from fastsymbolicgp import FastSymbolicRegressor


def test_regressor_runs():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )

    model = FastSymbolicRegressor(
        population_size=30,
        generations=2,
        max_depth=3,
        function_set="fast",
        random_state=42,
        verbose=0,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    assert len(pred) == len(y_test)
    assert isinstance(model.get_expression(), str)
