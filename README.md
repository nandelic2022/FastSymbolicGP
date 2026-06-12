# FastSymbolicGP

FastSymbolicGP is a standalone symbolic genetic programming library focused on speed, compact symbolic expressions, and sklearn-style usability.

It is designed around:

- postfix / bytecode symbolic expression representation
- fast expression evaluation with optional Numba acceleration
- NumPy fallback evaluator
- binary symbolic classification
- symbolic regression
- one-vs-rest multiclass symbolic classification
- symbolic ensemble classifiers and regressors
- subtree mutation
- subtree crossover
- point mutation
- elite expression export
- symbolic expression strings
- simple LaTeX export helper
- sklearn-style `.fit()`, `.predict()`, `.score()` API

## Installation

From the project root:

```bash
python -m pip install -e .[dev]
```

## Quick binary classification example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastsymbolicgp import FastSymbolicClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = FastSymbolicClassifier(
    population_size=300,
    generations=20,
    max_depth=4,
    random_state=42,
)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.get_expression(feature_names=data.feature_names))
```

## Regressor

```python
from fastsymbolicgp import FastSymbolicRegressor

reg = FastSymbolicRegressor(population_size=300, generations=20, random_state=42)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
```

## Multiclass classifier

```python
from fastsymbolicgp import FastSymbolicMultiClassifier

clf = FastSymbolicMultiClassifier(population_size=200, generations=15, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```

## Ensemble classifier

```python
from fastsymbolicgp import FastSymbolicEnsembleClassifier

ens = FastSymbolicEnsembleClassifier(n_estimators=5, population_size=150, generations=10)
ens.fit(X_train, y_train)
pred = ens.predict(X_test)
```

## Export elite expressions

```python
model.save_elite_expressions("elite_expressions.csv", feature_names=data.feature_names, n=50)
```

## Current status

This is an alpha research library. It is suitable for experimentation, benchmarking, and continued development.
