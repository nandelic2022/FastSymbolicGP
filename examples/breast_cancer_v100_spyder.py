# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from fastsymbolicgp import FastSymbolicClassifier, warmup

warmup("classification")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = FastSymbolicClassifier(
    population_size=64, generations=120, preset="balanced",
    random_state=42, verbose=1,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Balanced accuracy:", balanced_accuracy_score(y_test, pred))
print("Expression:", model.get_expression())
print("Profile:", model.profile_)
