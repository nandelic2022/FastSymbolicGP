from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from fastsymbolicgp import FastSymbolicClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

teacher = FastSymbolicClassifier(
    population_size=32,
    generations=80,
    prediction_mode="symbolic_ensemble",
    ensemble_size=5,
    optimization="nsga2",
    random_state=42,
    verbose=0,
).fit(X_train, y_train)

distilled = teacher.distill(
    X_train,
    max_nodes=35,
    population_size=40,
    generations=60,
    verbose=0,
)
print("Teacher BACC:", balanced_accuracy_score(y_test, teacher.predict(X_test)))
print("Distilled BACC:", balanced_accuracy_score(y_test, distilled.predict(X_test)))
print("Distilled stats:", distilled.get_expression_stats())

out = Path("deployment_v070")
out.mkdir(exist_ok=True)
teacher.export_python(out / "fastsymbolic_model.py")
teacher.export_c(out / "fastsymbolic_model.c")
teacher.export_java(out / "FastSymbolicModel.java")
teacher.export_kotlin(out / "FastSymbolicModel.kt")
teacher.export_javascript(out / "fastsymbolic_model.js")
print("Deployment files saved to:", out.resolve())
