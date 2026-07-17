from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from fastsymbolicgp import FastSymbolicClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = FastSymbolicClassifier(
    population_size=24,
    generations=80,
    multiclass_strategy="shared_softmax",
    shared_n_components=8,
    shared_max_correlation=0.95,
    multiclass_calibration="temperature",
    optimization="nsga2",
    evolution_model="islands",
    n_islands=3,
    subtree_cache=True,
    display="grid",
    dashboard_interval=10,
    random_state=42,
).fit(X_train, y_train)

probability = model.predict_proba(X_test)
print(classification_report(y_test, model.predict(X_test)))
print("Multiclass ROC-AUC:", roc_auc_score(y_test, probability, multi_class="ovr"))
print("Multiclass log loss:", log_loss(y_test, probability))
print("Calibration:", model.multiclass_calibration_report_)
print("Shared expressions:", model.shared_expressions_)
print("Statistics:", model.get_expression_stats())
