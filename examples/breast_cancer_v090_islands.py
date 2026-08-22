from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from fastsymbolicgp import FastSymbolicClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = FastSymbolicClassifier(
    population_size=32,
    generations=120,
    init_depth=(2, 10),
    max_depth=16,
    max_nodes=383,
    optimization="nsga2",
    evolution_model="islands",
    n_islands=4,
    island_profiles=("accurate", "compact", "diverse", "robust"),
    migration_interval=10,
    migration_size=2,
    prediction_mode="symbolic_ensemble",
    ensemble_size=5,
    probability_calibration="auto",
    threshold_strategy="balanced_accuracy",
    subtree_cache=True,
    dag_execution="auto",
    parsimony="adaptive",
    parsimony_target_nodes=40,
    display="grid",
    dashboard_interval=5,
    random_state=42,
)
model.fit(X_train, y_train)

probability = model.predict_proba(X_test)[:, 1]
prediction = model.predict(X_test)
print("Balanced accuracy:", balanced_accuracy_score(y_test, prediction))
print("ROC-AUC:", roc_auc_score(y_test, probability))
print("Log loss:", log_loss(y_test, probability))
print("Execution engine:", model.execution_engine_)
print("Expression:", model.get_expression())
print("Expression statistics:", model.get_expression_stats())
print("Cache statistics:", model.cache_statistics_)
print("Prediction profile:", model.profile_prediction(X_test[:100]))
