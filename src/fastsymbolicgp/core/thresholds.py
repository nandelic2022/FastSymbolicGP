import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)


def _score_binary_metric(y_true, y_pred, metric):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if metric == "mcc":
        return matthews_corrcoef(y_true, y_pred)
    raise ValueError(f"Unknown threshold metric: {metric}")


def optimize_binary_threshold(scores, y_binary, metric="balanced_accuracy", n_thresholds=101):
    scores = np.asarray(scores, dtype=np.float64)
    y_binary = np.asarray(y_binary, dtype=np.int64)

    if len(np.unique(scores)) < 2:
        return 0.0, 1, 0.0

    qs = np.linspace(0.0, 1.0, n_thresholds)
    thresholds = np.unique(np.quantile(scores, qs))

    best_value = -1e18
    best_thr = float(thresholds[0])
    best_direction = 1

    for thr in thresholds:
        pred_pos = (scores >= thr).astype(np.int64)
        value_pos = _score_binary_metric(y_binary, pred_pos, metric)

        pred_neg = (scores < thr).astype(np.int64)
        value_neg = _score_binary_metric(y_binary, pred_neg, metric)

        if value_pos > best_value:
            best_value = float(value_pos)
            best_thr = float(thr)
            best_direction = 1

        if value_neg > best_value:
            best_value = float(value_neg)
            best_thr = float(thr)
            best_direction = -1

    return best_thr, int(best_direction), float(best_value)
