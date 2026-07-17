"""Metrics, scaling, calibration helpers, and exact threshold search."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss,
    matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score,
)
from .functions import sigmoid


def resolve_class_weight(y: np.ndarray, class_weight) -> np.ndarray:
    if class_weight is None:
        return np.ones(y.shape[0], dtype=float)
    classes, counts = np.unique(y, return_counts=True)
    if class_weight == "balanced":
        mapping = {c: len(y) / (len(classes) * count) for c, count in zip(classes, counts)}
    elif isinstance(class_weight, dict):
        mapping = class_weight
    else:
        raise ValueError("class_weight must be None, 'balanced', or a dict")
    return np.asarray([float(mapping.get(v, 1.0)) for v in y], dtype=float)


def fit_logistic_scaler(raw: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
                        regularization: float = 1e-4, max_iter: int = 20,
                        initial: tuple[float, float] | None = None) -> tuple[float, float]:
    x = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if np.nanstd(x) < 1e-14:
        prior = np.clip(np.average(y, weights=w), 1e-8, 1 - 1e-8)
        return 0.0, float(np.log(prior / (1.0 - prior)))
    x_mean = np.average(x, weights=w)
    x_std = np.sqrt(np.average((x - x_mean) ** 2, weights=w)) + 1e-12
    z = (x - x_mean) / x_std
    a, b = initial if initial is not None else (1.0, 0.0)
    for _ in range(max_iter):
        p = sigmoid(a * z + b)
        q = np.maximum(p * (1.0 - p), 1e-9)
        g_a = np.sum(w * (p - y) * z) + regularization * a
        g_b = np.sum(w * (p - y))
        h_aa = np.sum(w * q * z * z) + regularization
        h_ab = np.sum(w * q * z)
        h_bb = np.sum(w * q) + 1e-12
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-18:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (-h_ab * g_a + h_aa * g_b) / det
        a -= float(np.clip(step_a, -5.0, 5.0))
        b -= float(np.clip(step_b, -5.0, 5.0))
        if abs(step_a) + abs(step_b) < 1e-8:
            break
    # Convert standardized coefficients back to raw space.
    raw_a = float(np.clip(a / x_std, -1e6, 1e6))
    raw_b = float(np.clip(b - a * x_mean / x_std, -1e6, 1e6))
    return raw_a, raw_b


def probability_from_raw(raw: np.ndarray, scale: float, intercept: float) -> np.ndarray:
    return np.clip(sigmoid(scale * raw + intercept), 1e-12, 1 - 1e-12)


def classification_loss(y: np.ndarray, proba: np.ndarray, metric: str,
                        sample_weight: np.ndarray | None = None,
                        threshold: float = 0.5) -> float:
    metric = metric.lower()
    pred = (proba >= threshold).astype(int)
    if metric in {"log_loss", "logloss"}:
        p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0 - 1e-12)
        yy = np.asarray(y, dtype=float)
        losses = -(yy * np.log(p) + (1.0 - yy) * np.log1p(-p))
        if sample_weight is None:
            return float(np.mean(losses))
        weights = np.asarray(sample_weight, dtype=float)
        return float(np.average(losses, weights=weights))
    if metric in {"balanced_accuracy", "bacc"}:
        return 1.0 - float(balanced_accuracy_score(y, pred, sample_weight=sample_weight))
    if metric == "accuracy":
        return 1.0 - float(accuracy_score(y, pred, sample_weight=sample_weight))
    if metric in {"f1", "f1_score"}:
        return 1.0 - float(f1_score(y, pred, sample_weight=sample_weight, zero_division=0))
    if metric in {"mcc", "matthews"}:
        return 1.0 - (float(matthews_corrcoef(y, pred, sample_weight=sample_weight)) + 1.0) / 2.0
    if metric in {"roc_auc", "auc"}:
        try: return 1.0 - float(roc_auc_score(y, proba, sample_weight=sample_weight))
        except ValueError: return 1.0
    if metric in {"average_precision", "ap"}:
        try: return 1.0 - float(average_precision_score(y, proba, sample_weight=sample_weight))
        except ValueError: return 1.0
    raise ValueError(f"Unsupported classification metric: {metric}")


def regression_loss(y: np.ndarray, pred: np.ndarray, metric: str,
                    sample_weight: np.ndarray | None = None) -> float:
    metric = metric.lower()
    if metric in {"mse", "mean_squared_error"}:
        return float(mean_squared_error(y, pred, sample_weight=sample_weight))
    if metric in {"rmse", "root_mean_squared_error"}:
        return float(np.sqrt(mean_squared_error(y, pred, sample_weight=sample_weight)))
    if metric in {"mae", "mean_absolute_error"}:
        return float(mean_absolute_error(y, pred, sample_weight=sample_weight))
    if metric in {"r2", "r2_score"}:
        return 1.0 - float(r2_score(y, pred, sample_weight=sample_weight))
    raise ValueError(f"Unsupported regression metric: {metric}")


def fit_linear_scaler(raw: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[float, float]:
    x = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    variance = np.average((x - mx) ** 2, weights=w)
    if variance < 1e-16:
        return 0.0, float(my)
    covariance = np.average((x - mx) * (y - my), weights=w)
    scale = covariance / variance
    return float(scale), float(my - scale * mx)


def optimize_threshold(y: np.ndarray, proba: np.ndarray, metric: str = "mcc",
                       sample_weight: np.ndarray | None = None) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(proba, dtype=float)
    w = np.ones_like(p) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    ys, ps, ws = y[order], p[order], w[order]
    total_pos = float(ws[ys == 1].sum())
    total_neg = float(ws[ys == 0].sum())
    tp = fp = 0.0
    best_score = -np.inf
    best_threshold = 0.5

    def score_counts(tp_, fp_):
        fn_ = total_pos - tp_
        tn_ = total_neg - fp_
        if metric in {"balanced_accuracy", "bacc"}:
            tpr = tp_ / total_pos if total_pos else 0.0
            tnr = tn_ / total_neg if total_neg else 0.0
            return 0.5 * (tpr + tnr)
        if metric in {"f1", "f1_score"}:
            den = 2 * tp_ + fp_ + fn_
            return 2 * tp_ / den if den else 0.0
        if metric == "accuracy":
            den = total_pos + total_neg
            return (tp_ + tn_) / den if den else 0.0
        # MCC default
        den = (tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_)
        return (tp_ * tn_ - fp_ * fn_) / np.sqrt(den) if den > 0 else 0.0

    index = 0
    while index < len(ps):
        value = ps[index]
        while index < len(ps) and ps[index] == value:
            if ys[index] == 1: tp += ws[index]
            else: fp += ws[index]
            index += 1
        next_value = ps[index] if index < len(ps) else 0.0
        threshold = float((value + next_value) / 2.0)
        score = score_counts(tp, fp)
        if score > best_score + 1e-15 or (abs(score - best_score) <= 1e-15 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score, best_threshold = score, threshold
    return float(np.clip(best_threshold, 0.0, 1.0)), float(best_score)
