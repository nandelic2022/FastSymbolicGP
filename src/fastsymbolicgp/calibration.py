"""Binary and multiclass probability calibration helpers."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold


class BetaCalibrator(BaseEstimator):
    def __init__(self, C=1.0, random_state=None):
        self.C = C
        self.random_state = random_state

    @staticmethod
    def _features(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
        return np.column_stack((np.log(p), -np.log1p(-p)))

    def fit(self, p, y, sample_weight=None):
        self.model_ = LogisticRegression(C=float(self.C), solver="lbfgs", random_state=self.random_state, max_iter=500)
        self.model_.fit(self._features(p), y, sample_weight=sample_weight)
        return self

    def predict(self, p):
        return self.model_.predict_proba(self._features(p))[:, 1]


class PlattCalibrator(BaseEstimator):
    def __init__(self, C=1.0, random_state=None):
        self.C = C
        self.random_state = random_state

    def fit(self, p, y, sample_weight=None):
        self.model_ = LogisticRegression(C=float(self.C), solver="lbfgs", random_state=self.random_state, max_iter=500)
        self.model_.fit(np.asarray(p).reshape(-1, 1), y, sample_weight=sample_weight)
        return self

    def predict(self, p):
        return self.model_.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


class TemperatureScaler(BaseEstimator):
    """Single-parameter multiclass temperature scaling.

    The input is a probability matrix. Internally it is converted to clipped
    log-probabilities and a positive temperature is optimized by a robust grid
    search followed by local refinement. This has no SciPy dependency.
    """

    def __init__(self, min_temperature=0.05, max_temperature=20.0):
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    @staticmethod
    def _softmax(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(np.clip(logits, -50.0, 50.0))
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, probabilities, y, sample_weight=None):
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        encoded = np.searchsorted(self.classes_, y)
        logp = np.log(probabilities)
        weights = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)

        def loss(temp):
            p = self._softmax(logp / float(temp))
            return float(-np.sum(weights * np.log(np.clip(p[np.arange(len(y)), encoded], 1e-12, 1.0))) / max(weights.sum(), 1.0))

        grid = np.exp(np.linspace(np.log(float(self.min_temperature)), np.log(float(self.max_temperature)), 121))
        losses = np.asarray([loss(t) for t in grid])
        best = float(grid[int(np.argmin(losses))])
        for _ in range(4):
            local = np.linspace(max(float(self.min_temperature), best / 1.8), min(float(self.max_temperature), best * 1.8), 61)
            local_losses = np.asarray([loss(t) for t in local])
            best = float(local[int(np.argmin(local_losses))])
        self.temperature_ = best
        self.training_log_loss_ = loss(best)
        return self

    def predict_proba(self, probabilities):
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return self._softmax(np.log(probabilities) / float(self.temperature_))


def expected_calibration_error(y, probabilities, n_bins=10) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    y = np.asarray(y)
    if probabilities.ndim == 1 or probabilities.shape[1] == 2:
        p = probabilities if probabilities.ndim == 1 else probabilities[:, 1]
        pred = (p >= 0.5).astype(int)
        confidence = np.where(pred == 1, p, 1 - p)
        correct = (pred == y).astype(float)
    else:
        pred = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        classes = np.unique(y)
        encoded = np.searchsorted(classes, y)
        correct = (pred == encoded).astype(float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if np.any(mask):
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def multiclass_brier_score(y, probabilities, classes=None) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    encoded = np.searchsorted(classes, y)
    target = np.zeros_like(probabilities)
    target[np.arange(len(y)), encoded] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def _make_calibrator(method, random_state=None):
    method = str(method).lower()
    if method in {"platt", "sigmoid", "cross_fitted_platt"}:
        return PlattCalibrator(C=1.0, random_state=random_state)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip")
    if method == "beta":
        return BetaCalibrator(C=1.0, random_state=random_state)
    if method in {"none", "identity"}:
        return None
    raise ValueError(f"Unsupported probability calibration: {method}")


def apply_calibrator(model, p):
    if model is None:
        return np.asarray(p, dtype=float)
    return np.asarray(model.predict(p), dtype=float)


def fit_calibrator(p, y, method="none", sample_weight=None, cv=3, random_state=None):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=int)
    method = str(method).lower()
    candidates = ["none", "platt", "isotonic", "beta"] if method == "auto" else [method]
    reports = {}
    best_method, best_oof = None, None
    best_loss = np.inf
    min_class = int(np.bincount(y).min()) if y.size else 0
    n_splits = max(2, min(int(cv), min_class)) if min_class >= 2 else 0

    for candidate in candidates:
        if candidate in {"none", "identity"} or n_splits == 0:
            oof = p.copy()
        else:
            oof = np.empty_like(p)
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_idx, test_idx in splitter.split(p, y):
                model = _make_calibrator(candidate, random_state)
                model.fit(p[train_idx], y[train_idx], sample_weight=None if sample_weight is None else np.asarray(sample_weight)[train_idx])
                oof[test_idx] = apply_calibrator(model, p[test_idx])
        oof = np.clip(oof, 1e-6, 1 - 1e-6)
        ll = float(log_loss(y, oof, sample_weight=sample_weight, labels=[0, 1]))
        bs = float(brier_score_loss(y, oof, sample_weight=sample_weight))
        reports[candidate] = {"oof_log_loss": ll, "oof_brier_score": bs}
        if ll < best_loss:
            best_loss, best_method, best_oof = ll, candidate, oof

    if method == "auto":
        baseline = reports["none"]["oof_log_loss"]
        required_gain = max(0.005, 0.05 * baseline)
        eligible = []
        for candidate, stats in reports.items():
            if candidate == "isotonic" and len(y) < 200:
                continue
            if stats["oof_log_loss"] <= baseline - required_gain:
                eligible.append((stats["oof_log_loss"], candidate))
        best_method = min(eligible)[1] if eligible else "none"
        best_oof = p.copy() if best_method == "none" else best_oof

    final = _make_calibrator(best_method, random_state)
    if final is not None:
        final.fit(p, y, sample_weight=sample_weight)
    report = {
        "requested_method": method,
        "selected_method": best_method,
        "cv_splits": n_splits,
        "candidates": reports,
    }
    return final, report, best_oof
