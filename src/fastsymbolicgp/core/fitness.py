import numpy as np


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.size == 0:
        return 0.0

    da = a - np.mean(a)
    db = b - np.mean(b)

    va = float(np.dot(da, da))
    vb = float(np.dot(db, db))

    if va < 1e-12 or vb < 1e-12:
        return 0.0

    return float(np.dot(da, db) / np.sqrt(va * vb))


def r2_score_fast(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot < 1e-12:
        return 0.0

    return 1.0 - ss_res / ss_tot


def rmse_fast(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae_fast(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def fit_linear_scaling(raw_score, y):
    raw_score = np.asarray(raw_score, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    var = float(np.var(raw_score))

    if var < 1e-12:
        return 0.0, float(np.mean(y))

    cov = float(np.mean((raw_score - np.mean(raw_score)) * (y - np.mean(y))))
    a = cov / var
    b = float(np.mean(y) - a * np.mean(raw_score))

    return float(a), float(b)
