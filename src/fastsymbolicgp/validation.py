"""Compatibility wrappers for scikit-learn validation API renames."""
from __future__ import annotations

from sklearn.utils.validation import check_array, check_X_y


def _finite_keyword(allow_nan: bool):
    return "allow-nan" if allow_nan else True


def check_array_finite(*args, allow_nan=False, **kwargs):
    value = _finite_keyword(bool(allow_nan))
    try:
        return check_array(*args, ensure_all_finite=value, **kwargs)
    except TypeError:
        return check_array(*args, force_all_finite=value, **kwargs)


def check_X_y_finite(*args, allow_nan=False, **kwargs):
    value = _finite_keyword(bool(allow_nan))
    try:
        return check_X_y(*args, ensure_all_finite=value, **kwargs)
    except TypeError:
        return check_X_y(*args, force_all_finite=value, **kwargs)
