"""Primitive functions used by symbolic programs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

Array = np.ndarray


def _finite(x: Array) -> Array:
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


def protected_div(a: Array, b: Array) -> Array:
    out = np.ones_like(a, dtype=float)
    np.divide(a, b, out=out, where=np.abs(b) > 1e-12)
    return _finite(out)


def protected_log(a: Array) -> Array:
    return _finite(np.log(np.abs(a) + 1e-12))


def protected_sqrt(a: Array) -> Array:
    return _finite(np.sqrt(np.abs(a)))


def protected_exp(a: Array) -> Array:
    return _finite(np.exp(np.clip(a, -30.0, 30.0)))


def is_missing(a: Array) -> Array:
    return np.isnan(a).astype(float)


def coalesce(a: Array, b: Array) -> Array:
    return _finite(np.where(np.isnan(a), b, a))


def protected_inv(a: Array) -> Array:
    out = np.zeros_like(a, dtype=float)
    np.divide(1.0, a, out=out, where=np.abs(a) > 1e-12)
    return _finite(out)


def sigmoid(a: Array) -> Array:
    a = np.clip(a, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-a))


@dataclass(frozen=True)
class Primitive:
    name: str
    arity: int
    function: Callable[..., Array]


PRIMITIVES: dict[str, Primitive] = {
    "add": Primitive("add", 2, lambda a, b: _finite(a + b)),
    "sub": Primitive("sub", 2, lambda a, b: _finite(a - b)),
    "mul": Primitive("mul", 2, lambda a, b: _finite(a * b)),
    "div": Primitive("div", 2, protected_div),
    "max": Primitive("max", 2, lambda a, b: np.maximum(a, b)),
    "min": Primitive("min", 2, lambda a, b: np.minimum(a, b)),
    "sin": Primitive("sin", 1, lambda a: _finite(np.sin(a))),
    "cos": Primitive("cos", 1, lambda a: _finite(np.cos(a))),
    "tanh": Primitive("tanh", 1, lambda a: _finite(np.tanh(a))),
    "abs": Primitive("abs", 1, lambda a: np.abs(a)),
    "neg": Primitive("neg", 1, lambda a: -a),
    "log": Primitive("log", 1, protected_log),
    "sqrt": Primitive("sqrt", 1, protected_sqrt),
    "exp": Primitive("exp", 1, protected_exp),
    "inv": Primitive("inv", 1, protected_inv),
    "is_missing": Primitive("is_missing", 1, is_missing),
    "coalesce": Primitive("coalesce", 2, coalesce),
}

DEFAULT_FUNCTION_SET = ("add", "sub", "mul", "div")


def validate_function_set(function_set: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    names = tuple(function_set)
    unknown = sorted(set(names).difference(PRIMITIVES))
    if unknown:
        raise ValueError(f"Unknown primitives: {unknown}. Available: {sorted(PRIMITIVES)}")
    if not names:
        raise ValueError("function_set must contain at least one primitive")
    return names
