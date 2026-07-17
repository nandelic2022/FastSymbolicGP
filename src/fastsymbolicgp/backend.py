"""Compiled postfix execution backend used by FastSymbolicGP V0.6.0."""
from __future__ import annotations

import math
import numpy as np

# terminal opcodes
FEATURE = 0
CONSTANT = 1
# binary
ADD, SUB, MUL, DIV, MAXIMUM, MINIMUM, COALESCE = 10, 11, 12, 13, 14, 15, 16
# unary
SIN, COS, TANH, ABS, NEG, LOG, SQRT, EXP, INV, IS_MISSING = 20, 21, 22, 23, 24, 25, 26, 27, 28, 29

NAME_TO_OPCODE = {
    "add": ADD, "sub": SUB, "mul": MUL, "div": DIV,
    "max": MAXIMUM, "min": MINIMUM, "coalesce": COALESCE,
    "sin": SIN, "cos": COS, "tanh": TANH, "abs": ABS,
    "neg": NEG, "log": LOG, "sqrt": SQRT, "exp": EXP, "inv": INV,
    "is_missing": IS_MISSING,
}

try:
    from numba import njit

    @njit(cache=True)
    def _execute_postfix_numba(X, opcodes, arguments):
        n_samples = X.shape[0]
        n_ops = opcodes.shape[0]
        out = np.empty(n_samples, dtype=np.float64)
        stack = np.empty(n_ops, dtype=np.float64)
        for i in range(n_samples):
            sp = 0
            for j in range(n_ops):
                op = opcodes[j]
                arg = arguments[j]
                if op == FEATURE:
                    value = X[i, int(arg)]
                    stack[sp] = value
                    sp += 1
                elif op == CONSTANT:
                    stack[sp] = arg
                    sp += 1
                elif op >= 10 and op <= 16:
                    b = stack[sp - 1]
                    a = stack[sp - 2]
                    sp -= 2
                    if op == ADD:
                        value = a + b
                    elif op == SUB:
                        value = a - b
                    elif op == MUL:
                        value = a * b
                    elif op == DIV:
                        value = 1.0 if abs(b) <= 1e-12 else a / b
                    elif op == MAXIMUM:
                        value = a if a >= b else b
                    elif op == MINIMUM:
                        value = a if a <= b else b
                    else:
                        value = b if math.isnan(a) else a
                    if not math.isfinite(value):
                        value = 0.0 if math.isnan(value) else (1e6 if value > 0 else -1e6)
                    if value > 1e6:
                        value = 1e6
                    elif value < -1e6:
                        value = -1e6
                    stack[sp] = value
                    sp += 1
                else:
                    a = stack[sp - 1]
                    sp -= 1
                    if op == SIN:
                        value = math.sin(a)
                    elif op == COS:
                        value = math.cos(a)
                    elif op == TANH:
                        value = math.tanh(a)
                    elif op == ABS:
                        value = abs(a)
                    elif op == NEG:
                        value = -a
                    elif op == LOG:
                        value = math.log(abs(a) + 1e-12)
                    elif op == SQRT:
                        value = math.sqrt(abs(a))
                    elif op == EXP:
                        value = math.exp(max(-30.0, min(30.0, a)))
                    elif op == INV:
                        value = 0.0 if abs(a) <= 1e-12 else 1.0 / a
                    else:
                        value = 1.0 if math.isnan(a) else 0.0
                    if not math.isfinite(value):
                        value = 0.0 if math.isnan(value) else (1e6 if value > 0 else -1e6)
                    if value > 1e6:
                        value = 1e6
                    elif value < -1e6:
                        value = -1e6
                    stack[sp] = value
                    sp += 1
            out[i] = stack[0]
        return out

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without numba
    NUMBA_AVAILABLE = False
    _execute_postfix_numba = None


def execute_postfix_numpy(X: np.ndarray, opcodes: np.ndarray, arguments: np.ndarray) -> np.ndarray:
    """Vectorized NumPy postfix interpreter."""
    stack: list[np.ndarray] = []
    n = X.shape[0]
    for op, arg in zip(opcodes, arguments):
        op = int(op)
        if op == FEATURE:
            stack.append(X[:, int(arg)])
        elif op == CONSTANT:
            stack.append(np.full(n, float(arg), dtype=float))
        elif 10 <= op <= 16:
            b = stack.pop(); a = stack.pop()
            with np.errstate(all="ignore"):
                if op == ADD: value = a + b
                elif op == SUB: value = a - b
                elif op == MUL: value = a * b
                elif op == DIV:
                    value = np.ones_like(a, dtype=float)
                    np.divide(a, b, out=value, where=np.abs(b) > 1e-12)
                elif op == MAXIMUM: value = np.maximum(a, b)
                elif op == MINIMUM: value = np.minimum(a, b)
                else: value = np.where(np.isnan(a), b, a)
            stack.append(np.clip(np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6))
        else:
            a = stack.pop()
            with np.errstate(all="ignore"):
                if op == SIN: value = np.sin(a)
                elif op == COS: value = np.cos(a)
                elif op == TANH: value = np.tanh(a)
                elif op == ABS: value = np.abs(a)
                elif op == NEG: value = -a
                elif op == LOG: value = np.log(np.abs(a) + 1e-12)
                elif op == SQRT: value = np.sqrt(np.abs(a))
                elif op == EXP: value = np.exp(np.clip(a, -30.0, 30.0))
                elif op == INV:
                    value = np.zeros_like(a, dtype=float)
                    np.divide(1.0, a, out=value, where=np.abs(a) > 1e-12)
                else: value = np.isnan(a).astype(float)
            stack.append(np.clip(np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6))
    return stack[-1]


def execute_postfix(X: np.ndarray, opcodes: np.ndarray, arguments: np.ndarray, backend: str = "numpy") -> np.ndarray:
    backend = str(backend).lower()
    if backend == "auto":
        backend = "numba" if NUMBA_AVAILABLE else "numpy"
    if backend == "numba":
        if not NUMBA_AVAILABLE:
            return execute_postfix_numpy(X, opcodes, arguments)
        return _execute_postfix_numba(X, opcodes, arguments)
    return execute_postfix_numpy(X, opcodes, arguments)
