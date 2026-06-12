import math
import numpy as np

from fastsymbolicgp.core.ops import (
    OP_VAR,
    OP_CONST,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_ABS,
    OP_NEG,
    OP_MIN,
    OP_MAX,
    OP_SQRT,
    OP_LOG,
    OP_SIN,
    OP_COS,
    OP_TANH,
    OP_SIGMOID,
    OP_SQUARE,
)

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    njit = None
    NUMBA_AVAILABLE = False


def _evaluate_program_numpy(ops, args, X):
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]
    stack = []

    for op, arg in zip(ops, args):
        op = int(op)

        if op == OP_VAR:
            stack.append(X[:, int(arg)].copy())

        elif op == OP_CONST:
            stack.append(np.full(n_samples, float(arg), dtype=np.float64))

        elif op == OP_ADD:
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)

        elif op == OP_SUB:
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)

        elif op == OP_MUL:
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)

        elif op == OP_DIV:
            b = stack.pop()
            a = stack.pop()
            stack.append(np.where(np.abs(b) < 1e-12, 1.0, a / b))

        elif op == OP_ABS:
            a = stack.pop()
            stack.append(np.abs(a))

        elif op == OP_NEG:
            a = stack.pop()
            stack.append(-a)

        elif op == OP_MIN:
            b = stack.pop()
            a = stack.pop()
            stack.append(np.minimum(a, b))

        elif op == OP_MAX:
            b = stack.pop()
            a = stack.pop()
            stack.append(np.maximum(a, b))

        elif op == OP_SQRT:
            a = stack.pop()
            stack.append(np.sqrt(np.abs(a)))

        elif op == OP_LOG:
            a = stack.pop()
            stack.append(np.log1p(np.abs(a)))

        elif op == OP_SIN:
            a = stack.pop()
            stack.append(np.sin(a))

        elif op == OP_COS:
            a = stack.pop()
            stack.append(np.cos(a))

        elif op == OP_TANH:
            a = stack.pop()
            stack.append(np.tanh(a))

        elif op == OP_SIGMOID:
            a = stack.pop()
            a = np.clip(a, -60.0, 60.0)
            stack.append(1.0 / (1.0 + np.exp(-a)))

        elif op == OP_SQUARE:
            a = stack.pop()
            stack.append(a * a)

        else:
            raise ValueError(f"Unknown operation code: {op}")

    if len(stack) != 1:
        raise ValueError("Invalid postfix program.")

    out = stack[0]
    out = np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
    return out.astype(np.float64, copy=False)


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _evaluate_program_numba(ops, args, X):
        n_samples = X.shape[0]
        max_stack = len(ops)
        stack = np.empty((max_stack, n_samples), dtype=np.float64)
        sp = 0

        for i in range(len(ops)):
            op = ops[i]
            arg = args[i]

            if op == OP_VAR:
                col = int(arg)
                for j in range(n_samples):
                    stack[sp, j] = X[j, col]
                sp += 1

            elif op == OP_CONST:
                for j in range(n_samples):
                    stack[sp, j] = arg
                sp += 1

            elif op == OP_ADD:
                for j in range(n_samples):
                    stack[sp - 2, j] = stack[sp - 2, j] + stack[sp - 1, j]
                sp -= 1

            elif op == OP_SUB:
                for j in range(n_samples):
                    stack[sp - 2, j] = stack[sp - 2, j] - stack[sp - 1, j]
                sp -= 1

            elif op == OP_MUL:
                for j in range(n_samples):
                    stack[sp - 2, j] = stack[sp - 2, j] * stack[sp - 1, j]
                sp -= 1

            elif op == OP_DIV:
                for j in range(n_samples):
                    numerator = stack[sp - 2, j]
                    denominator = stack[sp - 1, j]
                    if abs(denominator) < 1e-12:
                        stack[sp - 2, j] = 1.0
                    else:
                        stack[sp - 2, j] = numerator / denominator
                sp -= 1

            elif op == OP_ABS:
                for j in range(n_samples):
                    stack[sp - 1, j] = abs(stack[sp - 1, j])

            elif op == OP_NEG:
                for j in range(n_samples):
                    stack[sp - 1, j] = -stack[sp - 1, j]

            elif op == OP_MIN:
                for j in range(n_samples):
                    a = stack[sp - 2, j]
                    b = stack[sp - 1, j]
                    stack[sp - 2, j] = a if a < b else b
                sp -= 1

            elif op == OP_MAX:
                for j in range(n_samples):
                    a = stack[sp - 2, j]
                    b = stack[sp - 1, j]
                    stack[sp - 2, j] = a if a > b else b
                sp -= 1

            elif op == OP_SQRT:
                for j in range(n_samples):
                    stack[sp - 1, j] = math.sqrt(abs(stack[sp - 1, j]))

            elif op == OP_LOG:
                for j in range(n_samples):
                    stack[sp - 1, j] = math.log1p(abs(stack[sp - 1, j]))

            elif op == OP_SIN:
                for j in range(n_samples):
                    stack[sp - 1, j] = math.sin(stack[sp - 1, j])

            elif op == OP_COS:
                for j in range(n_samples):
                    stack[sp - 1, j] = math.cos(stack[sp - 1, j])

            elif op == OP_TANH:
                for j in range(n_samples):
                    stack[sp - 1, j] = math.tanh(stack[sp - 1, j])

            elif op == OP_SIGMOID:
                for j in range(n_samples):
                    z = stack[sp - 1, j]
                    if z > 60.0:
                        z = 60.0
                    elif z < -60.0:
                        z = -60.0
                    stack[sp - 1, j] = 1.0 / (1.0 + math.exp(-z))

            elif op == OP_SQUARE:
                for j in range(n_samples):
                    z = stack[sp - 1, j]
                    stack[sp - 1, j] = z * z

        out = np.empty(n_samples, dtype=np.float64)
        for j in range(n_samples):
            z = stack[0, j]
            if math.isnan(z):
                z = 0.0
            elif z > 1e12:
                z = 1e12
            elif z < -1e12:
                z = -1e12
            out[j] = z

        return out
else:
    _evaluate_program_numba = None


def evaluate_program(ops, args, X, backend="auto"):
    X = np.asarray(X, dtype=np.float64)
    X = np.ascontiguousarray(X)

    if backend in ("auto", "numba") and NUMBA_AVAILABLE:
        try:
            return _evaluate_program_numba(ops, args, X)
        except Exception:
            if backend == "numba":
                raise

    return _evaluate_program_numpy(ops, args, X)
