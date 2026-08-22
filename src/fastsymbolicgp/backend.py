"""Compiled population execution and fused fitness backends for V0.9.0.

V0.9.0 retains packed postfix execution and adds Numba kernels that combine
program evaluation, scaling and fitness reduction without constructing a
population-by-sample prediction matrix.
"""
from __future__ import annotations

import math
import os
from typing import Sequence

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
    from numba import config as _numba_config
    from numba import get_num_threads, njit, prange, set_num_threads

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
                elif 10 <= op <= 16:
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

    @njit(cache=True, parallel=True)
    def _execute_population_postfix_numba(X, opcodes, arguments, offsets):
        """Execute a packed population in parallel over symbolic programs."""
        n_programs = offsets.shape[0] - 1
        n_samples = X.shape[0]
        out = np.empty((n_programs, n_samples), dtype=np.float64)

        for program_index in prange(n_programs):
            start = offsets[program_index]
            end = offsets[program_index + 1]
            n_ops = end - start
            stack = np.empty(n_ops, dtype=np.float64)

            for i in range(n_samples):
                sp = 0
                for j in range(start, end):
                    op = opcodes[j]
                    arg = arguments[j]
                    if op == FEATURE:
                        value = X[i, int(arg)]
                        stack[sp] = value
                        sp += 1
                    elif op == CONSTANT:
                        stack[sp] = arg
                        sp += 1
                    elif 10 <= op <= 16:
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
                out[program_index, i] = stack[0]
        return out


    @njit(cache=True, inline="always")
    def _evaluate_packed_sample(X, sample_index, opcodes, arguments, start, end, stack):
        """Evaluate one packed postfix program for one row using reusable stack memory."""
        sp = 0
        for j in range(start, end):
            op = opcodes[j]
            arg = arguments[j]
            if op == FEATURE:
                value = X[sample_index, int(arg)]
                stack[sp] = value
                sp += 1
            elif op == CONSTANT:
                stack[sp] = arg
                sp += 1
            elif 10 <= op <= 16:
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
        return stack[0]

    @njit(cache=True, inline="always")
    def _regression_loss_from_sums(sse, weight_sum, y_sum, y_square_sum, metric_code):
        if weight_sum <= 0.0:
            return 1e12
        if sse < 0.0:
            sse = 0.0
        if metric_code == 0:  # mse
            return sse / weight_sum
        if metric_code == 1:  # rmse
            return math.sqrt(max(0.0, sse / weight_sum))
        # 1 - R2 = SSE / SST, matching sklearn force_finite behaviour.
        sst = y_square_sum - (y_sum * y_sum) / weight_sum
        if sst <= 1e-18:
            return 0.0 if sse <= 1e-18 else 1.0
        return sse / sst

    @njit(cache=True, parallel=True)
    def _score_population_regression_numba(
        X_train, y_train, w_train, X_val, y_val, w_val,
        opcodes, arguments, offsets, scaling_code, metric_code, val_metric_code,
    ):
        """Fused expression execution, affine scaling and regression fitness."""
        n_programs = offsets.shape[0] - 1
        scales = np.empty(n_programs, dtype=np.float64)
        intercepts = np.empty(n_programs, dtype=np.float64)
        train_losses = np.empty(n_programs, dtype=np.float64)
        val_losses = np.empty(n_programs, dtype=np.float64)
        has_val = X_val.shape[0] > 0

        for program_index in prange(n_programs):
            start = offsets[program_index]
            end = offsets[program_index + 1]
            stack = np.empty(end - start, dtype=np.float64)
            sw = sx = sy = sxx = sxy = syy = 0.0
            for i in range(X_train.shape[0]):
                raw = _evaluate_packed_sample(X_train, i, opcodes, arguments, start, end, stack)
                weight = w_train[i]
                target = y_train[i]
                sw += weight
                sx += weight * raw
                sy += weight * target
                sxx += weight * raw * raw
                sxy += weight * raw * target
                syy += weight * target * target

            if sw <= 0.0:
                scale = 0.0
                intercept = 0.0
            elif scaling_code == 1:
                variance_numerator = sxx - sx * sx / sw
                covariance_numerator = sxy - sx * sy / sw
                if variance_numerator <= 1e-16 * sw:
                    scale = 0.0
                    intercept = sy / sw
                else:
                    scale = covariance_numerator / variance_numerator
                    intercept = sy / sw - scale * sx / sw
            else:
                scale = 1.0
                intercept = 0.0

            sse = (
                syy + scale * scale * sxx + intercept * intercept * sw
                + 2.0 * scale * intercept * sx - 2.0 * scale * sxy - 2.0 * intercept * sy
            )
            scales[program_index] = scale
            intercepts[program_index] = intercept
            train_losses[program_index] = _regression_loss_from_sums(
                sse, sw, sy, syy, metric_code,
            )

            if has_val:
                val_sw = val_sy = val_syy = val_sse = 0.0
                for i in range(X_val.shape[0]):
                    raw = _evaluate_packed_sample(X_val, i, opcodes, arguments, start, end, stack)
                    target = y_val[i]
                    weight = w_val[i]
                    residual = scale * raw + intercept - target
                    val_sw += weight
                    val_sy += weight * target
                    val_syy += weight * target * target
                    val_sse += weight * residual * residual
                val_losses[program_index] = _regression_loss_from_sums(
                    val_sse, val_sw, val_sy, val_syy, val_metric_code,
                )
            else:
                val_losses[program_index] = train_losses[program_index]
        return scales, intercepts, train_losses, val_losses

    @njit(cache=True, inline="always")
    def _classification_loss_from_counts(tp, tn, fp, fn, total_loss, total_weight, metric_code):
        if total_weight <= 0.0:
            return 1e12
        if metric_code == 0:  # log loss
            return total_loss / total_weight
        if metric_code == 1:  # accuracy
            return 1.0 - (tp + tn) / total_weight
        if metric_code == 2:  # balanced accuracy
            pos = tp + fn
            neg = tn + fp
            tpr = tp / pos if pos > 0.0 else 0.0
            tnr = tn / neg if neg > 0.0 else 0.0
            if pos > 0.0 and neg > 0.0:
                score = 0.5 * (tpr + tnr)
            elif pos > 0.0:
                score = tpr
            else:
                score = tnr
            return 1.0 - score
        if metric_code == 3:  # F1
            den = 2.0 * tp + fp + fn
            score = 2.0 * tp / den if den > 0.0 else 0.0
            return 1.0 - score
        # MCC transformed to minimisation loss.
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = (tp * tn - fp * fn) / math.sqrt(den) if den > 0.0 else 0.0
        return 1.0 - (mcc + 1.0) / 2.0

    @njit(cache=True, inline="always")
    def _accumulate_classification(raw, target, weight, scale, intercept, metric_code):
        z = scale * raw + intercept
        predicted = 1 if z >= 0.0 else 0
        tp = weight if predicted == 1 and target >= 0.5 else 0.0
        tn = weight if predicted == 0 and target < 0.5 else 0.0
        fp = weight if predicted == 1 and target < 0.5 else 0.0
        fn = weight if predicted == 0 and target >= 0.5 else 0.0
        loss = 0.0
        if metric_code == 0:
            # Stable binary logistic loss without materialising probabilities.
            loss = weight * (max(z, 0.0) - target * z + math.log1p(math.exp(-abs(z))))
        return tp, tn, fp, fn, loss

    @njit(cache=True, parallel=True)
    def _score_population_classification_numba(
        X_train, y_train, w_train, X_val, y_val, w_val,
        opcodes, arguments, offsets, scaling_code, regularization,
        metric_code, val_metric_code,
    ):
        """Fused population evaluator for fixed or approximate logistic fitness."""
        n_programs = offsets.shape[0] - 1
        scales = np.empty(n_programs, dtype=np.float64)
        intercepts = np.empty(n_programs, dtype=np.float64)
        train_losses = np.empty(n_programs, dtype=np.float64)
        val_losses = np.empty(n_programs, dtype=np.float64)
        has_val = X_val.shape[0] > 0

        for program_index in prange(n_programs):
            start = offsets[program_index]
            end = offsets[program_index + 1]
            stack = np.empty(end - start, dtype=np.float64)
            raw_values = np.empty(X_train.shape[0], dtype=np.float64)
            sw = sx = sy = sxx = sxy = 0.0
            for i in range(X_train.shape[0]):
                raw = _evaluate_packed_sample(X_train, i, opcodes, arguments, start, end, stack)
                raw_values[i] = raw
                weight = w_train[i]
                target = y_train[i]
                sw += weight
                sx += weight * raw
                sy += weight * target
                sxx += weight * raw * raw
                sxy += weight * raw * target

            if scaling_code == 1 and sw > 0.0:
                prior = min(1.0 - 1e-8, max(1e-8, sy / sw))
                x_mean = sx / sw
                variance = max(0.0, sxx - sx * sx / sw)
                covariance = sxy - x_mean * sy
                curvature = max(prior * (1.0 - prior), 1e-8) * variance + regularization
                scale = covariance / curvature if variance > 1e-18 else 0.0
                scale = min(1e6, max(-1e6, scale))
                intercept = math.log(prior / (1.0 - prior)) - scale * x_mean
                intercept = min(1e6, max(-1e6, intercept))
            else:
                scale = 1.0
                intercept = 0.0

            tp = tn = fp = fn = total_loss = total_weight = 0.0
            for i in range(X_train.shape[0]):
                weight = w_train[i]
                a, b, c, d, loss = _accumulate_classification(
                    raw_values[i], y_train[i], weight, scale, intercept, metric_code,
                )
                tp += a; tn += b; fp += c; fn += d; total_loss += loss; total_weight += weight
            scales[program_index] = scale
            intercepts[program_index] = intercept
            train_losses[program_index] = _classification_loss_from_counts(
                tp, tn, fp, fn, total_loss, total_weight, metric_code,
            )

            if has_val:
                tp = tn = fp = fn = total_loss = total_weight = 0.0
                for i in range(X_val.shape[0]):
                    raw = _evaluate_packed_sample(X_val, i, opcodes, arguments, start, end, stack)
                    weight = w_val[i]
                    a, b, c, d, loss = _accumulate_classification(
                        raw, y_val[i], weight, scale, intercept, val_metric_code,
                    )
                    tp += a; tn += b; fp += c; fn += d; total_loss += loss; total_weight += weight
                val_losses[program_index] = _classification_loss_from_counts(
                    tp, tn, fp, fn, total_loss, total_weight, val_metric_code,
                )
            else:
                val_losses[program_index] = train_losses[program_index]
        return scales, intercepts, train_losses, val_losses

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without numba
    NUMBA_AVAILABLE = False
    _execute_postfix_numba = None
    _execute_population_postfix_numba = None
    _score_population_regression_numba = None
    _score_population_classification_numba = None
    get_num_threads = None
    set_num_threads = None
    _numba_config = None


def configure_numba_threads(n_jobs: int | None) -> int:
    """Set the Numba worker count and return the effective number of threads."""
    if not NUMBA_AVAILABLE:
        return 1
    requested = 1 if n_jobs is None else int(n_jobs)
    maximum = int(getattr(_numba_config, "NUMBA_NUM_THREADS", os.cpu_count() or 1))
    if requested < 0:
        requested = maximum
    requested = max(1, min(requested, maximum))
    set_num_threads(requested)
    return int(get_num_threads())


def pack_programs(programs: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack postfix programs into contiguous opcode, argument and offset arrays."""
    opcode_parts: list[np.ndarray] = []
    argument_parts: list[np.ndarray] = []
    offsets = np.empty(len(programs) + 1, dtype=np.int64)
    cursor = 0
    offsets[0] = 0
    for index, program in enumerate(programs):
        opcodes, arguments = program._compile_postfix()
        opcode_parts.append(np.asarray(opcodes, dtype=np.int16))
        argument_parts.append(np.asarray(arguments, dtype=np.float64))
        cursor += len(opcodes)
        offsets[index + 1] = cursor
    if opcode_parts:
        packed_opcodes = np.ascontiguousarray(np.concatenate(opcode_parts), dtype=np.int16)
        packed_arguments = np.ascontiguousarray(np.concatenate(argument_parts), dtype=np.float64)
    else:
        packed_opcodes = np.empty(0, dtype=np.int16)
        packed_arguments = np.empty(0, dtype=np.float64)
    return packed_opcodes, packed_arguments, offsets


def execute_population_postfix(
    X: np.ndarray,
    programs: Sequence,
    n_jobs: int = -1,
) -> np.ndarray:
    """Execute several programs in one call.

    The returned matrix has shape ``(n_programs, n_samples)``.  Numba is used
    when available; otherwise the function falls back to a deterministic
    NumPy loop.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if len(programs) == 0:
        return np.empty((0, X.shape[0]), dtype=np.float64)
    if NUMBA_AVAILABLE:
        configure_numba_threads(n_jobs)
        opcodes, arguments, offsets = pack_programs(programs)
        return _execute_population_postfix_numba(X, opcodes, arguments, offsets)
    return np.vstack([
        program.execute(X, backend="numpy") for program in programs
    ])



def _weights_or_ones(values, n_samples: int) -> np.ndarray:
    if values is None:
        return np.ones(int(n_samples), dtype=np.float64)
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _regression_metric_code(metric: str) -> int:
    name = str(metric).lower()
    if name in {"mse", "mean_squared_error"}:
        return 0
    if name in {"rmse", "root_mean_squared_error"}:
        return 1
    if name in {"r2", "r2_score"}:
        return 2
    raise ValueError(f"Fused regression backend does not support metric {metric!r}")


def _classification_metric_code(metric: str) -> int:
    name = str(metric).lower()
    if name in {"log_loss", "logloss"}:
        return 0
    if name == "accuracy":
        return 1
    if name in {"balanced_accuracy", "bacc"}:
        return 2
    if name in {"f1", "f1_score"}:
        return 3
    if name in {"mcc", "matthews"}:
        return 4
    raise ValueError(f"Fused classification backend does not support metric {metric!r}")


def score_population_regression_postfix(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    programs: Sequence,
    *,
    metric: str = "rmse",
    validation_metric: str | None = None,
    scaling: str = "linear",
    sample_weight_train: np.ndarray | None = None,
    sample_weight_val: np.ndarray | None = None,
    n_jobs: int = -1,
):
    """Score a regression population without a population-by-sample matrix."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("The fused regression backend requires Numba")
    X_train = np.ascontiguousarray(np.asarray(X_train, dtype=np.float64))
    y_train = np.ascontiguousarray(np.asarray(y_train, dtype=np.float64))
    if X_val is None:
        X_val = np.empty((0, X_train.shape[1]), dtype=np.float64)
        y_val = np.empty(0, dtype=np.float64)
    else:
        X_val = np.ascontiguousarray(np.asarray(X_val, dtype=np.float64))
        y_val = np.ascontiguousarray(np.asarray(y_val, dtype=np.float64))
    w_train = _weights_or_ones(sample_weight_train, X_train.shape[0])
    w_val = _weights_or_ones(sample_weight_val, X_val.shape[0])
    mode = str(scaling).lower() if scaling is not None else "none"
    if mode in {"linear", "auto"}:
        scaling_code = 1
    elif mode in {"none", "identity"}:
        scaling_code = 0
    else:
        raise ValueError("Fused regression scaling must be linear or none")
    configure_numba_threads(n_jobs)
    opcodes, arguments, offsets = pack_programs(programs)
    return _score_population_regression_numba(
        X_train, y_train, w_train, X_val, y_val, w_val,
        opcodes, arguments, offsets, scaling_code,
        _regression_metric_code(metric),
        _regression_metric_code(metric if validation_metric is None else validation_metric),
    )


def score_population_classification_postfix(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    programs: Sequence,
    *,
    metric: str = "log_loss",
    validation_metric: str | None = None,
    scaling: str = "approx_logistic",
    regularization: float = 1e-4,
    sample_weight_train: np.ndarray | None = None,
    sample_weight_val: np.ndarray | None = None,
    n_jobs: int = -1,
):
    """Fused binary-classification scoring with thread-local raw workspaces."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("The fused classification backend requires Numba")
    X_train = np.ascontiguousarray(np.asarray(X_train, dtype=np.float64))
    y_train = np.ascontiguousarray(np.asarray(y_train, dtype=np.float64))
    if X_val is None:
        X_val = np.empty((0, X_train.shape[1]), dtype=np.float64)
        y_val = np.empty(0, dtype=np.float64)
    else:
        X_val = np.ascontiguousarray(np.asarray(X_val, dtype=np.float64))
        y_val = np.ascontiguousarray(np.asarray(y_val, dtype=np.float64))
    w_train = _weights_or_ones(sample_weight_train, X_train.shape[0])
    w_val = _weights_or_ones(sample_weight_val, X_val.shape[0])
    mode = str(scaling).lower() if scaling is not None else "none"
    if mode in {"approx_logistic", "elite_logistic", "auto"}:
        scaling_code = 1
    elif mode in {"none", "identity", "fixed_sigmoid", "fixed"}:
        scaling_code = 0
    else:
        raise ValueError("Fused classification scaling supports approximate logistic or fixed sigmoid")
    configure_numba_threads(n_jobs)
    opcodes, arguments, offsets = pack_programs(programs)
    return _score_population_classification_numba(
        X_train, y_train, w_train, X_val, y_val, w_val,
        opcodes, arguments, offsets, scaling_code, float(regularization),
        _classification_metric_code(metric),
        _classification_metric_code(metric if validation_metric is None else validation_metric),
    )

def execute_postfix_numpy(X: np.ndarray, opcodes: np.ndarray, arguments: np.ndarray) -> np.ndarray:
    """Vectorised NumPy postfix interpreter."""
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
    if backend in {"population_numba", "population", "batched_numba"}:
        backend = "numba"
    if backend == "auto":
        backend = "numba" if NUMBA_AVAILABLE else "numpy"
    if backend == "numba":
        if not NUMBA_AVAILABLE:
            return execute_postfix_numpy(X, opcodes, arguments)
        return _execute_postfix_numba(X, opcodes, arguments)
    return execute_postfix_numpy(X, opcodes, arguments)
