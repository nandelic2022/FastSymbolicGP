"""Expression-quality utilities for FastSymbolicGP V1.2.4.

The module deliberately separates globally safe algebraic rewrites from
empirically certified reductions.  Protected operators (especially division)
do not obey every textbook identity globally, so aggressive reductions are
accepted only when the fitted program is numerically equivalent on a declared
reference set.
"""
from __future__ import annotations

import math
import numpy as np

from .program import Node, Program


def _is_const(node: Node, value=None, tol=1e-15) -> bool:
    if node.kind != "const":
        return False
    if value is None:
        return True
    return abs(float(node.value) - float(value)) <= tol


def _same(a: Node, b: Node) -> bool:
    # Stable textual equality is intentional here: this runs only in the final
    # simplification phase, not in the hot evolutionary loop.
    return a.to_string() == b.to_string()


def _const(value: float, unit=None) -> Node:
    return Node("const", float(value), (), unit)


def safe_simplify_node(node: Node) -> Node:
    """Apply identities that preserve FastSymbolicGP protected semantics."""
    if node.kind != "func":
        return node.clone()

    children = tuple(safe_simplify_node(c) for c in node.children)
    name = str(node.value)

    # Constant folding uses the actual protected primitive semantics through
    # Node.evaluate, not symbolic textbook arithmetic.
    if children and all(c.kind == "const" for c in children):
        try:
            value = Node("func", name, children, node.unit).evaluate(np.zeros((1, 1), float))[0]
            if np.isfinite(value):
                return _const(float(value), node.unit)
        except Exception:
            pass

    if name == "add" and len(children) == 2:
        a, b = children
        if _is_const(a, 0.0):
            return b
        if _is_const(b, 0.0):
            return a

    if name == "sub" and len(children) == 2:
        a, b = children
        if _is_const(b, 0.0):
            return a
        if _same(a, b):
            return _const(0.0, node.unit)
        if _is_const(a, 0.0):
            return Node("func", "neg", (b,), node.unit)

    if name == "mul" and len(children) == 2:
        a, b = children
        if _is_const(a, 0.0) or _is_const(b, 0.0):
            return _const(0.0, node.unit)
        if _is_const(a, 1.0):
            return b
        if _is_const(b, 1.0):
            return a
        if _is_const(a, -1.0):
            return Node("func", "neg", (b,), node.unit)
        if _is_const(b, -1.0):
            return Node("func", "neg", (a,), node.unit)

    if name == "div" and len(children) == 2:
        a, b = children
        # FastSymbolicGP protected division returns exactly 1 whenever the
        # denominator is zero, regardless of the numerator.
        if _is_const(b, 0.0):
            return _const(1.0, node.unit)
        if _is_const(b, 1.0):
            return a
        if _is_const(b, -1.0):
            return Node("func", "neg", (a,), node.unit)
        # With FastSymbolicGP protected division x/x == 1 even at x == 0,
        # because division-by-zero returns 1.
        if _same(a, b):
            return _const(1.0, node.unit)

    if name == "neg" and len(children) == 1:
        c = children[0]
        if c.kind == "func" and str(c.value) == "neg":
            return c.children[0]
        if _is_const(c):
            return _const(-float(c.value), c.unit)

    if name == "abs" and len(children) == 1:
        c = children[0]
        if c.kind == "func" and str(c.value) == "abs":
            return c

    if name in {"min", "max"} and len(children) == 2 and _same(children[0], children[1]):
        return children[0]

    return Node("func", name, children, node.unit)


def safe_simplify_program(program: Program, max_passes: int = 8) -> Program:
    q = program.clone()
    previous = None
    for _ in range(max(1, int(max_passes))):
        q.root = safe_simplify_node(q.root)
        text = q.root.to_string()
        if text == previous:
            break
        previous = text
    q.invalidate_caches()
    return q


def _reference_subset(X, max_points: int):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if len(X) <= int(max_points):
        return X
    # Deterministic evenly spaced coverage avoids adding RNG state to a fitted
    # model merely because final simplification is enabled.
    idx = np.linspace(0, len(X) - 1, int(max_points), dtype=int)
    return np.ascontiguousarray(X[idx])


def _equivalence(original: Program, candidate: Program, X, *, atol: float, rtol: float):
    a = np.asarray(original.predict(X), dtype=float)
    b = np.asarray(candidate.predict(X), dtype=float)
    delta = np.abs(a - b)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    scale = max(1.0, float(np.max(np.abs(a))) if a.size else 1.0)
    threshold = float(atol) + float(rtol) * scale
    return bool(np.all(np.isfinite(b)) and max_abs <= threshold), max_abs, threshold


def _replacement_candidates(node: Node):
    """Yield smaller local replacements; all non-global identities are certified."""
    seen = set()

    def emit(n):
        key = n.to_string()
        if key not in seen and n.size < node.size:
            seen.add(key)
            return n
        return None

    # Children are excellent candidates for removing neutral/noisy wrappers.
    for child in node.children:
        cand = emit(child)
        if cand is not None:
            yield cand

    # Simple constants frequently eliminate protected-division artefacts such
    # as 0/x when the denominator never hits its protected singularity on the
    # certification set.  They are never accepted without numerical proof.
    for value in (0.0, 1.0, -1.0):
        cand = emit(_const(value, node.unit))
        if cand is not None:
            yield cand

    if node.kind == "func" and len(node.children) == 2:
        a, b = node.children
        name = str(node.value)
        if name == "add" and _same(a, b):
            cand = emit(Node("func", "mul", (_const(2.0), a), node.unit))
            if cand is not None:
                yield cand
        if name == "sub" and _same(a, b):
            cand = emit(_const(0.0, node.unit))
            if cand is not None:
                yield cand
        if name == "div" and _same(a, b):
            cand = emit(_const(1.0, node.unit))
            if cand is not None:
                yield cand


def certified_simplify_program(
    program: Program,
    X,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    max_reference_points: int = 2048,
    max_passes: int = 4,
    max_trials: int = 512,
    admissible=None,
):
    """Greedily prune a fitted program while preserving reference predictions.

    This is an *empirical* equivalence certificate.  It does not claim global
    algebraic equivalence outside the supplied reference set.
    """
    ref = _reference_subset(X, max_reference_points)
    original = program.clone()
    q = safe_simplify_program(program)
    ok, initial_delta, threshold = _equivalence(original, q, ref, atol=atol, rtol=rtol)
    if not ok or (admissible is not None and not admissible(q)):
        q = original.clone()
        initial_delta = 0.0

    accepted = []
    trials = 0
    for _ in range(max(1, int(max_passes))):
        improved = False
        # Largest subexpressions first maximises node reduction per expensive
        # numerical equivalence check.
        paths = list(q.paths())
        paths.sort(key=lambda p: q.root.get(p).size, reverse=True)
        for path in paths:
            node = q.root.get(path)
            if node.size <= 1:
                continue
            for replacement in _replacement_candidates(node):
                trials += 1
                if trials > int(max_trials):
                    break
                trial = q.evolution_clone()
                trial.root = q.root.replace_shared(path, replacement)
                trial.invalidate_caches()
                trial = safe_simplify_program(trial)
                if trial.size >= q.size:
                    continue
                if admissible is not None and not admissible(trial):
                    continue
                equivalent, max_abs, _ = _equivalence(q, trial, ref, atol=atol, rtol=rtol)
                if equivalent:
                    accepted.append({
                        "path": tuple(path),
                        "from_nodes": int(q.size),
                        "to_nodes": int(trial.size),
                        "max_abs_prediction_delta": float(max_abs),
                    })
                    q = trial
                    improved = True
                    break
            if trials > int(max_trials) or improved:
                break
        if not improved or trials > int(max_trials):
            break

    # Final global certificate against the original fitted program.
    equivalent, max_abs, threshold = _equivalence(original, q, ref, atol=atol, rtol=rtol)
    if not equivalent:
        q = original.clone()
        accepted = []
        max_abs = 0.0

    report = {
        "mode": "certified",
        "scope": "empirical_reference_set",
        "reference_points": int(len(ref)),
        "atol": float(atol),
        "rtol": float(rtol),
        "equivalence_threshold": float(threshold),
        "passed": bool(equivalent),
        "original_nodes": int(original.size),
        "final_nodes": int(q.size),
        "nodes_removed": int(original.size - q.size),
        "reduction_fraction": float((original.size - q.size) / max(1, original.size)),
        "trials": int(trials),
        "accepted_rewrites": int(len(accepted)),
        "max_abs_prediction_delta": float(max_abs),
        "rewrites": accepted,
        "original_expression": original.to_string(),
        "final_expression": q.to_string(),
    }
    q.metadata["simplification_report"] = report
    return q, report
