"""Safe algebraic simplification for evolved trees."""
from __future__ import annotations

import math
from .program import Node, SymbolicProgram

_EPS = 1e-12


def _is_const(node: Node, value: float | None = None) -> bool:
    if node.kind != "constant":
        return False
    return value is None or abs(float(node.value) - value) <= _EPS


def _same(a: Node, b: Node) -> bool:
    return a.to_string(precision=12) == b.to_string(precision=12)


def _constant_fold(name: str, children: list[Node]) -> Node | None:
    if not all(c.kind == "constant" for c in children):
        return None
    values = [float(c.value) for c in children]
    try:
        if name == "add": value = values[0] + values[1]
        elif name == "sub": value = values[0] - values[1]
        elif name == "mul": value = values[0] * values[1]
        elif name == "div": value = 1.0 if abs(values[1]) <= _EPS else values[0] / values[1]
        elif name == "max": value = max(values)
        elif name == "min": value = min(values)
        elif name == "abs": value = abs(values[0])
        elif name == "neg": value = -values[0]
        elif name == "sin": value = math.sin(values[0])
        elif name == "cos": value = math.cos(values[0])
        elif name == "tanh": value = math.tanh(values[0])
        elif name == "sqrt": value = math.sqrt(abs(values[0]))
        elif name == "log": value = math.log(abs(values[0]) + 1e-12)
        elif name == "exp": value = math.exp(max(-30.0, min(30.0, values[0])))
        elif name == "inv": value = 0.0 if abs(values[0]) <= _EPS else 1.0 / values[0]
        else: return None
        if math.isfinite(value):
            return Node(kind="constant", value=float(value))
    except (ValueError, OverflowError, ZeroDivisionError):
        pass
    return None


def simplify_node(node: Node) -> Node:
    if node.kind != "function":
        return node.clone()
    children = [simplify_node(c) for c in node.children]
    name = str(node.name)
    folded = _constant_fold(name, children)
    if folded is not None:
        return folded

    if name == "add":
        if _is_const(children[0], 0.0): return children[1]
        if _is_const(children[1], 0.0): return children[0]
    elif name == "sub":
        if _is_const(children[1], 0.0): return children[0]
        if _same(children[0], children[1]): return Node(kind="constant", value=0.0)
    elif name == "mul":
        if _is_const(children[0], 0.0) or _is_const(children[1], 0.0):
            return Node(kind="constant", value=0.0)
        if _is_const(children[0], 1.0): return children[1]
        if _is_const(children[1], 1.0): return children[0]
    elif name == "div":
        if _is_const(children[0], 0.0): return Node(kind="constant", value=0.0)
        if _is_const(children[1], 1.0): return children[0]
    elif name == "neg" and children[0].kind == "function" and children[0].name == "neg":
        return children[0].children[0]
    elif name == "abs" and children[0].kind == "function" and children[0].name == "abs":
        return children[0]
    return Node(kind="function", name=name, children=children)


def simplify_program(program: SymbolicProgram) -> SymbolicProgram:
    result = program.clone()
    previous = ""
    for _ in range(10):
        current = result.root.to_string(precision=12)
        if current == previous:
            break
        previous = current
        result.root = simplify_node(result.root)
        result.metadata_.pop("_structural_hash", None)
        result.metadata_.pop("_size", None)
        result.metadata_.pop("_depth", None)
        result.metadata_.pop("_postfix", None)
    return result
