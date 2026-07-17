"""DAG compilation and common-subexpression evaluation.

The evaluator converts a tree into a topologically ordered directed acyclic
 graph. Structurally identical subtrees are represented once, which reduces
operation count and enables persistent subtree caching across individuals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import numpy as np

from .cache import ArrayLRUCache, dataset_token
from .functions import PRIMITIVES


@dataclass(frozen=True)
class DAGNode:
    kind: str
    name: str | None
    feature: int | None
    value: float | None
    children: tuple[int, ...]
    key: str


@dataclass
class CompiledDAG:
    nodes: list[DAGNode]
    root_index: int
    tree_nodes: int

    @property
    def dag_nodes(self) -> int:
        return len(self.nodes)

    @property
    def shared_subexpressions(self) -> int:
        return max(0, int(self.tree_nodes) - int(self.dag_nodes))

    @property
    def compression_ratio(self) -> float:
        return 0.0 if self.tree_nodes <= 0 else 1.0 - self.dag_nodes / self.tree_nodes

    def statistics(self) -> dict:
        return {
            "tree_nodes": int(self.tree_nodes),
            "dag_nodes": int(self.dag_nodes),
            "shared_subexpressions": int(self.shared_subexpressions),
            "compression_ratio": float(self.compression_ratio),
        }


def _terminal_key(kind: str, value: Any) -> str:
    payload = f"{kind}:{value!r}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def compile_dag(root) -> CompiledDAG:
    nodes: list[DAGNode] = []
    memo: dict[tuple, int] = {}

    def visit(node) -> int:
        if node.kind == "feature":
            signature = ("feature", int(node.feature))
            key = _terminal_key("feature", int(node.feature))
            children = ()
        elif node.kind == "constant":
            value = float(node.value)
            signature = ("constant", value)
            key = _terminal_key("constant", format(value, ".17g"))
            children = ()
        else:
            children = tuple(visit(child) for child in node.children)
            child_keys = tuple(nodes[index].key for index in children)
            signature = ("function", str(node.name), child_keys)
            payload = repr(signature).encode("utf-8")
            key = hashlib.blake2b(payload, digest_size=16).hexdigest()
        if signature in memo:
            return memo[signature]
        index = len(nodes)
        nodes.append(DAGNode(
            kind=str(node.kind), name=node.name,
            feature=None if node.feature is None else int(node.feature),
            value=None if node.value is None else float(node.value),
            children=children, key=key,
        ))
        memo[signature] = index
        return index

    root_index = visit(root)
    return CompiledDAG(nodes=nodes, root_index=root_index, tree_nodes=int(root.size))


def _finite(value: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def execute_dag(
    X: np.ndarray,
    dag: CompiledDAG,
    cache: ArrayLRUCache | None = None,
    cache_namespace: tuple | None = None,
) -> np.ndarray:
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    token = dataset_token(X) if cache_namespace is None else cache_namespace
    outputs: list[np.ndarray | None] = [None] * len(dag.nodes)
    n_samples = X.shape[0]

    for index, node in enumerate(dag.nodes):
        cache_key = (token, node.key)
        cached = None if cache is None else cache.get(cache_key)
        if cached is not None:
            outputs[index] = cached
            continue
        if node.kind == "feature":
            value = X[:, int(node.feature)]
        elif node.kind == "constant":
            value = np.full(n_samples, float(node.value), dtype=np.float64)
        else:
            primitive = PRIMITIVES[str(node.name)]
            args = [outputs[child] for child in node.children]
            with np.errstate(all="ignore"):
                value = primitive.function(*args)
            value = _finite(np.asarray(value, dtype=np.float64))
        # Preserve NaN feature terminals so is_missing/coalesce can inspect
        # them. All function outputs are finite by construction.
        outputs[index] = np.asarray(value, dtype=np.float64)
        if cache is not None and node.kind == "function":
            cache.set(cache_key, outputs[index])

    result = np.asarray(outputs[dag.root_index], dtype=np.float64)
    return _finite(result)
