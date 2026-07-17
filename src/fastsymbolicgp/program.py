"""Tree representation and genetic operators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence
import copy
import math
import numpy as np

from .functions import PRIMITIVES
from .backend import FEATURE, CONSTANT, NAME_TO_OPCODE, execute_postfix
from .dag import compile_dag


@dataclass
class Node:
    kind: str  # function | feature | constant
    name: str | None = None
    feature: int | None = None
    value: float | None = None
    children: list["Node"] = field(default_factory=list)

    def clone(self) -> "Node":
        return Node(
            kind=self.kind,
            name=self.name,
            feature=self.feature,
            value=self.value,
            children=[child.clone() for child in self.children],
        )

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children)

    @property
    def depth(self) -> int:
        return 1 if not self.children else 1 + max(c.depth for c in self.children)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "feature":
            return X[:, int(self.feature)]
        if self.kind == "constant":
            return np.full(X.shape[0], float(self.value), dtype=float)
        primitive = PRIMITIVES[str(self.name)]
        args = [child.evaluate(X) for child in self.children]
        with np.errstate(all="ignore"):
            out = primitive.function(*args)
        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def to_string(self, feature_names: Sequence[str] | None = None, precision: int = 6) -> str:
        if self.kind == "feature":
            idx = int(self.feature)
            return str(feature_names[idx]) if feature_names is not None else f"X{idx}"
        if self.kind == "constant":
            value = 0.0 if self.value is None else float(self.value)
            return f"{value:.{precision}g}"
        args = ", ".join(c.to_string(feature_names, precision) for c in self.children)
        return f"{self.name}({args})"

    def iter_paths(self, path: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], "Node"]]:
        yield path, self
        for index, child in enumerate(self.children):
            yield from child.iter_paths(path + (index,))

    def get_path(self, path: tuple[int, ...]) -> "Node":
        node = self
        for index in path:
            node = node.children[index]
        return node

    def replace_path(self, path: tuple[int, ...], replacement: "Node") -> "Node":
        if not path:
            return replacement.clone()
        root = self.clone()
        parent = root
        for index in path[:-1]:
            parent = parent.children[index]
        parent.children[path[-1]] = replacement.clone()
        return root


@dataclass
class SymbolicProgram:
    root: Node
    raw_fitness_: float | None = None
    validation_fitness_: float | None = None
    selection_fitness_: float | None = None
    scale_: float = 1.0
    intercept_: float = 0.0
    metadata_: dict = field(default_factory=dict)

    def clone(self) -> "SymbolicProgram":
        return SymbolicProgram(
            root=self.root.clone(),
            raw_fitness_=self.raw_fitness_,
            validation_fitness_=self.validation_fitness_,
            selection_fitness_=self.selection_fitness_,
            scale_=self.scale_,
            intercept_=self.intercept_,
            metadata_=dict(self.metadata_),
        )

    @property
    def size(self) -> int:
        cached = self.metadata_.get("_size")
        if cached is None:
            cached = self.root.size
            self.metadata_["_size"] = int(cached)
        return int(cached)

    @property
    def depth(self) -> int:
        cached = self.metadata_.get("_depth")
        if cached is None:
            cached = self.root.depth
            self.metadata_["_depth"] = int(cached)
        return int(cached)

    def _compile_postfix(self):
        cached = self.metadata_.get("_postfix")
        if cached is not None:
            return cached
        opcodes = []
        arguments = []
        def visit(node):
            for child in node.children:
                visit(child)
            if node.kind == "feature":
                opcodes.append(FEATURE); arguments.append(float(node.feature))
            elif node.kind == "constant":
                opcodes.append(CONSTANT); arguments.append(float(node.value))
            else:
                opcodes.append(NAME_TO_OPCODE[str(node.name)]); arguments.append(0.0)
        visit(self.root)
        compiled = (np.asarray(opcodes, dtype=np.int16), np.asarray(arguments, dtype=np.float64))
        self.metadata_["_postfix"] = compiled
        return compiled

    def compile_dag(self):
        cached = self.metadata_.get("_dag")
        if cached is None:
            cached = compile_dag(self.root)
            self.metadata_["_dag"] = cached
        return cached

    @property
    def dag_size(self) -> int:
        return int(self.compile_dag().dag_nodes)

    @property
    def dag_compression_ratio(self) -> float:
        return float(self.compile_dag().compression_ratio)

    def dag_stats(self) -> dict:
        return self.compile_dag().statistics()

    def feature_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, node in self.root.iter_paths():
            if node.kind == "feature":
                index = int(node.feature)
                counts[index] = counts.get(index, 0) + 1
        return counts

    def function_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, node in self.root.iter_paths():
            if node.kind == "function":
                name = str(node.name)
                counts[name] = counts.get(name, 0) + 1
        return counts

    def execute(self, X: np.ndarray, backend: str = "numpy") -> np.ndarray:
        if str(backend).lower() == "tree":
            return self.root.evaluate(X)
        opcodes, arguments = self._compile_postfix()
        return execute_postfix(X, opcodes, arguments, backend=backend)

    def __str__(self) -> str:
        return self.root.to_string()

    def to_string(self, feature_names: Sequence[str] | None = None, precision: int = 6) -> str:
        return self.root.to_string(feature_names, precision)

    def structural_hash(self) -> str:
        cached = self.metadata_.get("_structural_hash")
        if cached is None:
            cached = self.to_string(precision=12)
            self.metadata_["_structural_hash"] = cached
        return cached


def random_terminal(rng: np.random.Generator, n_features: int, const_range: tuple[float, float], const_probability: float) -> Node:
    if rng.random() < const_probability:
        return Node(kind="constant", value=float(rng.uniform(*const_range)))
    return Node(kind="feature", feature=int(rng.integers(0, n_features)))


def random_tree(
    rng: np.random.Generator,
    n_features: int,
    function_set: Sequence[str],
    min_depth: int,
    max_depth: int,
    const_range: tuple[float, float],
    const_probability: float,
    grow: bool,
    current_depth: int = 1,
    function_weights: dict[str, float] | None = None,
) -> Node:
    force_function = current_depth < min_depth
    force_terminal = current_depth >= max_depth
    choose_terminal = force_terminal or (grow and not force_function and rng.random() < 0.35)
    if choose_terminal:
        return random_terminal(rng, n_features, const_range, const_probability)
    if function_weights:
        probabilities = np.asarray([max(0.0, float(function_weights.get(name, 1.0))) for name in function_set], dtype=float)
        probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else None
        name = str(rng.choice(function_set, p=probabilities))
    else:
        name = str(rng.choice(function_set))
    primitive = PRIMITIVES[name]
    children = [
        random_tree(
            rng, n_features, function_set, min_depth, max_depth,
            const_range, const_probability, grow, current_depth + 1, function_weights,
        )
        for _ in range(primitive.arity)
    ]
    return Node(kind="function", name=name, children=children)


def subtree_crossover(a: SymbolicProgram, b: SymbolicProgram, rng: np.random.Generator) -> SymbolicProgram:
    a_paths = [p for p, _ in a.root.iter_paths()]
    b_nodes = [n for _, n in b.root.iter_paths()]
    path = a_paths[int(rng.integers(0, len(a_paths)))]
    donor = b_nodes[int(rng.integers(0, len(b_nodes)))]
    return SymbolicProgram(a.root.replace_path(path, donor))


def subtree_mutation(
    program: SymbolicProgram,
    rng: np.random.Generator,
    n_features: int,
    function_set: Sequence[str],
    mutation_depth: tuple[int, int],
    const_range: tuple[float, float],
    const_probability: float,
    function_weights: dict[str, float] | None = None,
) -> SymbolicProgram:
    paths = [p for p, _ in program.root.iter_paths()]
    path = paths[int(rng.integers(0, len(paths)))]
    min_d, max_d = mutation_depth
    replacement = random_tree(
        rng, n_features, function_set, min_d, max_d,
        const_range, const_probability, grow=True, function_weights=function_weights,
    )
    return SymbolicProgram(program.root.replace_path(path, replacement))


def hoist_mutation(program: SymbolicProgram, rng: np.random.Generator) -> SymbolicProgram:
    all_nodes = [(p, n) for p, n in program.root.iter_paths()]
    path, subtree = all_nodes[int(rng.integers(0, len(all_nodes)))]
    descendants = [n for _, n in subtree.iter_paths()]
    replacement = descendants[int(rng.integers(0, len(descendants)))]
    return SymbolicProgram(program.root.replace_path(path, replacement))


def point_mutation(
    program: SymbolicProgram,
    rng: np.random.Generator,
    n_features: int,
    function_set: Sequence[str],
    const_range: tuple[float, float],
    function_weights: dict[str, float] | None = None,
) -> SymbolicProgram:
    root = program.root.clone()
    paths_nodes = list(root.iter_paths())
    path, node = paths_nodes[int(rng.integers(0, len(paths_nodes)))]
    target = root.get_path(path)
    if target.kind == "feature":
        target.feature = int(rng.integers(0, n_features))
    elif target.kind == "constant":
        span = const_range[1] - const_range[0]
        target.value = float(np.clip(float(target.value) + rng.normal(0, 0.1 * span), *const_range))
    else:
        same_arity = [name for name in function_set if PRIMITIVES[name].arity == len(target.children)]
        if same_arity:
            if function_weights:
                probabilities = np.asarray([max(0.0, float(function_weights.get(name, 1.0))) for name in same_arity], dtype=float)
                probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else None
                target.name = str(rng.choice(same_arity, p=probabilities))
            else:
                target.name = str(rng.choice(same_arity))
    return SymbolicProgram(root)


def enforce_limits(program: SymbolicProgram, max_depth: int, max_nodes: int) -> bool:
    return program.depth <= max_depth and program.size <= max_nodes


def semantic_hash(values: np.ndarray, decimals: int = 8) -> int:
    clipped = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    return hash(np.round(clipped, decimals=decimals).tobytes())
