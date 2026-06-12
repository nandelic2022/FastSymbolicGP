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
    resolve_function_set,
    op_arity,
    is_unary,
    is_binary,
)


class FastProgram:
    """
    Compact postfix symbolic program.
    """

    def __init__(self, ops, args):
        self.ops = np.asarray(ops, dtype=np.int64)
        self.args = np.asarray(args, dtype=np.float64)
        self.fitness_ = None

        if len(self.ops) != len(self.args):
            raise ValueError("ops and args must have the same length.")

    @property
    def size(self):
        return int(len(self.ops))

    def copy(self):
        new = FastProgram(self.ops.copy(), self.args.copy())
        new.fitness_ = self.fitness_
        return new

    def key(self, decimals=8):
        rounded_args = tuple(np.round(self.args, decimals=decimals))
        return tuple(self.ops.tolist()), rounded_args

    def is_valid(self):
        balance = 0
        for op in self.ops:
            arity = op_arity(op)
            if arity == 0:
                balance += 1
            else:
                if balance < arity:
                    return False
                balance = balance - arity + 1
        return balance == 1

    def depth(self):
        stack = []
        for op in self.ops:
            arity = op_arity(op)
            if arity == 0:
                stack.append(1)
            elif arity == 1:
                if not stack:
                    return 0
                a = stack.pop()
                stack.append(a + 1)
            elif arity == 2:
                if len(stack) < 2:
                    return 0
                b = stack.pop()
                a = stack.pop()
                stack.append(max(a, b) + 1)
        return stack[0] if len(stack) == 1 else 0

    def to_string(self, feature_names=None):
        stack = []

        for op, arg in zip(self.ops, self.args):
            op = int(op)

            if op == OP_VAR:
                idx = int(arg)
                if feature_names is None:
                    stack.append(f"x{idx}")
                else:
                    stack.append(str(feature_names[idx]))

            elif op == OP_CONST:
                stack.append(f"{arg:.6g}")

            elif op == OP_ADD:
                b = stack.pop()
                a = stack.pop()
                stack.append(f"({a} + {b})")

            elif op == OP_SUB:
                b = stack.pop()
                a = stack.pop()
                stack.append(f"({a} - {b})")

            elif op == OP_MUL:
                b = stack.pop()
                a = stack.pop()
                stack.append(f"({a} * {b})")

            elif op == OP_DIV:
                b = stack.pop()
                a = stack.pop()
                stack.append(f"protected_div({a}, {b})")

            elif op == OP_ABS:
                a = stack.pop()
                stack.append(f"abs({a})")

            elif op == OP_NEG:
                a = stack.pop()
                stack.append(f"(-{a})")

            elif op == OP_MIN:
                b = stack.pop()
                a = stack.pop()
                stack.append(f"min({a}, {b})")

            elif op == OP_MAX:
                b = stack.pop()
                a = stack.pop()
                stack.append(f"max({a}, {b})")

            elif op == OP_SQRT:
                a = stack.pop()
                stack.append(f"sqrt_abs({a})")

            elif op == OP_LOG:
                a = stack.pop()
                stack.append(f"log1p_abs({a})")

            elif op == OP_SIN:
                a = stack.pop()
                stack.append(f"sin({a})")

            elif op == OP_COS:
                a = stack.pop()
                stack.append(f"cos({a})")

            elif op == OP_TANH:
                a = stack.pop()
                stack.append(f"tanh({a})")

            elif op == OP_SIGMOID:
                a = stack.pop()
                stack.append(f"sigmoid({a})")

            elif op == OP_SQUARE:
                a = stack.pop()
                stack.append(f"square({a})")

            else:
                stack.append(f"UNKNOWN_OP_{op}")

        return stack[0] if stack else "<empty>"

    def to_latex(self, feature_names=None):
        expr = self.to_string(feature_names=feature_names)
        replacements = {
            "protected_div": r"\operatorname{pdiv}",
            "sqrt_abs": r"\sqrt{|\cdot|}",
            "log1p_abs": r"\log(1+|\cdot|)",
            "sigmoid": r"\sigma",
            "square": r"\operatorname{square}",
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        return expr


def subtree_bounds(ops, root_index):
    """
    Return [start, end) bounds for a postfix subtree rooted at root_index.
    """

    need = 1
    start = int(root_index)

    while start >= 0 and need > 0:
        op = int(ops[start])
        need -= 1
        need += op_arity(op)
        start -= 1

    if need != 0:
        raise ValueError("Invalid postfix tree while locating subtree.")

    return start + 1, int(root_index) + 1


def random_program(
    rng,
    n_features,
    max_depth=4,
    const_range=(-1.0, 1.0),
    function_set="default",
    p_const=0.20,
    p_unary=0.25,
    init_method="grow",
):
    """
    Create random valid postfix symbolic expression.
    """

    functions = resolve_function_set(function_set)
    unary_ops = [op for op in functions if is_unary(op)]
    binary_ops = [op for op in functions if is_binary(op)]

    if not binary_ops:
        raise ValueError("function_set must contain at least one binary operation.")

    ops = []
    args = []

    def grow(depth):
        if init_method == "full":
            make_leaf = depth >= max_depth
        elif init_method == "grow":
            make_leaf = depth >= max_depth or rng.random() < 0.25
        elif init_method == "half_and_half":
            if rng.random() < 0.5:
                make_leaf = depth >= max_depth
            else:
                make_leaf = depth >= max_depth or rng.random() < 0.25
        else:
            raise ValueError("init_method must be 'grow', 'full', or 'half_and_half'.")

        if make_leaf:
            if rng.random() < p_const:
                ops.append(OP_CONST)
                args.append(rng.uniform(const_range[0], const_range[1]))
            else:
                ops.append(OP_VAR)
                args.append(rng.integers(0, n_features))
            return

        use_unary = unary_ops and rng.random() < p_unary

        if use_unary:
            grow(depth + 1)
            op = int(rng.choice(unary_ops))
            ops.append(op)
            args.append(-1.0)
        else:
            grow(depth + 1)
            grow(depth + 1)
            op = int(rng.choice(binary_ops))
            ops.append(op)
            args.append(-1.0)

    grow(0)
    program = FastProgram(ops, args)

    if not program.is_valid():
        return random_program(
            rng=rng,
            n_features=n_features,
            max_depth=max_depth,
            const_range=const_range,
            function_set=function_set,
            p_const=p_const,
            p_unary=p_unary,
            init_method=init_method,
        )

    return program


def point_mutation(
    program,
    rng,
    n_features,
    mutation_rate=0.10,
    const_range=(-1.0, 1.0),
    function_set="default",
):
    child = program.copy()
    functions = resolve_function_set(function_set)
    unary_ops = [op for op in functions if is_unary(op)]
    binary_ops = [op for op in functions if is_binary(op)]

    for i in range(child.size):
        if rng.random() > mutation_rate:
            continue

        op = int(child.ops[i])

        if op == OP_VAR:
            child.args[i] = rng.integers(0, n_features)

        elif op == OP_CONST:
            child.args[i] = rng.uniform(const_range[0], const_range[1])

        elif is_unary(op) and unary_ops:
            child.ops[i] = int(rng.choice(unary_ops))

        elif is_binary(op) and binary_ops:
            child.ops[i] = int(rng.choice(binary_ops))

    return child


def subtree_mutation(
    program,
    rng,
    n_features,
    max_depth=4,
    const_range=(-1.0, 1.0),
    function_set="default",
):
    if program.size <= 1:
        return random_program(
            rng=rng,
            n_features=n_features,
            max_depth=max_depth,
            const_range=const_range,
            function_set=function_set,
        )

    root = int(rng.integers(0, program.size))
    start, end = subtree_bounds(program.ops, root)

    replacement = random_program(
        rng=rng,
        n_features=n_features,
        max_depth=max(1, max_depth // 2),
        const_range=const_range,
        function_set=function_set,
    )

    new_ops = np.concatenate([program.ops[:start], replacement.ops, program.ops[end:]])
    new_args = np.concatenate([program.args[:start], replacement.args, program.args[end:]])

    child = FastProgram(new_ops, new_args)

    if not child.is_valid():
        return program.copy()

    return child


def hoist_mutation(program, rng):
    if program.size <= 1:
        return program.copy()

    root = int(rng.integers(0, program.size))
    start, end = subtree_bounds(program.ops, root)

    if end - start <= 1:
        return program.copy()

    inner_root = int(rng.integers(start, end))
    inner_start, inner_end = subtree_bounds(program.ops, inner_root)

    if inner_start < start or inner_end > end:
        return program.copy()

    new_ops = np.concatenate([program.ops[:start], program.ops[inner_start:inner_end], program.ops[end:]])
    new_args = np.concatenate([program.args[:start], program.args[inner_start:inner_end], program.args[end:]])

    child = FastProgram(new_ops, new_args)
    return child if child.is_valid() else program.copy()


def subtree_crossover(parent_a, parent_b, rng, max_nodes=None):
    if parent_a.size == 0 or parent_b.size == 0:
        return parent_a.copy()

    root_a = int(rng.integers(0, parent_a.size))
    root_b = int(rng.integers(0, parent_b.size))

    start_a, end_a = subtree_bounds(parent_a.ops, root_a)
    start_b, end_b = subtree_bounds(parent_b.ops, root_b)

    new_ops = np.concatenate([parent_a.ops[:start_a], parent_b.ops[start_b:end_b], parent_a.ops[end_a:]])
    new_args = np.concatenate([parent_a.args[:start_a], parent_b.args[start_b:end_b], parent_a.args[end_a:]])

    if max_nodes is not None and len(new_ops) > max_nodes:
        return parent_a.copy()

    child = FastProgram(new_ops, new_args)

    if not child.is_valid():
        return parent_a.copy()

    return child
