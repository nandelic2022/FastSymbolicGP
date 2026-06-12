"""
Operation codes for FastSymbolicGP.

Expressions are represented in postfix/bytecode form.

Example:
    x0 x1 mul x2 add

means:
    (x0 * x1) + x2
"""

OP_VAR = 0
OP_CONST = 1

OP_ADD = 2
OP_SUB = 3
OP_MUL = 4
OP_DIV = 5

OP_ABS = 6
OP_NEG = 7
OP_MIN = 8
OP_MAX = 9
OP_SQRT = 10
OP_LOG = 11
OP_SIN = 12
OP_COS = 13
OP_TANH = 14
OP_SIGMOID = 15
OP_SQUARE = 16


ARITY = {
    OP_VAR: 0,
    OP_CONST: 0,
    OP_ADD: 2,
    OP_SUB: 2,
    OP_MUL: 2,
    OP_DIV: 2,
    OP_ABS: 1,
    OP_NEG: 1,
    OP_MIN: 2,
    OP_MAX: 2,
    OP_SQRT: 1,
    OP_LOG: 1,
    OP_SIN: 1,
    OP_COS: 1,
    OP_TANH: 1,
    OP_SIGMOID: 1,
    OP_SQUARE: 1,
}

OP_NAMES = {
    OP_VAR: "x",
    OP_CONST: "c",
    OP_ADD: "add",
    OP_SUB: "sub",
    OP_MUL: "mul",
    OP_DIV: "div",
    OP_ABS: "abs",
    OP_NEG: "neg",
    OP_MIN: "min",
    OP_MAX: "max",
    OP_SQRT: "sqrt",
    OP_LOG: "log",
    OP_SIN: "sin",
    OP_COS: "cos",
    OP_TANH: "tanh",
    OP_SIGMOID: "sigmoid",
    OP_SQUARE: "square",
}

NAME_TO_OP = {v: k for k, v in OP_NAMES.items() if k not in (OP_VAR, OP_CONST)}

FAST_FUNCTION_SET = [
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_ABS,
    OP_NEG,
    OP_MIN,
    OP_MAX,
]

DEFAULT_FUNCTION_SET = [
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
    OP_SQUARE,
]

EXTENDED_FUNCTION_SET = [
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
]


def resolve_function_set(function_set):
    if function_set is None or function_set == "default":
        return list(DEFAULT_FUNCTION_SET)

    if function_set == "fast":
        return list(FAST_FUNCTION_SET)

    if function_set == "extended":
        return list(EXTENDED_FUNCTION_SET)

    resolved = []
    for item in function_set:
        if isinstance(item, str):
            if item not in NAME_TO_OP:
                raise ValueError(f"Unknown function name: {item}")
            resolved.append(NAME_TO_OP[item])
        else:
            resolved.append(int(item))

    return resolved


def op_arity(op):
    return ARITY[int(op)]


def is_unary(op):
    return ARITY[int(op)] == 1


def is_binary(op):
    return ARITY[int(op)] == 2
