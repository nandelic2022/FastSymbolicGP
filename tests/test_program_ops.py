import numpy as np

from fastsymbolicgp.core.program import random_program, subtree_crossover, subtree_mutation
from fastsymbolicgp.core.evaluator import evaluate_program


def test_random_program_valid_and_evaluates():
    rng = np.random.default_rng(42)
    p = random_program(rng, n_features=4, max_depth=3, function_set="fast")

    assert p.is_valid()
    assert p.size > 0

    X = rng.normal(size=(20, 4))
    y = evaluate_program(p.ops, p.args, X)

    assert y.shape == (20,)
    assert np.all(np.isfinite(y))


def test_subtree_mutation_and_crossover_valid():
    rng = np.random.default_rng(42)
    a = random_program(rng, n_features=4, max_depth=3, function_set="fast")
    b = random_program(rng, n_features=4, max_depth=3, function_set="fast")

    m = subtree_mutation(a, rng, n_features=4, max_depth=3, function_set="fast")
    c = subtree_crossover(a, b, rng)

    assert m.is_valid()
    assert c.is_valid()
