def warmup(target="all"):
    from .diagnostics import self_test
    return self_test(verbose=False)
