"""Lightweight constant optimization for final symbolic champions.

V1 keeps this dependency-free: a deterministic coordinate search refines only
constants in the selected champion, so it does not add SciPy as a core dependency.
"""
from __future__ import annotations

import numpy as np
from .program import SymbolicProgram


def optimize_program_constants(program: SymbolicProgram, objective, const_range=(-1.0, 1.0), rounds=2, scale=0.10):
    indexed = program.indexed_nodes()
    constant_paths = [path for path, node in indexed if node.kind == "constant"]
    if not constant_paths or int(rounds) <= 0:
        return program.clone(), {"enabled": True, "constants": len(constant_paths), "improved": False, "evaluations": 0}

    lo, hi = map(float, const_range)
    span = max(abs(hi - lo), 1e-9)
    current = program.clone()
    current_score = float(objective(current))
    initial_score = current_score
    evaluations = 1
    step = float(scale) * span

    for _ in range(int(rounds)):
        for path in constant_paths:
            base_value = float(current.root.get_path(path).value)
            best_local = current
            best_score = current_score
            for candidate_value in (base_value - step, base_value + step):
                trial = current.clone()
                trial.root.get_path(path).value = float(np.clip(candidate_value, lo, hi))
                # A clone carries cached metadata; structural caches are invalid after mutation.
                trial.metadata_.clear()
                score = float(objective(trial))
                evaluations += 1
                if score < best_score - 1e-15:
                    best_score = score
                    best_local = trial
            current, current_score = best_local, best_score
        step *= 0.35

    return current, {
        "enabled": True,
        "constants": len(constant_paths),
        "improved": bool(current_score < initial_score - 1e-15),
        "initial_objective": float(initial_score),
        "final_objective": float(current_score),
        "evaluations": int(evaluations),
        "rounds": int(rounds),
    }
