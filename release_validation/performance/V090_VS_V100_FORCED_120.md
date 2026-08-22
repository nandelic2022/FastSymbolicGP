# V0.9.0 vs V1.0.0 forced 120-generation acceptance benchmark

Configuration: Breast Cancer dataset, 70/30 stratified split, 64 programs, 120 generations, two islands, no early stopping, population Numba execution, fused fitness, warmed kernel, dashboard off.

| Version | Fit time | Test balanced accuracy |
|---|---:|---:|
| V0.9.0 | 4.3323 s | 0.950059 |
| V1.0.0 | 4.1867 s | 0.957871 |

A subsequent V1-only acceptance run completed the same forced 120 generations in 3.8615 s with balanced accuracy 0.957871. Runtime varies with host load and JIT/cache state; these values are build-environment validation, not universal guarantees.
