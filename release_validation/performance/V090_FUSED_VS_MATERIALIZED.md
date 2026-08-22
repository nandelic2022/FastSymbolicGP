# FastSymbolicGP V0.9.0 fused versus materialized scoring

This benchmark compares V0.9.0 with the V0.8-style materialized scoring path and the V0.9.0 fused scoring plus exact cache reuse.

| Task | Materialized median | Fused median | Speed-up |
|---|---:|---:|---:|
| Binary classification | 0.275646 s | 0.183810 s | 1.50x |
| Regression | 0.327222 s | 0.091014 s | 3.60x |

Results depend on the hardware, operating system, dataset, symbolic-program complexity, and enabled options.
