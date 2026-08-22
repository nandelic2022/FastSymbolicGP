# FastSymbolicGP V0.9.0 versus V0.8.0 speed validation

Warmed local benchmark: 12,000 samples, 30 features, 64 programs, five generations, four workers, five timed repetitions, and identical random seeds.

| Task | V0.8.0 median | V0.9.0 median | Speed-up | Runtime reduction |
|---|---:|---:|---:|---:|
| Binary classification | 0.277795 s | 0.215723 s | 1.29x | 22.3% |
| Regression | 0.339764 s | 0.123753 s | 2.75x | 63.6% |

V0.8.0 performed 320 nominal classifier evaluations and 320 nominal regressor evaluations. V0.9.0 performed 251 actual classifier evaluations and 259 actual regressor evaluations, with 69 and 61 exact cache/reuse hits, respectively.

This is a release-build validation result and not a universal performance guarantee.
