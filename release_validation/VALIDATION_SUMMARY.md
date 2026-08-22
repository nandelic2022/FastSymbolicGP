# FastSymbolicGP V1.0.0 validation summary

- Version: 1.0.0
- Automated tests: 29 passed
- Source compileall: passed
- Wheel build: passed
- Clean target wheel install/import: passed
- Installed wheel execution engine: population_numba
- Installed wheel fitness engine: fused
- `summary()` and `model_manifest()`: passed
- `warmup("all")`: passed
- Exact validation champion archive test: passed
- Named preset non-default override preservation: passed
- `preset="auto"` small-workload planner test: passed
- Flat compiled genome/index cache test: passed
- Champion constant optimization smoke test: passed
- Forced 120-generation Breast Cancer acceptance run: 3.8615 s on build environment
- Acceptance balanced accuracy: 0.957871

The acceptance timing is specific to the build host and warmed JIT state. It demonstrates that the target workload can execute below five seconds in this environment; it is not a hardware-independent guarantee.
