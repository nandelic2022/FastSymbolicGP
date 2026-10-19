# FastSymbolicGP V1.2.5 — Runtime UX & Smart Profiling

V1.2.5 keeps the V1.2.4 search/execution core and adds a dependency-free runtime console designed for long symbolic-evolution runs.

## Highlights
- Startup banner on normal runs showing version, task, data size, search budget, backend, fitness engine, CPU/jobs, islands, and active scientific/search modes.
- `verbose=1`: startup banner + periodic progress + final summary.
- `verbose=2`: startup banner + every generation + final summary.
- `display="off"`: fully silent mode.
- Per-generation status includes BEST marker, progress bar, train/validation loss, complexity, diversity, throughput, and ETA.
- Final summary reports runtime, evaluations, programs/s, champion complexity, resolved backend/engine, dominant bottleneck, cache hit rate, duplicate reuse, simplification, and the final fitted equation.
- No Rich/curses dependency; works in standard terminals and Spyder consoles.

## Compatibility
No search algorithm defaults were changed from V1.2.4.
