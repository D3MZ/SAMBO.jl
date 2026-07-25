# Python SAMBO parity matrix

Parity means equivalent capability and semantic output, not identical stochastic trajectories or internal design.

| Capability | Julia status | Tests / notes |
|---|---|---|
| `minimize` / result | Implemented | `test/algorithms.jl` |
| Continuous boxes | Implemented | `test/spaces.jl` |
| Integer ranges / categories | Implemented | `test/spaces.jl` |
| Callable constraints | Implemented | feasible sampling and proposals |
| Exact evaluation budgets | Implemented | all three algorithms |
| SCE-UA | Experimental | native compact complex evolution; deeper reference suite pending |
| SMBO | Experimental | native RBF uncertainty heuristic |
| Ask/tell | Implemented | identified pending batches |
| Multiple outstanding batches | Implemented | pending dictionary; dedicated tests pending |
| SHGO | Experimental | global/local heuristic; simplicial homology parity pending |
| Parallel evaluation | Implemented | `Threaded()` with serial trace commit |
| Convergence plot | Implemented | Makie extension; visual tests pending |
| Regret plot | Implemented | Makie extension; visual tests pending |
| Objective plot | Implemented | 1D PDP / 2D contours; categorical styling and embedding pending |
| Evaluations plot | Implemented | lower triangular matrix; embedding and reference images pending |
| Tables.jl observations | Implemented | `test/diagnostics.jl` |
| SearchCV / MLJ | Planned | should be an MLJTuning extension |
| Optimization.jl | Planned | extension |

## Release gate

Before claiming parity: port the upstream public behavioral cases, add saved seeded reference data, implement simplicial SHGO, add CairoMakie reference images covering continuous/integer/categorical spaces, and run statistical solution-quality comparisons over multiple seeds.
