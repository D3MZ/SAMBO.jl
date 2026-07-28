# Public API

The primary entry point is `minimize(objective, space; algorithm, options...)`.
The equivalent lifecycle is
`solve(problem, algorithm) = solve!(init(problem, algorithm))`; `step!`
advances one solver iteration.

Algorithms are immutable values: `SMBO()`, `SCEUA()`, `SHGO()`, and
`TopologicalMultistart()`. Solver and workspace state is mutable. Common options
include `maximum_evaluations`, `maximum_iterations`, tolerances, `time_limit`,
initial observations, `rng`, `executor`, and `callback`.

Results are read through `bestpoint`, `bestvalue`, `optimizationsense`, `trace`,
`retcode`, `evaluation_count`, `iteration_count`, `fittedmodel`, and
`observations`. `minimizer` and `minimum` remain compatibility aliases.

SMBO supports `ask!`, `tell!`, `cancel!`, `fail!`, `checkpoint`, `restore`, and
`result`. Outstanding `SAMBO.CandidateBatch` values may be partially completed and
resolved in any order. Callbacks observe committed batches atomically.

Configuration-dependent behavior is represented by policy values. Candidate
generators report the number of points generated and are never silently replaced.
Topology, uncertainty, refitting, equality, and SHGO minimization schedules are
likewise explicit.

SHGO uses `SAMBO.RandomShiftedSampling(SAMBO.SobolDesign())` by default so each seeded run
has one reproducible continuing QMC stream. Local searches use
`SAMBO.GlobalBoxLocalBounds()` by default; `SAMBO.TopographicalLocalBounds()` explicitly
restricts them to neighboring sample vertices.

## Surrogate extension protocol

External surrogate packages extend these public functions:

- `fitmodel`
- `predictmean!`
- `predictmeanvariance!`
- `predictionworkspace`
- `clone_surrogate`

Candidate generators extend `generate_candidates!`. SHGO local solvers extend
`local_minimize!` and use the documented solver-context accessors `problem`,
`space`, `rng`, `remaining_evaluations`, `iteration`, `evaluate!`, and
`isfeasible`; extension code does not access solver fields.

Deterministic surrogate models pair with `GreedyMean()`. Acquisitions requiring
uncertainty use a stochastic model or an explicit composition such as
`DistanceUncertainty(model)`.
