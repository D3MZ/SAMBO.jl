# Public API

The primary entry point is `minimize(objective, space; algorithm, options...)`. The
equivalent lifecycle is `solve(problem, algorithm) = solve!(init(problem, algorithm))`;
`step!` advances one solver iteration.

Algorithms are immutable values: `SMBO()`, `SCEUA()`, `SHGO()`, and
`TopologicalMultistart()`. Common options are
`maximum_evaluations`, `maximum_iterations`, `absolute_tolerance`,
`relative_tolerance`, `stall_evaluations`, `time_limit`, `initial_points`,
`initial_values`, `rng`, `executor`, and `callback`.

Coordinate and objective scalar types are independent. Coordinates follow the search
space; `objective_type` selects the objective scalar type and defaults to `Float64`.

Results are read through `minimizer`, `minimum`, `trace`, `retcode`,
`evaluation_count`, `iteration_count`, `fittedmodel`, and `observations`.

SMBO additionally supports `ask!`, `tell!`, `cancel!`, `fail!`, `checkpoint`,
`restore`, and `result`. A `CandidateBatch` has a stable identifier and owns its
latent points; outstanding batches may be partially completed and resolved in any
order. Callbacks observe committed batches atomically.

Optional package extensions activate automatically for Makie, SurrogatesBase,
OptimizationBase, and MLJTuning. `SAMBOTuning()` is the MLJTuning strategy constructor.
