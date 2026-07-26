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

Configuration-dependent behavior is represented by policy values. Examples include
`FixedRefit`/`SquareRootRefit`, `ExactCandidateEquality`/
`ApproximateCandidateEquality`, `MinimizeEveryRefinement`/
`MinimizeAtTermination`, and `DelaunayTopology`/`KNearestTopology`.
`MixtureCandidates` composes candidate generators with exact component quotas; a
generator returns the number of candidates it produced and is never silently replaced
by another generator.

```julia
SMBO(
    refit_schedule=FixedRefit(4),
    candidate_sampler=MixtureCandidates(
        UniformCandidates(),
        EliteGaussianCandidates(0.1);
        global_fraction=0.25,
    ),
    candidate_equality=ApproximateCandidateEquality(1e-8),
)

SHGO(
    topology=DelaunayTopology(),
    minimization_schedule=MinimizeAtTermination(),
)
```

`DelaunayTopology()` raises `ComplexConstructionError` for a degenerate complex.
`KNearestTopology(neighbors=8)` is a separate, explicit topology, not a recovery path.
`CandidateGenerationError` reports a generator that cannot produce its assigned
candidates.

Deterministic SurrogatesBase models use `GreedyMean()`. An acquisition requiring
uncertainty must use a stochastic model or an explicit uncertainty composition such as
`DistanceUncertainty(model)`.

The native Gaussian process accepts typed length-scale and jitter policies:

```julia
GaussianProcessSurrogate(
    length_scale=ARDLengthScale([0.2, 0.8]),
    jitter=GeometricJitter(1e-10, 10.0, 8),
)
```

`convergencedata` includes known observations by default, while `regretdata` charges
only actual internal or external evaluations. Partial dependence defaults to
`FeasibleConditionalDependence()`; `UnconstrainedModelDependence()` is available when
the fitted model should be queried without constraint conditioning.

Optional package extensions activate automatically for Makie, SurrogatesBase,
OptimizationBase, and MLJTuning. `SAMBOTuning()` is the MLJTuning strategy constructor.
