```@meta
CurrentModule = SAMBO
```

# SAMBO

SAMBO is a Julia-native global black-box optimizer with SCE-UA, SMBO, and
iterative simplicial homology global optimization (SHGO),
mixed-variable spaces, constraints, ask/tell operation, observation tables, and optional
Makie plots.

```@example getting-started
using SAMBO, Random

space = SearchSpace(
    count=0:20,
    rate=Continuous(0.1, 2.0),
    method=Choices(:fast, :accurate),
)

result = minimize(
    space;
    algorithm=SMBO(candidate_pool=512),
    maximum_evaluations=30,
    rng=MersenneTwister(42),
) do point
    (point.count - 8)^2 + (point.rate - 0.7)^2 +
        (point.method == :fast ? 1.0 : 0.0)
end

(bestvalue(result), bestpoint(result), evaluation_count(result))
```

For externally evaluated objectives:

```julia
state = init(Problem(space), SMBO(); maximum_evaluations=100)
batch = ask!(state, 4)
tell!(state, batch, map(expensive_measurement, batch))
```

Partial completion, cancellation, failure recording, and restart are available through
`tell!(state, batch, indices, values)`, `cancel!`, `fail!`, `checkpoint`, and `restore`.

## Explicit policies

Semantic choices are values passed to the algorithms:

```julia
SMBO(
    refit_schedule=SquareRootRefit(2),
    candidate_sampler=MixtureCandidates(
        UniformCandidates(),
        EliteGaussianCandidates(0.1);
        global_fraction=0.25,
    ),
    candidate_equality=ApproximateCandidateEquality(1e-8),
)

SHGO(
    topology=KNearestTopology(neighbors=8),
    minimization_schedule=MinimizeAtTermination(),
)
```

Candidate generation, topology, surrogate uncertainty, refitting, duplicate handling,
and SHGO local-minimization timing are never changed by an implicit fallback.

Deterministic SurrogatesBase models pair with `GreedyMean()`. Use
`DistanceUncertainty(model)` only when distance-based uncertainty is deliberately part
of the model.

Search-space coordinates are parametric:

```julia
SearchSpace(Float32; x=Continuous(-1, 1), n=0:10)
SearchSpace(BigFloat; x=Continuous(big"-1", big"1"))
```

`convergencedata` includes known observations at evaluation zero by default.
`regretdata` counts evaluated observations only. Constrained partial dependence uses
`FeasibleConditionalDependence()`; select `UnconstrainedModelDependence()` explicitly
to query the fitted model without feasibility conditioning.
`evaluationsdata` and `evaluationsplot` also default to evaluated observations;
use `IncludeKnownObservations()` to include seeded points.

OptimizationBase callbacks receive `OptimizationState` and the current objective.
`SAMBOTuning()` preserves MLJ measure orientation and linear, logarithmic, or custom
numeric-range scaling.

```@index
```

```@autodocs
Modules = [SAMBO]
```
