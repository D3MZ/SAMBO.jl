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

(minimum(result), minimizer(result), evaluation_count(result))
```

For externally evaluated objectives:

```julia
state = init(Problem(space), SMBO(); maximum_evaluations=100)
batch = ask!(state, 4)
tell!(state, batch, map(expensive_measurement, batch))
```

Partial completion, cancellation, failure recording, and restart are available through
`tell!(state, batch, indices, values)`, `cancel!`, `fail!`, `checkpoint`, and `restore`.

The detailed behavioral contracts live in
[`spec/api.md`](https://github.com/D3MZ/SAMBO.jl/blob/main/spec/api.md),
[`spec/spaces.md`](https://github.com/D3MZ/SAMBO.jl/blob/main/spec/spaces.md), and
[`spec/termination.md`](https://github.com/D3MZ/SAMBO.jl/blob/main/spec/termination.md).

```@index
```

```@autodocs
Modules = [SAMBO]
```
