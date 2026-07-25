# Sambo.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://D3MZ.github.io/Sambo.jl/dev/)
[![CI](https://github.com/D3MZ/Sambo.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/D3MZ/Sambo.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/D3MZ/Sambo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/D3MZ/Sambo.jl)

Julia-native global black-box optimization inspired by Python [SAMBO](https://sambo-optimization.github.io/). It currently provides experimental SCE-UA-style and SHGO-style heuristics, sequential model-based optimization (including ask/tell), mixed variables, constraints, traces, Tables.jl observations, and optional Makie diagnostics.

> **Status:** experimental pre-1.0 software. The APIs work and are tested, but algorithmic/visual parity is still tracked in [`spec/parity.md`](spec/parity.md); this repository does not claim numerical trajectory or pixel parity yet.

## Install

```julia
pkg> add https://github.com/D3MZ/Sambo.jl
```

## Use

```julia
using Sambo, Random

space = SearchSpace(
    count = 0:99,
    price = Continuous(0.5, 9.0),
    channel = Choices(:search, :social, :print),
)

result = minimize(space; algorithm=SMBO(), maximum_evaluations=100,
                  rng=Xoshiro(42)) do x
    (x.count - 40)^2 + (x.price - 3)^2 + (x.channel == :search ? 0 : 10)
end

minimum(result)
minimizer(result)
observations(result) # Tables.jl row source
```

Other solvers use the same API:

```julia
solve(Problem(f, Box(lower, upper)), SCEUA(); maximum_evaluations=500)
solve(Problem(f, Box(lower, upper)), SHGO(); maximum_evaluations=500)
```

External evaluations use `ask!`/`tell!`:

```julia
state = init(Problem(space), SMBO(); maximum_evaluations=100)
batch = ask!(state, 4)
tell!(state, batch, map(expensive_measurement, batch))
```

Load Makie to activate `convergenceplot`, `regretplot`, `objectiveplot`, and `evaluationsplot` through a package extension.

## Reproducible microbenchmark

The benchmark scripts use the two-dimensional Rosenbrock objective and report solution quality and actual objective-call counts alongside timing. They deliberately do **not** publish a cross-language speedup table:

- the Julia implementations are compact experimental heuristics, not equivalent implementations of Python SAMBO's production algorithms;
- Julia's `maximum_evaluations` and Python SAMBO's `max_iter` are different controls, so requested settings and observed calls are both reported;
- Julia allocation counts and Python `tracemalloc` blocks have different definitions and are only meaningful within their own runtime; and
- Julia uses `BenchmarkTools` samples after compilation, while Python reports the median of uninstrumented runs and measures memory separately.

Set up and run the Julia benchmark with:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/benchmarks.jl
```

Run the Python reference separately with `python3 benchmark/python_reference.py` after installing its `sambo`, SciPy, scikit-learn, and joblib dependencies. Capture environment details with any results; machine-specific CSV snapshots are intentionally not treated as package documentation.

## License and provenance

AGPL-3.0-or-later. Python SAMBO is AGPL-licensed; using the compatible license is the conservative choice for a source-guided port. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for provenance rules.
