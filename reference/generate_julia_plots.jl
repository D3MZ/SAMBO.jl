using CairoMakie
using DelimitedFiles
using Random
using SAMBO

root = @__DIR__
output = joinpath(root, "plots", "julia")
mkpath(output)

raw = readdlm(joinpath(root, "plots", "trace.csv"), ',', Float64; skipstart=1)
space = SAMBO.Box(fill(-1.0, 3), fill(1.0, 3))
latent = reduce(hcat, (SAMBO.encode(space, @view raw[row, 1:3]) for row in axes(raw, 1)))
values = vec(raw[:, 4])
trace = SAMBO.Trace{Float64}(3, length(values))
trace.latent_points .= latent
trace.objective_values .= values
trace.source .= SAMBO.InternalEvaluation
trace.evaluation_numbers .= eachindex(values)
trace.iterations .= eachindex(values)
trace.elapsed_seconds .= 0
trace.count = length(values)
model = SAMBO.fitmodel(
    SAMBO.GaussianProcessSurrogate(),
    SAMBO.latentpoints(trace),
    SAMBO.objectivevalues(trace),
    MersenneTwister(42),
)
best = argmin(values)
result = SAMBO.Result(
    SAMBO.Problem(nothing, space),
    space,
    SAMBO.decode(space, @view latent[:, best]),
    values[best],
    trace,
    model,
    SAMBO.SMBO(),
    SAMBO.Minimize(),
    :evaluation_limit,
    (evaluations=length(values), iterations=length(values)),
)

figures = (
    convergence=SAMBO.convergenceplot(result; optimum=0.0),
    regret=SAMBO.regretplot(result; optimum=0.0),
    objective=SAMBO.objectiveplot(
        result;
        dimensions=1:3,
        names=["x", "y", "z"],
        resolution=18,
        samples=128,
        rng=MersenneTwister(42),
    ),
    evaluations=SAMBO.evaluationsplot(
        result;
        dimensions=1:3,
        names=["x", "y", "z"],
    ),
)

for (name, figure) in pairs(figures)
    save(joinpath(output, "$name.png"), figure; px_per_unit=1.2)
end
