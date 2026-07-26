using CairoMakie
using DelimitedFiles
using Random
using Sambo

root = @__DIR__
output = joinpath(root, "plots", "julia")
mkpath(output)

raw = readdlm(joinpath(root, "plots", "trace.csv"), ',', Float64; skipstart=1)
space = Sambo.Box(fill(-1.0, 3), fill(1.0, 3))
latent = reduce(hcat, (Sambo.encode(space, @view raw[row, 1:3]) for row in axes(raw, 1)))
values = vec(raw[:, 4])
trace = Sambo.Trace{Float64}(3, length(values))
trace.latent_points .= latent
trace.objective_values .= values
trace.feasible .= true
trace.iterations .= eachindex(values)
trace.elapsed_seconds .= 0
trace.count = length(values)
model = Sambo.fitmodel(
    Sambo.GaussianProcessSurrogate(),
    Sambo.latentpoints(trace),
    Sambo.objectivevalues(trace),
    MersenneTwister(42),
)
best = argmin(values)
result = Sambo.Result(
    space,
    Sambo.decode(space, @view latent[:, best]),
    values[best],
    trace,
    model,
    Sambo.SMBO(),
    :evaluation_limit,
    (evaluations=length(values), iterations=length(values)),
)

figures = (
    convergence=Sambo.convergenceplot(result; optimum=0.0),
    regret=Sambo.regretplot(result; optimum=0.0),
    objective=Sambo.objectiveplot(
        result;
        dimensions=1:3,
        names=["x", "y", "z"],
        resolution=18,
        samples=128,
        rng=MersenneTwister(42),
    ),
    evaluations=Sambo.evaluationsplot(
        result;
        dimensions=1:3,
        names=["x", "y", "z"],
    ),
)

for (name, figure) in pairs(figures)
    save(joinpath(output, "$name.png"), figure; px_per_unit=1.2)
end
