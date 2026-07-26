using SAMBO
using JET
using Random

problem = Problem(x -> sum(abs2, x), Box(fill(-1.0, 2), fill(1.0, 2)))
candidates = rand(MersenneTwister(430), 2, 4)
values = zeros(4)
JET.@test_opt SAMBO.evaluate!(
    values,
    Serial(),
    problem,
    candidates,
    ErrorOnNonfinite(),
)

training_points = rand(MersenneTwister(431), 2, 8)
training_values = vec(sum(abs2, training_points; dims=1))
model = SAMBO.fitmodel(
    GaussianProcessSurrogate(),
    training_points,
    training_values,
)
means = zeros(4)
variances = zeros(4)
prediction = GPPredictionWorkspace(Float64)
SAMBO.predictmeanvariance!(
    means,
    variances,
    model,
    candidates,
    prediction,
)
JET.@test_opt SAMBO.predictmeanvariance!(
    means,
    variances,
    model,
    candidates,
    prediction,
)

sceua = init(
    problem,
    SCEUA();
    maximum_evaluations=50,
    rng=MersenneTwister(432),
)
step!(sceua)
JET.@test_opt step!(sceua)

smbo = init(
    Problem(Box(fill(-1.0, 2), fill(1.0, 2))),
    SMBO(initial_points=5, candidate_pool=64);
    maximum_evaluations=20,
    rng=MersenneTwister(433),
)
JET.@test_opt ask!(smbo, 2)
