using SAMBO
using SurrogatesBase
using Random
using Test

mutable struct NearestSurrogate <: SurrogatesBase.AbstractDeterministicSurrogate
    points::Vector{Vector{Float64}}
    values::Vector{Float64}
end
NearestSurrogate() = NearestSurrogate(Vector{Float64}[], Float64[])
function (surrogate::NearestSurrogate)(points)
    map(points) do point
        index = argmin(
            sum(abs2, point .- observation)
            for observation in surrogate.points
        )
        surrogate.values[index]
    end
end
function SurrogatesBase.update!(surrogate::NearestSurrogate, points, values)
    append!(surrogate.points, points)
    append!(surrogate.values, values)
    return surrogate
end

@testset "SurrogatesBase extension" begin
    mean_only = SAMBO.minimize(
        point -> sum(abs2, point),
        SAMBO.Box([-1.0], [1.0]);
        algorithm=SAMBO.SMBO(
            surrogate=NearestSurrogate(),
            acquisition=SAMBO.GreedyMean(),
            candidate_pool=16,
        ),
        maximum_evaluations=4,
        rng=MersenneTwister(51),
    )
    @test SAMBO.evaluation_count(mean_only) == 4

    @test_throws MethodError SAMBO.minimize(
        point -> sum(abs2, point),
        SAMBO.Box([-1.0], [1.0]);
        algorithm=SAMBO.SMBO(
            surrogate=NearestSurrogate(),
            acquisition=SAMBO.LowerConfidenceBound(),
            initial_points=2,
            candidate_pool=16,
        ),
        maximum_evaluations=4,
        rng=MersenneTwister(52),
    )

    uncertain = SAMBO.minimize(
        point -> sum(abs2, point),
        SAMBO.Box([-1.0], [1.0]);
        algorithm=SAMBO.SMBO(
            surrogate=SAMBO.DistanceUncertainty(NearestSurrogate()),
            acquisition=SAMBO.LowerConfidenceBound(),
            initial_points=2,
            candidate_pool=16,
        ),
        maximum_evaluations=4,
        rng=MersenneTwister(53),
    )
    @test SAMBO.evaluation_count(uncertain) == 4
end
