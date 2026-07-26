using SurrogatesBase
using OptimizationBase
using MLJTuning

mutable struct NearestSurrogate <: SurrogatesBase.AbstractDeterministicSurrogate
    points::Vector{Vector{Float64}}
    values::Vector{Float64}
end

mutable struct TuningDummy
    rate::Float64
    mode::Symbol
end

@testset "MLJTuning extension" begin
    tuning = SAMBO.SAMBOTuning(
        algorithm=SAMBO.SMBO(candidate_pool=32, batch_size=1),
        rng=MersenneTwister(13),
    )
    ranges = [
        range(Float64, :rate; lower=0.1, upper=1.0),
        range(Symbol, :mode; values=[:fast, :accurate]),
    ]
    state = MLJTuning.setup(tuning, TuningDummy(0.5, :fast), ranges, 3, 0)
    models, state = MLJTuning.models(
        tuning,
        TuningDummy(0.5, :fast),
        NamedTuple[],
        state,
        3,
        0,
    )
    @test length(models) == 1
    @test 0.1 <= models[1].rate <= 1.0
    @test models[1].mode in (:fast, :accurate)
    history = [(measurement=[models[1].rate],)]
    next_models, state = MLJTuning.models(
        tuning,
        TuningDummy(0.5, :fast),
        history,
        state,
        2,
        0,
    )
    @test length(next_models) == 1
end

@testset "OptimizationBase extension" begin
    problem = OptimizationBase.OptimizationProblem(
        (point, parameters) -> sum(abs2, point),
        zeros(2);
        lb=fill(-1.0, 2),
        ub=fill(1.0, 2),
    )
    solution = OptimizationBase.solve(
        problem,
        SAMBO.SMBO(candidate_pool=64);
        maxiters=10,
        rng=MersenneTwister(6),
    )
    @test solution.stats.fevals == 10
    @test solution.original isa SAMBO.Result
    @test solution.u == SAMBO.minimizer(solution.original)
end
NearestSurrogate() = NearestSurrogate(Vector{Float64}[], Float64[])
function (surrogate::NearestSurrogate)(points)
    map(points) do point
        index = argmin(sum(abs2, point .- observation) for observation in surrogate.points)
        surrogate.values[index]
    end
end
function SurrogatesBase.update!(surrogate::NearestSurrogate, points, values)
    append!(surrogate.points, points)
    append!(surrogate.values, values)
    return surrogate
end

@testset "SurrogatesBase extension" begin
    result = SAMBO.minimize(
        point -> sum(abs2, point),
        SAMBO.Box([-1.0, -1.0], [1.0, 1.0]);
        algorithm=SAMBO.SMBO(
            surrogate=NearestSurrogate(),
            candidate_pool=64,
        ),
        maximum_evaluations=10,
        rng=MersenneTwister(5),
    )
    @test SAMBO.evaluation_count(result) == 10
    @test isfinite(SAMBO.minimum(result))
    @test !isnothing(SAMBO.fittedmodel(result))
end
