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
    optimization_extension = Base.get_extension(SAMBO, :SAMBOOptimizationExt)
    constrained = (
        lcons=[0.0],
        ucons=[1.0],
        f=(cons=(point, parameters) -> [point[1]],),
        p=nothing,
    )
    constraint = optimization_extension._constraint(constrained)
    @test constraint([0.5])
    @test !constraint([1.5])
    @test_throws ArgumentError optimization_extension._constraint((
        lcons=[0.0],
        ucons=[1.0],
        f=identity,
        p=nothing,
    ))

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

    maximization_problem = OptimizationBase.OptimizationProblem(
        (point, parameters) -> -(point[1] - 0.7)^2,
        nothing;
        lb=[0.0],
        ub=[1.0],
        sense=OptimizationBase.MaxSense,
    )
    maximized = OptimizationBase.solve(
        maximization_problem,
        SAMBO.SMBO(candidate_pool=32);
        maxiters=12,
        rng=MersenneTwister(60),
    )
    @test maximized.objective > -0.05
    @test maximized.original.sense isa SAMBO.Maximize
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
            acquisition=SAMBO.GreedyMean(),
            candidate_pool=64,
        ),
        maximum_evaluations=10,
        rng=MersenneTwister(5),
    )
    @test SAMBO.evaluation_count(result) == 10
    @test isfinite(SAMBO.minimum(result))
    @test !isnothing(SAMBO.fittedmodel(result))

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
        rng=MersenneTwister(51),
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
        rng=MersenneTwister(52),
    )
    @test SAMBO.evaluation_count(uncertain) == 4
end
