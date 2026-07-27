using SAMBO
using Test
using Random
using Statistics
using OptimizationBase
using MLJTuning
using SurrogatesBase

const MLJBase = MLJTuning.MLJBase

mutable struct VerdictLinearModel <: MLJBase.Deterministic
    slope::Float64
end

MLJBase.fit(model::VerdictLinearModel, verbosity::Int, X, y) =
    model.slope, nothing, nothing
MLJBase.predict(::VerdictLinearModel, slope, Xnew) =
    slope .* Xnew.x

mutable struct VerdictStochasticSurrogate <:
        SurrogatesBase.AbstractStochasticSurrogate
    points::Vector{Vector{Float64}}
    values::Vector{Float64}
end

VerdictStochasticSurrogate() =
    VerdictStochasticSurrogate(Vector{Float64}[], Float64[])

function SurrogatesBase.update!(
    surrogate::VerdictStochasticSurrogate,
    points,
    values,
)
    append!(surrogate.points, points)
    append!(surrogate.values, values)
    return surrogate
end

struct VerdictPosterior
    means::Vector{Float64}
    variances::Vector{Float64}
end

Statistics.mean(posterior::VerdictPosterior) = posterior.means
Statistics.var(posterior::VerdictPosterior) = posterior.variances

function SurrogatesBase.finite_posterior(
    surrogate::VerdictStochasticSurrogate,
    points,
)
    means = map(points) do point
        index = argmin(
            sum(abs2, point .- observation)
            for observation in surrogate.points
        )
        surrogate.values[index]
    end
    return VerdictPosterior(means, fill(0.05, length(points)))
end

@testset "Verdict interop and performance" begin

@testset "OptimizationBase semantics" begin
    no_initial_point = OptimizationBase.OptimizationProblem(
        (point, parameters) -> sum(abs2, point),
        nothing;
        lb=fill(-1.0, 2),
        ub=fill(1.0, 2),
    )
    no_initial_solution = OptimizationBase.solve(
        no_initial_point,
        SAMBO.SMBO(candidate_pool=32);
        maxiters=6,
        rng=MersenneTwister(900),
    )
    @test no_initial_solution.stats.fevals == 6

    maximization = OptimizationBase.OptimizationProblem(
        (point, parameters) -> -(point[1] - 0.75)^2,
        nothing;
        lb=[0.0],
        ub=[1.0],
        sense=OptimizationBase.MaxSense,
    )
    maximized = OptimizationBase.solve(
        maximization,
        SAMBO.SMBO(candidate_pool=32);
        maxiters=10,
        rng=MersenneTwister(901),
    )
    @test maximized.original.sense isa SAMBO.Maximize
    @test maximized.objective > -0.1

    constrained_function = OptimizationBase.OptimizationFunction(
        (point, parameters) -> sum(abs2, point);
        cons=(residual, point, parameters) -> begin
            residual[1] = point[1] + point[2]
            residual
        end,
    )
    constrained_problem = OptimizationBase.OptimizationProblem(
        constrained_function,
        nothing;
        lb=fill(-1.0, 2),
        ub=fill(1.0, 2),
        lcons=[0.5],
        ucons=[Inf],
    )
    constrained = nothing
    try
        constrained = OptimizationBase.solve(
            constrained_problem,
            SAMBO.SCEUA();
            maxiters=30,
            rng=MersenneTwister(902),
        )
    catch
        @test false
    end
    isnothing(constrained) || @test sum(constrained.u) >= 0.5

    out_of_place_function = OptimizationBase.OptimizationFunction{false}(
        (point, parameters) -> sum(abs2, point);
        cons=(point, parameters) -> [point[1] + point[2]],
    )
    out_of_place_problem = OptimizationBase.OptimizationProblem(
        out_of_place_function,
        nothing;
        lb=fill(-1.0, 2),
        ub=fill(1.0, 2),
        lcons=[0.5],
        ucons=[Inf],
    )
    out_of_place = OptimizationBase.solve(
        out_of_place_problem,
        SAMBO.SCEUA();
        maxiters=30,
        rng=MersenneTwister(907),
    )
    @test sum(out_of_place.u) >= 0.5

    limited = OptimizationBase.solve(
        no_initial_point,
        SAMBO.SMBO(candidate_pool=16);
        maximum_evaluations=2,
        rng=MersenneTwister(903),
    )
    @test limited.original.retcode == :evaluation_limit
    @test limited.retcode == OptimizationBase.ReturnCode.MaxIters

    timed = OptimizationBase.solve(
        no_initial_point,
        SAMBO.SMBO(candidate_pool=16);
        maximum_evaluations=10,
        time_limit=eps(),
        rng=MersenneTwister(904),
    )
    @test timed.original.retcode == :time_limit
    @test timed.retcode == OptimizationBase.ReturnCode.MaxTime

    stalled = OptimizationBase.solve(
        OptimizationBase.OptimizationProblem(
            (point, parameters) -> 1.0,
            nothing;
            lb=[0.0],
            ub=[1.0],
        ),
        SAMBO.SMBO(candidate_pool=16);
        maximum_evaluations=10,
        stall_evaluations=1,
        rng=MersenneTwister(905),
    )
    @test stalled.original.retcode == :stalled
    @test stalled.retcode == OptimizationBase.ReturnCode.Stalled

    callback_states = Any[]
    callback = OptimizationBase.solve(
        no_initial_point,
        SAMBO.SMBO(candidate_pool=16);
        maximum_evaluations=10,
        callback=(state, value) -> begin
            push!(callback_states, (state, value))
            true
        end,
        rng=MersenneTwister(906),
    )
    @test callback.original.retcode == :callback_stop
    @test callback.retcode == OptimizationBase.ReturnCode.Terminated
    @test length(callback_states) == 1
    @test callback_states[1][1] isa OptimizationBase.OptimizationState
    @test callback_states[1][1].objective == callback_states[1][2]
end

@testset "MLJTuning lifecycle and orientation" begin
    model = VerdictLinearModel(0.5)
    parameter_range = range(
        model,
        :slope;
        lower=0.0,
        upper=2.0,
    )
    tuning = SAMBO.SAMBOTuning(
        algorithm=SAMBO.SMBO(
            initial_points=2,
            candidate_pool=32,
            batch_size=2,
        ),
        rng=MersenneTwister(910),
    )
    state = MLJTuning.setup(tuning, model, parameter_range, 6, 0)
    first_models, state =
        MLJTuning.models(tuning, model, NamedTuple[], state, 6, 0)
    @test length(first_models) == 2
    pending = state.pending
    repeated_models, state =
        MLJTuning.models(tuning, model, NamedTuple[], state, 4, 0)
    @test isempty(repeated_models)
    @test state.pending === pending
    first_measurements = [0.8, 0.2]
    history = [
        (
            measurement=[first_measurements[index]],
            measure=[MLJBase.rms],
        )
        for index in eachindex(first_models)
    ]
    second_models, state =
        MLJTuning.models(tuning, model, history, state, 4, 0)
    @test length(second_models) == 2
    @test SAMBO.objectivevalues(SAMBO.trace(state.solver)) ==
        first_measurements
    @test state.history_cursor == 2

    zero_state = MLJTuning.setup(tuning, model, parameter_range, 1, 0)
    zero_models, zero_state =
        MLJTuning.models(tuning, model, NamedTuple[], zero_state, 0, 0)
    @test isempty(zero_models)
    @test isnothing(zero_state.pending)

    logarithmic_range = range(
        model,
        :slope;
        lower=0.01,
        upper=100.0,
        scale=:log10,
    )
    logarithmic_state =
        MLJTuning.setup(tuning, model, logarithmic_range, 4, 0)
    logarithmic_models, logarithmic_state = MLJTuning.models(
        tuning,
        model,
        NamedTuple[],
        logarithmic_state,
        4,
        0,
    )
    @test all(candidate -> 0.01 <= candidate.slope <= 100.0, logarithmic_models)
    logarithmic_dimension =
        logarithmic_state.solver.core.problem.space.dimensions[1]
    @test logarithmic_dimension.lower ≈ -2
    @test logarithmic_dimension.upper ≈ 2

    score_tuning = SAMBO.SAMBOTuning(
        algorithm=SAMBO.SMBO(
            initial_points=1,
            candidate_pool=16,
            batch_size=1,
        ),
        rng=MersenneTwister(911),
    )
    score_state =
        MLJTuning.setup(score_tuning, model, parameter_range, 3, 0)
    score_models, score_state = MLJTuning.models(
        score_tuning,
        model,
        NamedTuple[],
        score_state,
        3,
        0,
    )
    score_history = [(
        measurement=[0.9, 0.1],
        measure=[MLJBase.rsquared, MLJBase.rms],
    )]
    _, score_state = MLJTuning.models(
        score_tuning,
        model,
        score_history,
        score_state,
        2,
        0,
    )
    @test only(SAMBO.objectivevalues(SAMBO.trace(score_state.solver))) == -0.9

    X = (x=collect(range(-1.0, 1.0; length=30)),)
    y = copy(X.x)
    for measure in (MLJBase.rms, MLJBase.rsquared)
        tuned_model = MLJTuning.TunedModel(
            model=VerdictLinearModel(0.5),
            tuning=SAMBO.SAMBOTuning(
                algorithm=SAMBO.SMBO(
                    initial_points=2,
                    candidate_pool=32,
                ),
                rng=MersenneTwister(912),
            ),
            range=parameter_range,
            measure=measure,
            n=6,
            resampling=MLJBase.Holdout(fraction_train=0.7),
        )
        machine = MLJBase.machine(tuned_model, X, y)
        MLJBase.fit!(machine; verbosity=0)
        report = MLJBase.report(machine)
        @test length(report.history) == 6
        @test report.best_history_entry.measure[1] == measure
    end
end

@testset "SurrogatesBase stochastic model" begin
    result = SAMBO.minimize(
        point -> sum(abs2, point),
        SAMBO.Box([-1.0], [1.0]);
        algorithm=SAMBO.SMBO(
            surrogate=VerdictStochasticSurrogate(),
            acquisition=SAMBO.LowerConfidenceBound(),
            initial_points=2,
            candidate_pool=16,
        ),
        maximum_evaluations=5,
        rng=MersenneTwister(920),
    )
    @test SAMBO.evaluation_count(result) == 5
    @test isfinite(SAMBO.minimum(result))
end

@testset "Threaded batches and exceptions" begin
    candidates = [
        0.0 0.25 0.5 0.75
        1.0 0.75 0.5 0.25
    ]
    values = zeros(4)
    SAMBO.evaluate!(
        values,
        SAMBO.Threaded(),
        SAMBO.Problem(x -> sum(abs2, x), SAMBO.Box(zeros(2), ones(2))),
        candidates,
        SAMBO.ErrorOnNonfinite(),
    )
    @test values == vec(sum(abs2, candidates; dims=1))

    throwing = SAMBO.Problem(
        x -> x[1] == 0.5 ? error("threaded objective failure") : sum(abs2, x),
        SAMBO.Box(zeros(2), ones(2)),
    )
    @test_throws Exception SAMBO.evaluate!(
        values,
        SAMBO.Threaded(),
        throwing,
        candidates,
        SAMBO.ErrorOnNonfinite(),
    )

    batched = SAMBO.solve(
        SAMBO.Problem(x -> sum(abs2, x), SAMBO.Box(zeros(2), ones(2))),
        SAMBO.SMBO(
            initial_points=4,
            batch_size=4,
            candidate_pool=32,
        );
        maximum_evaluations=8,
        executor=SAMBO.Threaded(),
        rng=MersenneTwister(930),
    )
    @test SAMBO.evaluation_count(batched) == 8
end

@testset "Warm allocation coverage" begin
    rng = MersenneTwister(940)
    points = rand(rng, 2, 20)
    values = vec(sum(abs2, points; dims=1))
    model = SAMBO.fitmodel(SAMBO.GaussianProcessSurrogate(), points, values)
    queries = rand(rng, 2, 64)
    means = zeros(64)
    variances = zeros(64)
    prediction_workspace = SAMBO.GPPredictionWorkspace(Float64)
    SAMBO.predictmeanvariance!(
        means,
        variances,
        model,
        queries,
        prediction_workspace,
    )
    @test @allocated(
        SAMBO.predictmeanvariance!(
            means,
            variances,
            model,
            queries,
            prediction_workspace,
        ),
    ) <= 256

    sceua = SAMBO.init(
        SAMBO.Problem(
            x -> sum(abs2, x),
            SAMBO.Box(fill(-1.0, 2), fill(1.0, 2)),
        ),
        SAMBO.SCEUA();
        maximum_evaluations=100,
        rng=MersenneTwister(941),
    )
    SAMBO.step!(sceua)
    SAMBO.step!(sceua)
    @test @allocated(SAMBO.step!(sceua)) <= 2_048

    smbo = SAMBO.init(
        SAMBO.Problem(
            x -> sum(abs2, x),
            SAMBO.Box(fill(-1.0, 2), fill(1.0, 2)),
        ),
        SAMBO.SMBO(initial_points=4, candidate_pool=64);
        maximum_evaluations=20,
        rng=MersenneTwister(942),
    )
    initial = SAMBO.ask!(smbo, 4)
    SAMBO.tell!(smbo, initial, [sum(abs2, point) for point in initial])
    pool = @view smbo.workspace.candidates[:, 1:64]
    SAMBO.generate_candidates!(
        pool,
        smbo,
        smbo.algorithm.candidate_sampler,
    )
    @test @allocated(
        SAMBO.generate_candidates!(
            pool,
            smbo,
            smbo.algorithm.candidate_sampler,
        ),
    ) <= 65_536
    batch = SAMBO.ask!(smbo, 1)
    SAMBO.cancel!(smbo, batch)
    @test @allocated(SAMBO.ask!(smbo, 1)) <= 150_000

    shgo = SAMBO.init(
        SAMBO.Problem(
            x -> sum(abs2, x),
            SAMBO.Box(fill(-1.0, 2), fill(1.0, 2)),
        ),
        SAMBO.SHGO(sampling_points=16);
        maximum_evaluations=100,
        rng=MersenneTwister(943),
    )
    SAMBO.step!(shgo)
    shgo_bytes = @allocated(SAMBO.step!(shgo))
    @test shgo_bytes <= 2_000_000

    pd_result = SAMBO.minimize(
        x -> sum(abs2, x),
        SAMBO.Box(fill(-1.0, 2), fill(1.0, 2));
        algorithm=SAMBO.SMBO(candidate_pool=32),
        maximum_evaluations=8,
        rng=MersenneTwister(944),
    )
    pd_workspace =
        SAMBO.PartialDependenceWorkspace(Float64, Float64, 32, 8)
    SAMBO.partialdependence(
        pd_result;
        dimensions=(1, 2),
        resolution=6,
        samples=8,
        workspace=pd_workspace,
        rng=MersenneTwister(945),
    )
    pd_bytes = @allocated SAMBO.partialdependence(
        pd_result;
        dimensions=(1, 2),
        resolution=6,
        samples=8,
        workspace=pd_workspace,
        rng=MersenneTwister(945),
    )
    @test pd_bytes <= 250_000
end

@testset "CI and correctness entry points" begin
    root = normpath(joinpath(@__DIR__, ".."))
    workflow = read(joinpath(root, ".github", "workflows", "CI.yml"), String)
    @test occursin("JULIA_NUM_THREADS: 4", workflow)
    @test occursin("Core without weak dependencies", workflow)
    @test occursin("Cross-runtime solution-quality checks", workflow)
    @test occursin("--profile native-default-v1", workflow)
    @test occursin("--profile exact-v1", workflow)
    @test occursin("requirements-correctness.txt", workflow)
    @test occursin("Native inference checks", workflow)
    @test occursin("Run doctests", workflow)
    @test occursin("Extension -", workflow)

    inference = read(joinpath(@__DIR__, "inference.jl"), String)
    for native_path in (
        "SAMBO.evaluate!",
        "SAMBO.predictmeanvariance!",
        "step!(sceua)",
        "ask!(smbo, 1)",
    )
        @test occursin(native_path, inference)
    end

    for script in (
        "correctness_julia.jl",
        "correctness_python.py",
        "compare_correctness.py",
        "python_reference.py",
    )
        path = joinpath(root, "benchmark", script)
        @test filesize(path) > 1_000
        @test occursin("main", read(path, String))
    end
end

end
