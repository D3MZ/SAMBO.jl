rosenbrock(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2

mutable struct AcceptThenRejectRepair
    remaining::Int
end
function SAMBO.repair!(
    rng,
    proposal,
    policy::AcceptThenRejectRepair,
    problem,
    centroid,
)
    policy.remaining == 0 && return false
    policy.remaining -= 1
    SAMBO.project!(proposal, problem.space)
    return true
end

@testset "algorithms" begin
    for algorithm in (
        SCEUA(),
        SMBO(candidate_pool=256),
        TopologicalMultistart(samples=20),
    )
        result = solve(
            Problem(rosenbrock, Box([-2.0, -1.0], [2.0, 3.0])),
            algorithm;
            maximum_evaluations=40,
            rng=MersenneTwister(4),
        )
        @test evaluation_count(result) == 40
        @test retcode(result) == :evaluation_limit
        @test isfinite(minimum(result))
        @test all(0 .<= latentpoints(trace(result)) .<= 1)
    end

    for algorithm in (
        SCEUA(),
        SMBO(candidate_pool=32),
        TopologicalMultistart(
            samples=6,
            topology=KNearestTopology(neighbors=2),
        ),
    )
        typed = solve(
            Problem(x -> sum(abs2, x), Box(BigFloat[-1, -1], BigFloat[1, 1])),
            algorithm;
            objective_type=BigFloat,
            maximum_evaluations=8,
            rng=MersenneTwister(41),
        )
        @test eltype(latentpoints(trace(typed))) == BigFloat
        @test eltype(objectivevalues(trace(typed))) == BigFloat
    end

    problem = Problem(SearchSpace(x=Continuous(-1.0, 1.0)))
    state = init(problem, SMBO(); maximum_evaluations=4, rng=MersenneTwister(1))
    batch = ask!(state, 2)
    original = copy(latentpoints(batch))
    latentpoints(batch)[1, 1] = 0.0
    @test_throws ArgumentError tell!(state, batch, [x.x^2 for x in batch])
    latentpoints(batch) .= original
    tell!(state, batch, [x.x^2 for x in batch])
    @test evaluation_count(result(state)) == 2
    @test_throws ArgumentError tell!(state, batch, [1.0, 2.0])

    second_batch = ask!(state, 2)
    @test_throws DimensionMismatch tell!(state, second_batch, [1.0])
    @test_throws ArgumentError tell!(state, second_batch, [1.0, NaN])
    tell!(state, second_batch, [x.x^2 for x in second_batch])
    @test retcode(result(state)) == :evaluation_limit

    constrained = minimize(
        x -> x.x,
        SearchSpace(x=Continuous(0.0, 1.0));
        constraint=x -> x.x >= 0.8,
        algorithm=SMBO(candidate_pool=16),
        maximum_evaluations=4,
        rng=MersenneTwister(2),
    )
    @test minimizer(constrained).x >= 0.8

    parameterized = minimize(
        (x, target) -> abs2(x.x - target),
        SearchSpace(x=Continuous(0.0, 1.0));
        parameters=0.5,
        maximum_evaluations=3,
        rng=MersenneTwister(3),
    )
    @test isfinite(minimum(parameterized))

    invalid_objective = Problem(_ -> NaN, Box([0.0], [1.0]))
    @test_throws ArgumentError solve(invalid_objective, SMBO(); maximum_evaluations=1)
    @test_throws MethodError init(problem, SMBO(); misspelled_option=true)
    @test SCEUA(reflection=big"1.0").reflection isa BigFloat
    @test_throws ArgumentError init(problem, SMBO(candidate_pool=0))
    @test_throws ArgumentError init(problem, TopologicalMultistart(local_starts=0))
    @test_throws ArgumentError SCEUA(complex_size=1)

    @testset "SCE-UA kernels and repair" begin
        @test SAMBO._complex_members(12, 3, 2) == [2, 5, 8, 11]
        population = reshape(collect(1.0:12.0), 2, 6)
        centroid = zeros(2)
        members = [2, 4, 6]
        SAMBO._complex_centroid!(centroid, population, members, 1)
        @test centroid == (
            population[:, 1] + population[:, 2] + population[:, 4]
        ) / 3
        reflected = zeros(2)
        contracted = zeros(2)
        SAMBO._reflection!(reflected, centroid, population[:, 6], 1.0)
        SAMBO._contraction!(contracted, centroid, population[:, 6], 0.5)
        @test reflected ≈ 2centroid - population[:, 6]
        @test contracted ≈ 0.5 .* (centroid + population[:, 6])

        constrained_problem = Problem(
            x -> (x[1] - 0.9)^2,
            Box([0.0], [1.0]);
            constraint=x -> x[1] >= 0.8,
        )
        infeasible = [-1.0]
        @test !SAMBO.repair!(
            MersenneTwister(1),
            infeasible,
            SAMBO.ProjectToBounds(),
            constrained_problem,
            [0.9],
        )
        moving = [-1.0]
        @test SAMBO.repair!(
            MersenneTwister(1),
            moving,
            SAMBO.MoveTowardCentroid(),
            constrained_problem,
            [0.9],
        )
        @test moving[1] >= 0.8

        repair_policy = AcceptThenRejectRepair(2)
        repaired = solve(
            Problem(_ -> 1.0, Box(fill(-1.0, 2), fill(1.0, 2))),
            SCEUA(
                complexes=2,
                complex_size=3,
                repair=repair_policy,
            );
            maximum_evaluations=9,
            rng=MersenneTwister(101),
        )
        evaluated = [
            Tuple(@view latentpoints(trace(repaired))[:, column])
            for column in axes(latentpoints(trace(repaired)), 2)
        ]
        @test evaluated[end] ∉ evaluated[1:end-1]
    end

    @testset "stopping, initialization, and parallel evaluation" begin
        events = Any[]
        callback_result = minimize(
            x -> x[1]^2,
            Box([-1.0], [1.0]);
            algorithm=SMBO(candidate_pool=32),
            maximum_evaluations=20,
            callback=event -> (push!(events, event); event.evaluation == 3),
            rng=MersenneTwister(12),
        )
        @test retcode(callback_result) == :callback_stop
        @test evaluation_count(callback_result) == 3
        @test length(events) == 3
        @test events[end] isa SAMBO.BatchProgressEvent
        @test all(event -> event.batch_size == 1, events)
        @test all(event -> isfinite(event.best_value), events)

        batch_events = Any[]
        batch_state = init(
            Problem(SearchSpace(x=Continuous(0.0, 1.0))),
            SMBO(initial_points=3);
            maximum_evaluations=3,
            callback=event -> (push!(batch_events, event); false),
            rng=MersenneTwister(120),
        )
        callback_batch = ask!(batch_state, 3)
        tell!(
            batch_state,
            callback_batch,
            [point.x^2 for point in callback_batch],
        )
        @test length(batch_events) == 1
        @test only(batch_events).batch_size == 3
        @test only(batch_events).evaluation == 3

        known = minimize(
            x -> abs2(x[1] - 0.25),
            Box([0.0], [1.0]);
            algorithm=SMBO(candidate_pool=32),
            initial_points=[[0.25]],
            initial_values=[0.0],
            maximum_evaluations=4,
            rng=MersenneTwister(7),
        )
        @test evaluation_count(known) == 4
        @test trace(known).count == 5
        @test minimum(known) == 0
        known_rows = collect(observations(known))
        @test known_rows[1].source == KnownObservation
        @test known_rows[1].evaluation == 0
        @test all(row -> row.source == InternalEvaluation, known_rows[2:end])
        @test getproperty.(known_rows[2:end], :evaluation) == 1:4

        snapshot_state = init(
            Problem(SearchSpace(x=Continuous(0.0, 1.0))),
            SMBO(initial_points=2);
            maximum_evaluations=2,
            rng=MersenneTwister(71),
        )
        snapshot_batch = ask!(snapshot_state, 1)
        tell!(snapshot_state, snapshot_batch, [1.0])
        frozen = result(snapshot_state)
        next_batch = ask!(snapshot_state, 1)
        tell!(snapshot_state, next_batch, [0.0])
        @test trace(frozen).count == 1
        @test minimum(frozen) == 1.0
        @test trace(snapshot_state).count == 2
        @test minimum(result(snapshot_state)) == 0.0

        for (TX, TY) in ((Float64, Float64), (BigFloat, BigFloat))
            typed_state = init(
                Problem(Box(TX[-1], TX[1])),
                SMBO();
                objective_type=TY,
                initial_points=[TX[0]],
                initial_values=[TY(1.25)],
                maximum_evaluations=1,
                rng=MersenneTwister(72),
            )
            typed = result(typed_state)
            @test eltype(latentpoints(trace(typed))) == TX
            @test eltype(objectivevalues(trace(typed))) == TY
            @test minimum(typed) == TY(1.25)
        end

        serial = solve(
            Problem(x -> sum(abs2, x), Box(fill(-1.0, 3), fill(1.0, 3))),
            SCEUA();
            maximum_evaluations=24,
            rng=MersenneTwister(22),
            executor=Serial(),
        )
        threaded = solve(
            Problem(x -> sum(abs2, x), Box(fill(-1.0, 3), fill(1.0, 3))),
            SCEUA();
            maximum_evaluations=24,
            rng=MersenneTwister(22),
            executor=Threaded(),
        )
        @test latentpoints(trace(serial)) == latentpoints(trace(threaded))
        @test objectivevalues(trace(serial)) == objectivevalues(trace(threaded))
    end

    @testset "out-of-order ask/tell" begin
        state = init(
            Problem(SearchSpace(x=Continuous(0.0, 1.0))),
            SMBO(initial_points=2, candidate_pool=32);
            maximum_evaluations=4,
            rng=MersenneTwister(9),
        )
        first_batch = ask!(state, 2)
        second_batch = ask!(state, 2)
        tell!(state, second_batch, [point.x^2 for point in second_batch])
        tell!(state, first_batch, [point.x^2 for point in first_batch])
        if evaluation_count(result(state)) < 4
            final_batch = ask!(state, 4 - evaluation_count(result(state)))
            tell!(state, final_batch, [point.x^2 for point in final_batch])
        end
        @test evaluation_count(result(state)) == 4
        @test retcode(result(state)) == :evaluation_limit
    end

    @test_throws ArgumentError init(
        Problem(_ -> 0.0, SearchSpace(x=Choices(:a, :b))),
        TopologicalMultistart(),
    )
end
