rosenbrock(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2

@testset "algorithms" begin
    for algorithm in (SCEUA(), SMBO(candidate_pool=256), SHGO(samples=20))
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
    @test SCEUA(reflection=1.0f0).reflection isa Float32
    @test_throws ArgumentError init(problem, SMBO(candidate_pool=0))
    @test_throws ArgumentError init(problem, SHGO(local_starts=0))

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
        @test events[end] isa Sambo.ProgressEvent
        @test all(event -> isfinite(event.best_value), events)

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
        @test evaluation_count(result(state)) == 4
        @test retcode(result(state)) == :evaluation_limit
    end

    @test_throws ArgumentError init(
        Problem(_ -> 0.0, SearchSpace(x=Choices(:a, :b))),
        SHGO(),
    )
end
