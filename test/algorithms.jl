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
end
