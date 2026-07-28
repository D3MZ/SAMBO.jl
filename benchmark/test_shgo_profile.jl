using Random
using SAMBO
using Test

Base.include(SAMBO, joinpath(@__DIR__, "python_sambo_profile.jl"))

function profile_result(seed; budget=100)
    return solve(
        Problem(x -> sum(abs2, x), Box(zeros(2), ones(2))),
        SHGO(
            SAMBO.PythonSAMBOProfile();
            sampling_points=12,
            local_starts=1,
            local_budget=8,
            convergence_tolerance=0.0,
            convergence_window=0,
        );
        maximum_evaluations=budget,
        rng=MersenneTwister(seed),
    )
end

@testset "Python SAMBO SHGO benchmark profile" begin
    profile = SHGO(SAMBO.PythonSAMBOProfile())
    @test profile.sampling isa SAMBO.ScrambledHaltonDesign
    @test profile.topology isa SAMBO.PythonIncrementalDelaunayTopology
    @test profile.sampling_points == 80
    @test profile.local_starts == 4
    @test profile.maximum_refinements == 1
    @test profile.minimum_local_reserve == 0
    @test !profile.divide_automatic_local_budget
    @test profile.convergence_tolerance == 1e-6
    @test profile.convergence_window == 30
    @test profile.local_solver isa SAMBO.QuasiNewtonSearch

    first = profile_result(39)
    repeated = profile_result(39)
    different = profile_result(40)
    @test first.statistics.refinements == 1
    @test retcode(first) == :success
    @test latentpoints(trace(first)) == latentpoints(trace(repeated))
    @test objectivevalues(trace(first)) == objectivevalues(trace(repeated))
    @test latentpoints(trace(first)) != latentpoints(trace(different))

    box = solve(
        Problem(x -> abs2(x[2] - 3), Box([2.0, -5.0], [2.0, 5.0])),
        SHGO();
        maximum_evaluations=40,
        rng=MersenneTwister(38),
    )
    search = solve(
        Problem(
            x -> abs2(x.y - 3),
            SearchSpace(x=Continuous(2.0, 2.0), y=Continuous(-5.0, 5.0)),
        ),
        SHGO();
        maximum_evaluations=40,
        rng=MersenneTwister(38),
    )
    @test bestpoint(box)[1] == bestpoint(search).x == 2
    @test bestpoint(box)[2] ≈ bestpoint(search).y atol=1e-10
    @test bestvalue(box) ≈ bestvalue(search) atol=1e-10

    constant = solve(
        Problem(_ -> 1.0, Box(zeros(5), ones(5))),
        profile;
        maximum_evaluations=1000,
        rng=MersenneTwister(41),
    )
    @test evaluation_count(constant) == 30
    @test retcode(constant) == :success

    limited = solve(
        Problem(_ -> 1.0, Box(zeros(5), ones(5))),
        profile;
        maximum_evaluations=30,
        rng=MersenneTwister(41),
    )
    @test evaluation_count(limited) == 30
    @test retcode(limited) == :evaluation_limit
end
