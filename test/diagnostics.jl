@testset "diagnostics and tables" begin
    result = minimize(
        x -> sum(abs2, x),
        Box([-1.0, -1.0], [1.0, 1.0]);
        algorithm=SMBO(candidate_pool=128),
        maximum_evaluations=12,
        rng=MersenneTwister(2),
    )
    convergence = convergencedata(result)
    @test length(convergence.best) == 12
    @test issorted(convergence.best; rev=true)
    @test length(regretdata(result; optimum=0).regret) == 12
    @test size(partialdependence(result; dimensions=(1,), resolution=8).values) == (8,)
    @test size(partialdependence(result; dimensions=(1, 2), resolution=5).values) == (5, 5)
    @test length(Tables.rows(observations(result))) == 12
    @test first(observations(result)).evaluation == 1

    @test_throws ArgumentError evaluationsdata(result; dimensions=Int[])
    @test_throws ArgumentError partialdependence(result; dimensions=(1, 1))
    @test_throws ArgumentError partialdependence(result; dimensions=(3,))
    @test_throws ArgumentError partialdependence(result; resolution=1)

    empty_result = Sambo.result(init(Problem(Box([0.0], [1.0])), SMBO()))
    @test isempty(convergencedata(empty_result).best)
    @test_throws ArgumentError partialdependence(empty_result)
end
