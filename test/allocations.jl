@testset "warm allocation regressions" begin
    rng = MersenneTwister(420)
    points = rand(rng, 2, 20)
    values = vec(sum(abs2, points; dims=1))
    model = SAMBO.fitmodel(GaussianProcessSurrogate(), points, values)
    queries = rand(rng, 2, 64)
    means = zeros(64)
    variances = zeros(64)
    workspace = GPPredictionWorkspace(Float64)
    SAMBO.predictmeanvariance!(
        means,
        variances,
        model,
        queries,
        workspace,
    )
    @test @allocated(
        SAMBO.predictmeanvariance!(
            means,
            variances,
            model,
            queries,
            workspace,
        ),
    ) <= 256

    state = init(
        Problem(x -> sum(abs2, x), Box(fill(-1.0, 2), fill(1.0, 2))),
        SCEUA();
        maximum_evaluations=100,
        rng=MersenneTwister(421),
    )
    step!(state)
    step!(state)
    @test @allocated(step!(state)) <= 2_048
end
