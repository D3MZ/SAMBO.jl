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

    smbo = init(
        Problem(x -> sum(abs2, x), Box(fill(-1.0, 2), fill(1.0, 2))),
        SMBO(initial_points=4, candidate_pool=64);
        maximum_evaluations=20,
        rng=MersenneTwister(422),
    )
    initial = ask!(smbo, 4)
    tell!(smbo, initial, [sum(abs2, point) for point in initial])
    pool = @view smbo.workspace.candidates[:, 1:64]
    generate_candidates!(pool, smbo, smbo.algorithm.candidate_sampler)
    @test @allocated(
        generate_candidates!(pool, smbo, smbo.algorithm.candidate_sampler),
    ) <= 65_536
    @test @allocated(SAMBO._select_candidates(smbo, pool, 1)) <= 150_000

    shgo = init(
        Problem(x -> sum(abs2, x), Box(fill(-1.0, 2), fill(1.0, 2))),
        SHGO(sampling_points=16);
        maximum_evaluations=100,
        rng=MersenneTwister(423),
    )
    step!(shgo)
    @test @allocated(step!(shgo)) <= 2_000_000

    partial_result = minimize(
        x -> sum(abs2, x),
        Box(fill(-1.0, 2), fill(1.0, 2));
        algorithm=SMBO(candidate_pool=32),
        maximum_evaluations=8,
        rng=MersenneTwister(424),
    )
    partial_workspace =
        PartialDependenceWorkspace(Float64, Float64, 32, 8)
    partialdependence(
        partial_result;
        dimensions=(1, 2),
        resolution=6,
        samples=8,
        workspace=partial_workspace,
        rng=MersenneTwister(425),
    )
    @test @allocated(
        partialdependence(
            partial_result;
            dimensions=(1, 2),
            resolution=6,
            samples=8,
            workspace=partial_workspace,
            rng=MersenneTwister(425),
        ),
    ) <= 250_000
end
