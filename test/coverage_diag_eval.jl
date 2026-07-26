using SAMBO
using Random
using Test

@testset "diagnostics, evaluation, and space coverage" begin
    space = SearchSpace(
        x=Continuous(0.0, 1.0),
        y=Continuous(0.0, 1.0),
    )
    constrained = minimize(
        point -> point.x^2 + point.y^2,
        space;
        constraint=point -> point.y - 0.5,
        algorithm=SMBO(candidate_pool=16),
        maximum_evaluations=6,
        rng=MersenneTwister(812),
    )

    workspace = PartialDependenceWorkspace(Float64, Float64, 1, 1)
    old_queries = workspace.queries
    background = [0.25 0.75; 0.25 0.75]
    dependence = partialdependence(
        constrained;
        dimensions=(1,),
        resolution=3,
        samples=2,
        background=background,
        workspace=workspace,
    )
    @test all(isfinite, dependence.values)
    @test workspace.queries !== old_queries
    @test size(workspace.queries) == (2, 2)
    @test size(workspace.feasible_queries) == (2, 2)

    box = Box([0.0, 0.0], [1.0, 1.0])
    points = [0.1 0.9; 0.2 0.8]
    latent = SAMBO._points_to_latent(box, points)
    @test latent == points
    @test latent !== points
    @test_throws DimensionMismatch SAMBO._points_to_latent(
        box,
        reshape([0.1, 0.2, 0.3], 1, 3),
    )

    iteration_limited = minimize(
        point -> sum(abs2, point),
        box;
        algorithm=SMBO(candidate_pool=16),
        maximum_evaluations=20,
        maximum_iterations=1,
        rng=MersenneTwister(813),
    )
    @test retcode(iteration_limited) == :iteration_limit

    @test SAMBO._encode_discrete((:low, :high), :high) == 1.0
    @test SAMBO._encode_discrete((:only,), :only) == 0.0
    @test_throws ArgumentError SAMBO._encode_discrete((:low, :high), :missing)
end
