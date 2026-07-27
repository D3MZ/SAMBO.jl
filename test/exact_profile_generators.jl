module ExactProfileGeneratorTests

using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "benchmark", "correctness_julia.jl"))

@testset "Exact-v1 fixture consumer" begin
    fixture = exact_fixture()
    @test length(fixture) == 45
    sce = fixture[("hartmann6", "SCE-UA", 1)]
    @test Set(getproperty.(sce, :phase)) ==
        Set(("initial_population", "replacement_sample"))
    smbo = fixture[("hartmann6", "SMBO", 1)]
    pools = filter(item -> item.phase == "candidate_pool", smbo)
    @test all(item -> item.pool_id > 0, pools)
    @test all(item -> item.acquisition_coefficient !== nothing, pools)
    @test all(
        left.pool_id != right.pool_id ||
        left.acquisition_coefficient == right.acquisition_coefficient
        for (left, right) in zip(pools, @view pools[2:end])
    )
    @test Set(
        key[3]
        for key in keys(fixture)
        if key[1:2] == ("hartmann6", "SHGO")
    ) == Set(1:5)
    @test startswith(fixture_hash(), "sha256:")
    @test requested_profile(["--profile", "exact-v1"]) == "exact-v1"
end

end
