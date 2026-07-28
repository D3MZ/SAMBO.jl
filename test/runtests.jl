using SAMBO
using Test
using Random
using Tables

Base.include(
    SAMBO,
    joinpath(@__DIR__, "..", "benchmark", "python_sambo_profile.jl"),
)

@testset "SAMBO" begin
    include("spaces.jl")
    include("surrogates.jl")
    include("algorithms.jl")
    include("advanced.jl")
    include("diagnostics.jl")
    include("allocations.jl")
    include("corrected_allocation_review.jl")
    include("coverage_diag_eval.jl")
    include("coverage_sce_shgo.jl")
    include("shgo_python_profile.jl")
    include("coverage_smbo_gp.jl")
    include("interoperability.jl")
    include("plotting.jl")
    include("quality.jl")
    include("verdict_semantics.jl")
    include("verdict_plotting.jl")
    include("verdict_interop_perf.jl")
    include("corrected_core_review.jl")
    include("corrected_native_contracts.jl")
    include("corrected_interop_contracts.jl")
    include("corrected_performance_review.jl")
    include("native_default_review.jl")
    include("smbo_python_alignment.jl")
end
