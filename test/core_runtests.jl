using SAMBO
using Test
using Random
using Tables

@testset "SAMBO core" begin
    include("spaces.jl")
    include("surrogates.jl")
    include("algorithms.jl")
    include("advanced.jl")
    include("diagnostics.jl")
    include("allocations.jl")
end
