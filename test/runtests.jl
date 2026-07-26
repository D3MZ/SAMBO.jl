using SAMBO
using Test
using Random
using Tables

@testset "SAMBO" begin
    include("spaces.jl")
    include("surrogates.jl")
    include("algorithms.jl")
    include("advanced.jl")
    include("diagnostics.jl")
    include("interoperability.jl")
    include("plotting.jl")
    include("quality.jl")
end
