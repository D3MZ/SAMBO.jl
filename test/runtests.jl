using Sambo
using Test
using Random
using Tables

@testset "Sambo" begin
    include("spaces.jl")
    include("algorithms.jl")
    include("diagnostics.jl")
    include("interoperability.jl")
    include("plotting.jl")
    include("quality.jl")
end
