using SAMBO
using MLJTuning
using Random
using Test

mutable struct TuningDummy
    rate::Float64
    mode::Symbol
end

@testset "MLJTuning extension" begin
    tuning = SAMBO.SAMBOTuning(
        algorithm=SAMBO.SMBO(candidate_pool=32, batch_size=1),
        rng=MersenneTwister(13),
    )
    ranges = [
        range(Float64, :rate; lower=0.1, upper=1.0),
        range(Symbol, :mode; values=[:fast, :accurate]),
    ]
    model = TuningDummy(0.5, :fast)
    state = MLJTuning.setup(tuning, model, ranges, 3, 0)
    first_models, state = MLJTuning.models(tuning, model, NamedTuple[], state, 3, 0)
    @test length(first_models) == 1
    history = [(measurement=[first_models[1].rate],)]
    second_models, state = MLJTuning.models(tuning, model, history, state, 2, 0)
    @test length(second_models) == 1
    push!(history, (measurement=[second_models[1].rate],))
    third_models, state = MLJTuning.models(tuning, model, history, state, 1, 0)
    @test length(third_models) == 1
end
