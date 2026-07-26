module SAMBOMLJTuningExt

using SAMBO
import MLJTuning
import Random

mutable struct _SAMBOTuning{A,R} <: MLJTuning.TuningStrategy
    algorithm::A
    rng::R
end

function SAMBO.SAMBOTuning(; algorithm=SAMBO.SMBO(), rng=Random.default_rng())
    algorithm isa SAMBO.SMBO ||
        throw(ArgumentError("SAMBOTuning currently requires an SMBO algorithm"))
    resolved_rng = rng isa Integer ? Random.MersenneTwister(rng) : rng
    return _SAMBOTuning(algorithm, resolved_rng)
end

mutable struct TuningState{S,F}
    solver::S
    fields::F
    pending::Any
end

function _dimension(range::MLJTuning.MLJBase.NumericRange{T}) where {T}
    isfinite(range.lower) && isfinite(range.upper) ||
        throw(ArgumentError("SAMBOTuning requires bounded numeric ranges"))
    if T <: Integer
        return round(Int, range.lower) : round(Int, range.upper)
    end
    return SAMBO.Continuous(range.lower, range.upper)
end
_dimension(range::MLJTuning.MLJBase.NominalRange) = SAMBO.Choices(range.values)

function MLJTuning.setup(tuning::_SAMBOTuning, model, user_range, n, verbosity)
    ranges = user_range isa AbstractVector ? collect(user_range) : [user_range]
    fields = map(range -> range.field, ranges)
    space = SAMBO.SearchSpace(tuple(map(_dimension, ranges)...))
    solver = SAMBO.init(
        SAMBO.Problem(space),
        tuning.algorithm;
        maximum_evaluations=n,
        rng=deepcopy(tuning.rng),
    )
    return TuningState(solver, fields, nothing)
end

function _complete_pending!(state::TuningState, history)
    isnothing(state.pending) && return state
    count = length(state.pending)
    length(history) >= count || return state
    values = [entry.measurement[1] for entry in @view history[end-count+1:end]]
    SAMBO.tell!(state.solver, state.pending, values)
    state.pending = nothing
    return state
end

function MLJTuning.models(
    tuning::_SAMBOTuning,
    model,
    history,
    state::TuningState,
    n_remaining,
    verbosity,
)
    _complete_pending!(state, history)
    count = min(n_remaining, tuning.algorithm.batch_size)
    batch = SAMBO.ask!(state.solver, count)
    state.pending = batch
    models = map(batch) do point
        clone = deepcopy(model)
        for (index, field) in pairs(state.fields)
            MLJTuning.MLJBase.recursive_setproperty!(clone, field, point[index])
        end
        clone
    end
    return models, state
end

MLJTuning.default_n(::_SAMBOTuning, range) = 50

end
