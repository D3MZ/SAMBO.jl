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

mutable struct TuningState{S,F,R}
    solver::S
    fields::F
    ranges::R
    pending::Union{Nothing,SAMBO.CandidateBatch}
    history_cursor::Int
end

_measurement_loss(value, ::Val{:loss}) = value
_measurement_loss(value, ::Val{:score}) = -value
function _measurement_loss(value, orientation::Symbol)
    orientation in (:loss, :score) ||
        throw(ArgumentError("unsupported MLJ measure orientation: $orientation"))
    return _measurement_loss(value, Val(orientation))
end
function _measurement_loss(entry)
    value = entry.measurement[1]
    hasproperty(entry, :measure) || return value
    isempty(entry.measure) && return value
    return _measurement_loss(
        value,
        MLJTuning.MLJBase.orientation(entry.measure[1]),
    )
end

function _dimension(
    range::MLJTuning.MLJBase.NumericRange{T},
) where {T<:Integer}
    isfinite(range.lower) && isfinite(range.upper) ||
        throw(ArgumentError("SAMBOTuning requires bounded numeric ranges"))
    range.scale === :linear &&
        return round(Int, range.lower):round(Int, range.upper)
    scale = MLJTuning.MLJBase.scale(range.scale)
    return SAMBO.Continuous(
        MLJTuning.MLJBase.transform(
            MLJTuning.MLJBase.Scale,
            scale,
            range.lower,
        ),
        MLJTuning.MLJBase.transform(
            MLJTuning.MLJBase.Scale,
            scale,
            range.upper,
        ),
    )
end
function _dimension(
    range::MLJTuning.MLJBase.NumericRange{T},
) where {T<:AbstractFloat}
    isfinite(range.lower) && isfinite(range.upper) ||
        throw(ArgumentError("SAMBOTuning requires bounded numeric ranges"))
    scale = MLJTuning.MLJBase.scale(range.scale)
    return SAMBO.Continuous(
        MLJTuning.MLJBase.transform(
            MLJTuning.MLJBase.Scale,
            scale,
            range.lower,
        ),
        MLJTuning.MLJBase.transform(
            MLJTuning.MLJBase.Scale,
            scale,
            range.upper,
        ),
    )
end
_dimension(range::MLJTuning.MLJBase.NominalRange) = SAMBO.Choices(range.values)

_parameter_value(range::MLJTuning.MLJBase.NominalRange, value) = value
function _parameter_value(
    range::MLJTuning.MLJBase.NumericRange{T},
    value,
) where {T}
    decoded = MLJTuning.MLJBase.inverse_transform(
        MLJTuning.MLJBase.Scale,
        MLJTuning.MLJBase.scale(range.scale),
        value,
    )
    return T <: Integer ? round(T, decoded) : T(decoded)
end

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
    return TuningState(solver, fields, ranges, nothing, 0)
end

function _complete_pending!(state::TuningState, history)
    isnothing(state.pending) && return state
    count = length(state.pending)
    last = state.history_cursor + count
    length(history) >= last || return state
    values = [
        _measurement_loss(history[index])
        for index in state.history_cursor+1:last
    ]
    SAMBO.tell!(state.solver, state.pending, values)
    state.pending = nothing
    state.history_cursor = last
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
    isnothing(state.pending) || return typeof(model)[], state
    n_remaining > 0 || return typeof(model)[], state
    count = min(n_remaining, tuning.algorithm.batch_size)
    batch = SAMBO.ask!(state.solver, count)
    isempty(batch) && return typeof(model)[], state
    state.pending = batch
    models = map(batch) do point
        clone = deepcopy(model)
        for (index, field) in pairs(state.fields)
            value = _parameter_value(state.ranges[index], point[index])
            MLJTuning.MLJBase.recursive_setproperty!(clone, field, value)
        end
        clone
    end
    return models, state
end

MLJTuning.default_n(::_SAMBOTuning, range) = 50

end
