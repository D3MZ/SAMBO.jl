struct Unconstrained end
(::Unconstrained)(x) = true
struct Serial end
struct Threaded end

abstract type OptimizationSense end
struct Minimize <: OptimizationSense end
struct Maximize <: OptimizationSense end

abstract type NonfinitePolicy end
struct ErrorOnNonfinite <: NonfinitePolicy end
struct PenalizeNonfinite <: NonfinitePolicy end
struct AllowInfinite <: NonfinitePolicy end
struct NumericalFailureError <: Exception
    message::String
end
Base.showerror(io::IO, error::NumericalFailureError) =
    print(io, error.message)

struct Problem{F,S,C,P,O<:OptimizationSense}
    objective::F
    space::S
    constraint::C
    parameters::P
    sense::O
end
Problem(
    objective,
    space;
    constraint=Unconstrained(),
    parameters=nothing,
    sense=Minimize(),
) = Problem(objective, space, constraint, parameters, sense)
Problem(space; constraint=Unconstrained(), parameters=nothing, sense=Minimize()) =
    Problem(nothing, space, constraint, parameters, sense)
Problem(objective, space, constraint, parameters) =
    Problem(objective, space, constraint, parameters, Minimize())

@inline _loss(::Minimize, value) = value
@inline _loss(::Maximize, value) = -value
@inline _loss(problem::Problem, value) = _loss(problem.sense, value)
@inline _isbetter(problem::Problem, left, right) =
    _loss(problem, left) < _loss(problem, right)
@inline _worst(::Type{T}, ::Minimize) where {T} = T(Inf)
@inline _worst(::Type{T}, ::Maximize) where {T} = T(-Inf)

@enum ObservationSource::UInt8 begin
    KnownObservation
    InternalEvaluation
    ExternalEvaluation
end

mutable struct Trace{TX<:AbstractFloat,TY<:AbstractFloat}
    latent_points::Matrix{TX}
    objective_values::Vector{TY}
    source::Vector{ObservationSource}
    evaluation_numbers::Vector{Int}
    iterations::Vector{Int}
    elapsed_seconds::Vector{Float64}
    count::Int
end
Trace{TX,TY}(d, n) where {TX<:AbstractFloat,TY<:AbstractFloat} = Trace(
    Matrix{TX}(undef, d, n),
    Vector{TY}(undef, n),
    Vector{ObservationSource}(undef, n),
    zeros(Int, n),
    zeros(Int, n),
    zeros(n),
    0,
)
Trace{T}(d, n) where {T<:AbstractFloat} = Trace{T,T}(d, n)
latentpoints(t::Trace) = @view t.latent_points[:, 1:t.count]
objectivevalues(t::Trace) = @view t.objective_values[1:t.count]
function _snapshot(t::Trace{TX,TY}) where {TX,TY}
    count = t.count
    return Trace{TX,TY}(
        Matrix(@view t.latent_points[:, 1:count]),
        collect(@view t.objective_values[1:count]),
        collect(@view t.source[1:count]),
        collect(@view t.evaluation_numbers[1:count]),
        collect(@view t.iterations[1:count]),
        collect(@view t.elapsed_seconds[1:count]),
        count,
    )
end

struct Result{P,S,X,T,TR,M,A,O,ST}
    problem::P
    space::S
    minimizer::X
    minimum::T
    trace::TR
    model::M
    algorithm::A
    sense::O
    retcode::Symbol
    statistics::ST
end
minimizer(r::Result) = r.minimizer
minimum(r::Result) = r.minimum
trace(r::Result) = r.trace
retcode(r::Result) = r.retcode
evaluation_count(r::Result) = get(r.statistics, :evaluations, r.trace.count)
iteration_count(r::Result) = get(r.statistics, :iterations, 0)
fittedmodel(r::Result) = r.model

struct ProgressEvent{T,X}
    evaluation::Int
    iteration::Int
    latest_value::T
    best_value::T
    latest_point::X
end

struct StopCriteria{T<:AbstractFloat}
    maximum_evaluations::Int
    maximum_iterations::Int
    absolute_tolerance::T
    relative_tolerance::T
    stall_evaluations::Int
    time_limit::Float64
end

@inline function _checkedvalue(
    ::Type{T},
    value,
    ::ErrorOnNonfinite,
    sense,
) where {T<:AbstractFloat}
    value isa Real || throw(ArgumentError("objective must return a real scalar"))
    isfinite(value) || throw(ArgumentError("objective must return a finite real scalar"))
    converted = T(value)
    isfinite(converted) || throw(ArgumentError("objective value is not representable as $T"))
    return converted
end
@inline function _checkedvalue(
    ::Type{T},
    value,
    ::PenalizeNonfinite,
    sense,
) where {T<:AbstractFloat}
    value isa Real || return _worst(T, sense)
    isfinite(value) || return _worst(T, sense)
    converted = T(value)
    return isfinite(converted) ? converted : _worst(T, sense)
end
@inline function _checkedvalue(
    ::Type{T},
    value,
    ::AllowInfinite,
    sense,
) where {T<:AbstractFloat}
    value isa Real || throw(ArgumentError("objective must return a real scalar"))
    isnan(value) && throw(ArgumentError("objective must not return NaN"))
    converted = T(value)
    isnan(converted) && throw(ArgumentError("objective value is not representable as $T"))
    return converted
end

function _commit!(
    t::Trace{TX,TY},
    z,
    value::TY,
    source::ObservationSource,
    evaluation_number,
    iteration,
    started,
) where {TX,TY}
    index = t.count + 1
    if index > size(t.latent_points, 2)
        old_capacity = size(t.latent_points, 2)
        new_capacity = max(index, max(1, 2old_capacity))
        points = Matrix{TX}(undef, size(t.latent_points, 1), new_capacity)
        points[:, 1:old_capacity] .= t.latent_points
        resize!(t.objective_values, new_capacity)
        resize!(t.source, new_capacity)
        resize!(t.evaluation_numbers, new_capacity)
        resize!(t.iterations, new_capacity)
        resize!(t.elapsed_seconds, new_capacity)
        t.latent_points = points
    end
    length(z) == size(t.latent_points, 1) ||
        throw(DimensionMismatch("candidate and trace differ in dimension"))
    copyto!(@view(t.latent_points[:, index]), z)
    t.objective_values[index] = value
    t.source[index] = source
    t.evaluation_numbers[index] = evaluation_number
    t.iterations[index] = iteration
    t.elapsed_seconds[index] = (time_ns() - started) / 1e9
    t.count = index
    return t
end

function _evaluate(problem, z, ::Type{T}, nonfinite) where {T<:AbstractFloat}
    point = decode(problem.space, z)
    value = isnothing(problem.parameters) ?
        problem.objective(point) : problem.objective(point, problem.parameters)
    return _checkedvalue(T, value, nonfinite, problem.sense)
end

function minimize(
    objective,
    space;
    algorithm=SMBO(),
    constraint=Unconstrained(),
    parameters=nothing,
    sense=Minimize(),
    kwargs...,
)
    solve(Problem(objective, space; constraint, parameters, sense), algorithm; kwargs...)
end
function minimize(
    objective,
    space,
    algorithm;
    constraint=Unconstrained(),
    parameters=nothing,
    sense=Minimize(),
    kwargs...,
)
    solve(Problem(objective, space; constraint, parameters, sense), algorithm; kwargs...)
end

struct ObservationRows{R}
    result::R
end
observations(r::Result) = ObservationRows(r)
Tables.istable(::Type{<:ObservationRows}) = true
Tables.rowaccess(::Type{<:ObservationRows}) = true
Tables.rows(x::ObservationRows) = x
Tables.schema(rows::ObservationRows) = isempty(rows) ? nothing :
    Tables.Schema(propertynames(first(rows)), map(typeof, values(first(rows))))
Base.length(x::ObservationRows) = x.result.trace.count
Base.eltype(::Type{ObservationRows{R}}) where {R} = Any
function Base.iterate(rows::ObservationRows, state=1)
    state > length(rows) && return nothing
    result = rows.result
    trace = result.trace
    point = decode(result.space, @view trace.latent_points[:, state])
    names = ntuple(i -> Symbol("x", i), dimension(result.space))
    coordinates = point isa NamedTuple ? point : NamedTuple{names}(Tuple(point))
    metadata = (
        evaluation=trace.evaluation_numbers[state],
        source=trace.source[state],
        iteration=trace.iterations[state],
        objective=trace.objective_values[state],
        elapsed_seconds=trace.elapsed_seconds[state],
    )
    return merge(metadata, coordinates), state + 1
end
