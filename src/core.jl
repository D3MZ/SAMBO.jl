struct Unconstrained end
(::Unconstrained)(x) = true
struct Serial end
struct Threaded end

struct Problem{F,S,C,P}
    objective::F
    space::S
    constraint::C
    parameters::P
end
Problem(objective, space; constraint=Unconstrained(), parameters=nothing) =
    Problem(objective, space, constraint, parameters)
Problem(space; constraint=Unconstrained(), parameters=nothing) =
    Problem(nothing, space, constraint, parameters)

mutable struct Trace{T<:AbstractFloat}
    latent_points::Matrix{T}
    objective_values::Vector{T}
    feasible::BitVector
    iterations::Vector{Int}
    elapsed_seconds::Vector{Float64}
    count::Int
end
Trace{T}(d, n) where {T<:AbstractFloat} = Trace(
    Matrix{T}(undef, d, n), Vector{T}(undef, n), falses(n), zeros(Int, n), zeros(n), 0,
)
latentpoints(t::Trace) = @view t.latent_points[:, 1:t.count]
objectivevalues(t::Trace) = @view t.objective_values[1:t.count]

struct Result{S,X,T,TR,M,A,ST}
    space::S
    minimizer::X
    minimum::T
    trace::TR
    model::M
    algorithm::A
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

@inline function _checkedvalue(::Type{T}, value) where {T<:AbstractFloat}
    value isa Real || throw(ArgumentError("objective must return a real scalar"))
    isfinite(value) || throw(ArgumentError("objective must return a finite real scalar"))
    converted = T(value)
    isfinite(converted) || throw(ArgumentError("objective value is not representable as $T"))
    return converted
end

function _commit!(t::Trace{T}, z, value::T, iteration, started) where {T}
    index = t.count + 1
    if index > size(t.latent_points, 2)
        old_capacity = size(t.latent_points, 2)
        new_capacity = max(index, max(1, 2old_capacity))
        points = Matrix{T}(undef, size(t.latent_points, 1), new_capacity)
        points[:, 1:old_capacity] .= t.latent_points
        resize!(t.objective_values, new_capacity)
        resize!(t.feasible, new_capacity)
        resize!(t.iterations, new_capacity)
        resize!(t.elapsed_seconds, new_capacity)
        t.latent_points = points
    end
    length(z) == size(t.latent_points, 1) ||
        throw(DimensionMismatch("candidate and trace differ in dimension"))
    copyto!(@view(t.latent_points[:, index]), z)
    t.objective_values[index] = value
    t.feasible[index] = true
    t.iterations[index] = iteration
    t.elapsed_seconds[index] = time() - started
    t.count = index
    return t
end

function _evaluate(problem, z, ::Type{T}) where {T<:AbstractFloat}
    point = decode(problem.space, z)
    value = isnothing(problem.parameters) ?
        problem.objective(point) : problem.objective(point, problem.parameters)
    return _checkedvalue(T, value)
end

function minimize(objective, space; algorithm=SMBO(), constraint=Unconstrained(), parameters=nothing, kwargs...)
    solve(Problem(objective, space; constraint, parameters), algorithm; kwargs...)
end
function minimize(objective, space, algorithm; constraint=Unconstrained(), parameters=nothing, kwargs...)
    solve(Problem(objective, space; constraint, parameters), algorithm; kwargs...)
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
        evaluation=state,
        iteration=trace.iterations[state],
        objective=trace.objective_values[state],
        feasible=trace.feasible[state],
        elapsed_seconds=trace.elapsed_seconds[state],
    )
    return merge(metadata, coordinates), state + 1
end
