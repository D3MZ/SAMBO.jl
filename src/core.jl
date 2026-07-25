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
Problem(objective, space; constraint=Unconstrained(), parameters=nothing) = Problem(objective, space, constraint, parameters)
Problem(space; constraint=Unconstrained(), parameters=nothing) = Problem(nothing, space, constraint, parameters)

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
    space::S; minimizer::X; minimum::T; trace::TR; model::M; algorithm::A
    retcode::Symbol; statistics::ST
end
minimizer(r::Result) = r.minimizer
minimum(r::Result) = r.minimum
trace(r::Result) = r.trace
retcode(r::Result) = r.retcode
evaluation_count(r::Result) = r.trace.count
iteration_count(r::Result) = get(r.statistics, :iterations, 0)

function _checkedvalue(::Type{T}, value) where {T<:AbstractFloat}
    value isa Real || throw(ArgumentError("objective must return a real scalar"))
    isfinite(value) || throw(ArgumentError("objective must return a finite real scalar"))
    converted = T(value)
    isfinite(converted) || throw(ArgumentError("objective value is not representable as $T"))
    return converted
end

function _commit!(t::Trace{T}, z, value::T, iteration, started) where {T}
    i = t.count + 1
    i <= size(t.latent_points, 2) || throw(BoundsError(t.objective_values, i))
    length(z) == size(t.latent_points, 1) || throw(DimensionMismatch("candidate and trace differ in dimension"))
    t.latent_points[:, i] .= z
    t.objective_values[i] = value
    t.feasible[i] = true
    t.iterations[i] = iteration
    t.elapsed_seconds[i] = time() - started
    t.count = i
    return t
end

function _evaluate(problem, z, ::Type{T}) where {T<:AbstractFloat}
    point = decode(problem.space, z)
    value = isnothing(problem.parameters) ? problem.objective(point) : problem.objective(point, problem.parameters)
    return _checkedvalue(T, value)
end

function minimize(objective, space; algorithm=SMBO(), constraint=Unconstrained(), parameters=nothing, kwargs...)
    solve(Problem(objective, space; constraint, parameters), algorithm; kwargs...)
end
function minimize(objective, space, algorithm; constraint=Unconstrained(), parameters=nothing, kwargs...)
    solve(Problem(objective, space; constraint, parameters), algorithm; kwargs...)
end

struct ObservationRows{R}; result::R; end
observations(r::Result) = ObservationRows(r)
Tables.istable(::Type{<:ObservationRows}) = true
Tables.rowaccess(::Type{<:ObservationRows}) = true
Tables.rows(x::ObservationRows) = x
Base.length(x::ObservationRows) = evaluation_count(x.result)
function Base.iterate(rows::ObservationRows, state=1)
    state > length(rows) && return nothing
    r = rows.result
    t = r.trace
    p = decode(r.space, @view t.latent_points[:, state])
    names = ntuple(i -> Symbol("x", i), dimension(r.space))
    coords = p isa NamedTuple ? p : NamedTuple{names}(Tuple(p))
    metadata = (
        evaluation=state,
        iteration=t.iterations[state],
        objective=t.objective_values[state],
        feasible=t.feasible[state],
        elapsed_seconds=t.elapsed_seconds[state],
    )
    return merge(metadata, coords), state + 1
end
