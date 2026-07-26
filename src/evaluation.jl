constraint_violation(::Unconstrained, point) = 0.0
constraint_violation(::Unconstrained, point, parameters) = 0.0
@inline violation(value::Bool) = ifelse(value, 0.0, 1.0)
@inline function violation(value::Real)
    return value
end
function constraint_violation(constraint, point)
    value = constraint(point)
    value isa Real ||
        throw(ArgumentError("constraint must return Bool or a real violation"))
    return violation(value)
end
function constraint_violation(problem::Problem, point)
    return constraint_violation(problem.constraint, point)
end
function isfeasible(problem::Problem, point)
    violation = constraint_violation(problem, point)
    return !isnan(violation) && violation <= 0
end

function evaluate!(values, ::Serial, problem::Problem, candidates, nonfinite)
    T = eltype(values)
    for column in axes(candidates, 2)
        values[column] = _evaluate(
            problem,
            @view(candidates[:, column]),
            T,
            nonfinite,
        )
    end
    return values
end

function evaluate!(values, ::Threaded, problem::Problem, candidates, nonfinite)
    T = eltype(values)
    Threads.@threads for column in axes(candidates, 2)
        values[column] = _evaluate(
            problem,
            @view(candidates[:, column]),
            T,
            nonfinite,
        )
    end
    return values
end

mutable struct SolverCore{P,R,E,C,TR,SC,N,T}
    problem::P
    rng::R
    executor::E
    callback::C
    trace::TR
    criteria::SC
    nonfinite::N
    iteration::Int
    started::UInt64
    retcode::Symbol
    best_value::T
    best_index::Int
    last_significant_improvement_evaluation::Int
    evaluations::Int
end

function _makecore(
    problem;
    maximum_evaluations=100,
    maximum_iterations=typemax(Int),
    absolute_tolerance=0.0,
    relative_tolerance=0.0,
    stall_evaluations=0,
    time_limit=Inf,
    rng=Random.default_rng(),
    executor=Serial(),
    callback=Returns(false),
    objective_type=Float64,
    nonfinite=ErrorOnNonfinite(),
)
    maximum_evaluations > 0 || throw(ArgumentError("maximum_evaluations must be positive"))
    maximum_iterations > 0 || throw(ArgumentError("maximum_iterations must be positive"))
    stall_evaluations >= 0 || throw(ArgumentError("stall_evaluations must be nonnegative"))
    time_limit > 0 || throw(ArgumentError("time_limit must be positive"))
    TX = latenttype(problem.space)
    TY = float(objective_type)
    TY <: AbstractFloat || throw(ArgumentError("objective_type must be an AbstractFloat type"))
    criteria = StopCriteria{TY}(
        maximum_evaluations,
        maximum_iterations,
        TY(absolute_tolerance),
        TY(relative_tolerance),
        stall_evaluations,
        Float64(time_limit),
    )
    return SolverCore(
        problem,
        rng,
        executor,
        callback,
        Trace{TX,TY}(dimension(problem.space), maximum_evaluations),
        criteria,
        nonfinite,
        0,
        time_ns(),
        :running,
        _worst(TY, problem.sense),
        0,
        0,
        0,
    )
end

_remaining(core::SolverCore) = max(0, core.criteria.maximum_evaluations - core.evaluations)
_finished(core::SolverCore) = core.retcode != :running

function _evaluate_batch!(values, core::SolverCore, candidates)
    length(values) == size(candidates, 2) ||
        throw(DimensionMismatch("one value per candidate required"))
    evaluate!(values, core.executor, core.problem, candidates, core.nonfinite)
    return values
end
function _evaluate_batch(core::SolverCore, candidates)
    values = Vector{eltype(core.trace.objective_values)}(
        undef,
        size(candidates, 2),
    )
    return _evaluate_batch!(values, core, candidates)
end

function _validated_values(core::SolverCore, candidates, values)
    length(values) == size(candidates, 2) ||
        throw(DimensionMismatch("one value per candidate required"))
    T = eltype(core.trace.objective_values)
    converted = Vector{T}(undef, length(values))
    for index in eachindex(converted)
        converted[index] = _checkedvalue(
            T,
            values[index],
            core.nonfinite,
            core.problem.sense,
        )
    end
    return converted
end

function _update_best!(
    core::SolverCore,
    source::ObservationSource,
)
    count = core.trace.count
    latest = core.trace.objective_values[count]
    latest_loss = _loss(core.problem, latest)
    best_loss = _loss(core.problem, core.best_value)
    significant = if isfinite(core.best_value)
        tolerance = core.criteria.absolute_tolerance +
            core.criteria.relative_tolerance * max(abs(core.best_value), one(latest))
        latest_loss < best_loss - tolerance
    else
        true
    end
    if core.best_index == 0 ||
            _isbetter(core.problem, latest, core.best_value)
        core.best_value = latest
        core.best_index = count
    end
    significant && source != KnownObservation &&
        (core.last_significant_improvement_evaluation = core.evaluations)
    return core
end

function _update_stop!(core::SolverCore, latest_point, latest, batch_size)
    _finished(core) && return core
    event = BatchProgressEvent(
        core.evaluations,
        core.iteration,
        latest,
        core.best_value,
        latest_point,
        batch_size,
    )
    if core.callback(event)
        core.retcode = :callback_stop
    elseif (time_ns() - core.started) / 1e9 >= core.criteria.time_limit
        core.retcode = :time_limit
    elseif core.evaluations >= core.criteria.maximum_evaluations
        core.retcode = :evaluation_limit
    elseif core.iteration >= core.criteria.maximum_iterations
        core.retcode = :iteration_limit
    elseif core.criteria.stall_evaluations > 0 &&
            core.evaluations - core.last_significant_improvement_evaluation >=
                core.criteria.stall_evaluations
        core.retcode = :stalled
    end
    return core
end

function _commit_batch!(
    core::SolverCore,
    candidates,
    values;
    source=InternalEvaluation,
)
    count_evaluations = source != KnownObservation
    !count_evaluations || size(candidates, 2) <= _remaining(core) ||
        throw(ArgumentError("batch exceeds the remaining evaluation budget"))
    batch_size = size(candidates, 2)
    latest_point = nothing
    for column in axes(candidates, 2)
        count_evaluations && (core.evaluations += 1)
        _commit!(
            core.trace,
            @view(candidates[:, column]),
            values[column],
            source,
            count_evaluations ? core.evaluations : 0,
            core.iteration,
            core.started,
        )
        latest_point = decode(core.problem.space, @view candidates[:, column])
        _update_best!(core, source)
    end
    count_evaluations && batch_size > 0 &&
        _update_stop!(core, latest_point, values[batch_size], batch_size)
    return core
end

function _evaluate_commit!(core::SolverCore, candidates)
    isempty(candidates) && return core
    values = _evaluate_batch(core, candidates)
    _commit_batch!(core, candidates, values)
    return core
end

function _sample_feasible(core::SolverCore, n, sampler=UniformDesign())
    candidates = Matrix{eltype(core.trace.latent_points)}(
        undef,
        dimension(core.problem.space),
        n,
    )
    _sample_feasible!(core.rng, candidates, sampler, core.problem)
    return candidates
end

function _result(core::SolverCore, algorithm, model=nothing; statistics=(iterations=core.iteration,))
    statistics = merge((evaluations=core.evaluations,), statistics)
    frozen_trace = _snapshot(core.trace)
    if core.trace.count == 0
        return Result(
            core.problem,
            core.problem.space,
            nothing,
            _worst(eltype(core.trace.objective_values), core.problem.sense),
            frozen_trace,
            model,
            algorithm,
            core.problem.sense,
            core.retcode,
            statistics,
        )
    end
    core.best_index == 0 &&
        throw(AssertionError("nonempty trace has no best observation"))
    index = core.best_index
    point = decode(core.problem.space, @view core.trace.latent_points[:, index])
    return Result(
        core.problem,
        core.problem.space,
        point,
        core.trace.objective_values[index],
        frozen_trace,
        model,
        algorithm,
        core.problem.sense,
        core.retcode,
        statistics,
    )
end

function _points_to_latent(space, points)
    d = dimension(space)
    if points isa NamedTuple ||
            (space isa Box && points isa AbstractVector{<:Real} && length(points) == d) ||
            (space isa SearchSpace && isnothing(dimensionnames(space)) &&
                points isa Tuple && length(points) == d)
        return reshape(encode(space, points), d, 1)
    elseif points isa AbstractMatrix
        size(points, 1) == d ||
            throw(DimensionMismatch("initial-point matrix must have one row per dimension"))
        latent = Matrix{latenttype(space)}(undef, d, size(points, 2))
        for column in axes(points, 2)
            latent[:, column] .= encode(space, @view points[:, column])
        end
        return latent
    end
    collected = collect(points)
    latent = Matrix{latenttype(space)}(undef, d, length(collected))
    for (column, point) in pairs(collected)
        latent[:, column] .= encode(space, point)
    end
    return latent
end

function _seed_initial!(core::SolverCore, points, values)
    if isnothing(points)
        isnothing(values) || throw(ArgumentError("initial_values requires initial_points"))
        return core
    end
    latent = _points_to_latent(core.problem.space, points)
    for column in axes(latent, 2)
        isfeasible(core.problem, decode(core.problem.space, @view(latent[:, column]))) ||
            throw(ArgumentError("initial point $column is infeasible"))
    end
    if isnothing(values)
        size(latent, 2) <= _remaining(core) ||
            throw(ArgumentError("initial points exceed the evaluation budget"))
        evaluated = _evaluate_batch(core, latent)
        _commit_batch!(core, latent, evaluated)
    else
        converted = _validated_values(core, latent, values)
        _commit_batch!(core, latent, converted; source=KnownObservation)
    end
    return core
end
