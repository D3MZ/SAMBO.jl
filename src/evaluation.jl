isfeasible(problem::Problem, point) = Bool(problem.constraint(point))

function evaluate!(values, ::Serial, problem::Problem, candidates)
    T = eltype(values)
    for column in axes(candidates, 2)
        values[column] = _evaluate(problem, @view(candidates[:, column]), T)
    end
    return values
end

function evaluate!(values, ::Threaded, problem::Problem, candidates)
    T = eltype(values)
    Threads.@threads for column in axes(candidates, 2)
        values[column] = _evaluate(problem, @view(candidates[:, column]), T)
    end
    return values
end

mutable struct SolverCore{P,R,E,C,T}
    problem::P
    rng::R
    executor::E
    callback::C
    trace::Trace{T}
    criteria::StopCriteria{T}
    iteration::Int
    started::Float64
    retcode::Symbol
    best_value::T
    last_improvement::Int
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
)
    maximum_evaluations > 0 || throw(ArgumentError("maximum_evaluations must be positive"))
    maximum_iterations > 0 || throw(ArgumentError("maximum_iterations must be positive"))
    stall_evaluations >= 0 || throw(ArgumentError("stall_evaluations must be nonnegative"))
    time_limit > 0 || throw(ArgumentError("time_limit must be positive"))
    T = latenttype(problem.space)
    criteria = StopCriteria{T}(
        maximum_evaluations,
        maximum_iterations,
        T(absolute_tolerance),
        T(relative_tolerance),
        stall_evaluations,
        Float64(time_limit),
    )
    return SolverCore(
        problem,
        rng,
        executor,
        callback,
        Trace{T}(dimension(problem.space), maximum_evaluations),
        criteria,
        0,
        time(),
        :running,
        T(Inf),
        0,
        0,
    )
end

_remaining(core::SolverCore) = max(0, core.criteria.maximum_evaluations - core.evaluations)
_finished(core::SolverCore) = core.retcode != :running

function _evaluate_batch(core::SolverCore, candidates)
    values = Vector{eltype(core.trace.objective_values)}(undef, size(candidates, 2))
    evaluate!(values, core.executor, core.problem, candidates)
    return values
end

function _validated_values(core::SolverCore, candidates, values)
    length(values) == size(candidates, 2) ||
        throw(DimensionMismatch("one value per candidate required"))
    T = eltype(core.trace.objective_values)
    converted = Vector{T}(undef, length(values))
    for index in eachindex(converted)
        converted[index] = _checkedvalue(T, values[index])
    end
    return converted
end

function _update_stop!(core::SolverCore, latest_point)
    count = core.trace.count
    latest = core.trace.objective_values[count]
    improved = if isfinite(core.best_value)
        tolerance = core.criteria.absolute_tolerance +
            core.criteria.relative_tolerance * max(abs(core.best_value), one(latest))
        latest < core.best_value - tolerance
    else
        true
    end
    if improved
        core.best_value = latest
        core.last_improvement = count
    end
    event = ProgressEvent(core.evaluations, core.iteration, latest, core.best_value, latest_point)
    if core.callback(event)
        core.retcode = :callback_stop
    elseif core.evaluations >= core.criteria.maximum_evaluations
        core.retcode = :evaluation_limit
    elseif core.iteration >= core.criteria.maximum_iterations
        core.retcode = :iteration_limit
    elseif core.criteria.stall_evaluations > 0 &&
            count - core.last_improvement >= core.criteria.stall_evaluations
        core.retcode = :stalled
    elseif time() - core.started >= core.criteria.time_limit
        core.retcode = :time_limit
    end
    return core
end

function _commit_batch!(core::SolverCore, candidates, values; count_evaluations=true)
    !count_evaluations || size(candidates, 2) <= _remaining(core) ||
        throw(ArgumentError("batch exceeds the remaining evaluation budget"))
    for column in axes(candidates, 2)
        count_evaluations && (core.evaluations += 1)
        _commit!(
            core.trace,
            @view(candidates[:, column]),
            values[column],
            core.iteration,
            core.started,
        )
        point = decode(core.problem.space, @view candidates[:, column])
        _update_stop!(core, point)
    end
    return core
end

function _evaluate_commit!(core::SolverCore, candidates)
    isempty(candidates) && return core
    values = _evaluate_batch(core, candidates)
    _commit_batch!(core, candidates, values)
    return core
end

function _sample_feasible(core::SolverCore, n, sampler=UniformDesign())
    candidates = Matrix{eltype(core.trace.objective_values)}(
        undef,
        dimension(core.problem.space),
        n,
    )
    _sample_feasible!(core.rng, candidates, sampler, core.problem)
    return candidates
end

function _result(core::SolverCore, algorithm, model=nothing; statistics=(iterations=core.iteration,))
    statistics = merge((evaluations=core.evaluations,), statistics)
    if core.trace.count == 0
        return Result(
            core.problem.space,
            nothing,
            eltype(core.trace.objective_values)(Inf),
            core.trace,
            model,
            algorithm,
            core.retcode,
            statistics,
        )
    end
    index = argmin(objectivevalues(core.trace))
    point = decode(core.problem.space, @view core.trace.latent_points[:, index])
    return Result(
        core.problem.space,
        point,
        core.trace.objective_values[index],
        core.trace,
        model,
        algorithm,
        core.retcode,
        statistics,
    )
end

function _points_to_latent(space, points)
    d = dimension(space)
    if points isa NamedTuple ||
            (space isa Box && points isa AbstractVector{<:Real} && length(points) == d) ||
            (space isa SearchSpace && isnothing(space.names) && points isa Tuple && length(points) == d)
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
        _commit_batch!(core, latent, converted; count_evaluations=false)
    end
    return core
end
