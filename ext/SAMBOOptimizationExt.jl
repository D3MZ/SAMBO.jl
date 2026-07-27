module SAMBOOptimizationExt

using SAMBO
import OptimizationBase

const _SAMBOAlgorithm =
    Union{SAMBO.SMBO,SAMBO.SCEUA,SAMBO.SHGO,SAMBO.TopologicalMultistart}

struct _InPlaceConstraint{F,P,L,U,B}
    constraint::F
    parameters::P
    lower::L
    upper::U
    buffers::B
end

struct _OutOfPlaceConstraint{F,P,L,U}
    constraint::F
    parameters::P
    lower::L
    upper::U
end

function _satisfies_bounds(values, lower, upper)
    length(values) == length(lower) == length(upper) ||
        throw(DimensionMismatch("constraint values and bounds differ in length"))
    for index in eachindex(values, lower, upper)
        lower[index] <= values[index] <= upper[index] || return false
    end
    return true
end

function (constraint::_InPlaceConstraint)(point)
    values = constraint.buffers[Threads.threadid()]
    constraint.constraint(values, point, constraint.parameters)
    return _satisfies_bounds(values, constraint.lower, constraint.upper)
end

function (constraint::_OutOfPlaceConstraint)(point)
    values = constraint.constraint(point, constraint.parameters)
    return _satisfies_bounds(values, constraint.lower, constraint.upper)
end

function _constraint(problem)
    isnothing(problem.lcons) && return SAMBO.Unconstrained()
    hasproperty(problem.f, :cons) ||
        throw(ArgumentError("constraint bounds require an OptimizationFunction constraint"))
    constraint_function = getproperty(problem.f, :cons)
    isnothing(constraint_function) &&
        throw(ArgumentError("constraint bounds require an OptimizationFunction constraint"))
    prototype = hasproperty(problem, :lb) ?
        vec(problem.lb) :
        zeros(promote_type(eltype(problem.lcons), eltype(problem.ucons)), length(problem.lcons))
    T = promote_type(
        eltype(prototype),
        eltype(problem.lcons),
        eltype(problem.ucons),
    )
    residual = Vector{T}(undef, length(problem.lcons))
    if applicable(constraint_function, residual, prototype, problem.p)
        return _InPlaceConstraint(
            constraint_function,
            problem.p,
            problem.lcons,
            problem.ucons,
            [
                Vector{T}(undef, length(problem.lcons))
                for _ in 1:Threads.maxthreadid()
            ],
        )
    elseif applicable(constraint_function, prototype, problem.p)
        return _OutOfPlaceConstraint(
            constraint_function,
            problem.p,
            problem.lcons,
            problem.ucons,
        )
    end
    throw(ArgumentError(
        "constraint must accept either (residual, point, parameters) or (point, parameters)",
    ))
end

_returncode(::Val{:success}) = OptimizationBase.ReturnCode.Success
_returncode(::Val{:evaluation_limit}) = OptimizationBase.ReturnCode.MaxIters
_returncode(::Val{:iteration_limit}) = OptimizationBase.ReturnCode.MaxIters
_returncode(::Val{:time_limit}) = OptimizationBase.ReturnCode.MaxTime
_returncode(::Val{:stalled}) = OptimizationBase.ReturnCode.Stalled
_returncode(::Val{:callback_stop}) = OptimizationBase.ReturnCode.Terminated
_returncode(::Val{:infeasible_space}) = OptimizationBase.ReturnCode.Infeasible
_returncode(::Val{:numerical_failure}) = OptimizationBase.ReturnCode.Failure
_returncode(::Val{:space_exhausted}) = OptimizationBase.ReturnCode.Success
_returncode(::Val) = OptimizationBase.ReturnCode.Failure

_returncode(retcode::Symbol) = _returncode(Val(retcode))

function _callback(callback, parameters)
    return function (event)
        state = OptimizationBase.OptimizationState(
            iter=event.iteration,
            u=event.latest_point,
            objective=event.latest_value,
            original=event,
            p=parameters,
        )
        return callback(state, event.latest_value)
    end
end

function OptimizationBase.solve(
    problem::OptimizationBase.OptimizationProblem,
    algorithm::_SAMBOAlgorithm;
    maxiters=100,
    maximum_evaluations=maxiters,
    maximum_iterations=maxiters,
    maxtime=nothing,
    time_limit=isnothing(maxtime) ? Inf : maxtime,
    abstol=0.0,
    reltol=0.0,
    callback=OptimizationBase.DEFAULT_CALLBACK,
    kwargs...,
)
    isnothing(problem.lb) &&
        throw(ArgumentError("SAMBO algorithms require lower and upper box bounds"))
    isnothing(problem.ub) &&
        throw(ArgumentError("SAMBO algorithms require lower and upper box bounds"))
    length(problem.lb) == length(problem.ub) ||
        throw(DimensionMismatch("OptimizationProblem bounds differ in length"))
    sense = problem.sense == OptimizationBase.MaxSense ?
        SAMBO.Maximize() : SAMBO.Minimize()
    native_problem = SAMBO.Problem(
        point -> problem.f(point, problem.p),
        SAMBO.Box(vec(problem.lb), vec(problem.ub));
        constraint=_constraint(problem),
        sense,
    )
    native_result = SAMBO.solve(
        native_problem,
        algorithm;
        maximum_evaluations,
        maximum_iterations,
        initial_points=isnothing(problem.u0) ? nothing : vec(problem.u0),
        time_limit,
        absolute_tolerance=abstol,
        relative_tolerance=reltol,
        callback=_callback(callback, problem.p),
        kwargs...,
    )
    statistics = OptimizationBase.OptimizationStats(
        iterations=SAMBO.iteration_count(native_result),
        time=isempty(SAMBO.trace(native_result).elapsed_seconds) ? 0.0 :
            SAMBO.trace(native_result).elapsed_seconds[SAMBO.trace(native_result).count],
        fevals=SAMBO.evaluation_count(native_result),
    )
    cache = OptimizationBase.SciMLBase.DefaultOptimizationCache(problem.f, problem.p)
    return OptimizationBase.SciMLBase.build_solution(
        cache,
        algorithm,
        SAMBO.minimizer(native_result),
        SAMBO.minimum(native_result);
        retcode=_returncode(SAMBO.retcode(native_result)),
        original=native_result,
        stats=statistics,
    )
end

end
