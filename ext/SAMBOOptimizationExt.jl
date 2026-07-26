module SAMBOOptimizationExt

using SAMBO
import OptimizationBase

const _SAMBOAlgorithm =
    Union{SAMBO.SMBO,SAMBO.SCEUA,SAMBO.SHGO,SAMBO.TopologicalMultistart}

function _constraint(problem)
    isnothing(problem.lcons) && return SAMBO.Unconstrained()
    hasproperty(problem.f, :cons) ||
        throw(ArgumentError("constraint bounds require an OptimizationFunction constraint"))
    constraint_function = getproperty(problem.f, :cons)
    return function (point)
        values = constraint_function(point, problem.p)
        return all(problem.lcons .<= values) && all(values .<= problem.ucons)
    end
end

function OptimizationBase.solve(
    problem::OptimizationBase.OptimizationProblem,
    algorithm::_SAMBOAlgorithm;
    maxiters=100,
    maximum_evaluations=maxiters,
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
        initial_points=isnothing(problem.u0) ? nothing : vec(problem.u0),
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
        retcode=SAMBO.retcode(native_result) in (:success, :evaluation_limit) ?
            OptimizationBase.ReturnCode.Success : OptimizationBase.ReturnCode.Terminated,
        original=native_result,
        stats=statistics,
    )
end

end
