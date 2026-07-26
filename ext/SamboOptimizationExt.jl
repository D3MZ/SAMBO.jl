module SamboOptimizationExt

using Sambo
import OptimizationBase

const _SamboAlgorithm = Union{Sambo.SMBO,Sambo.SCEUA,Sambo.SHGO}

function _constraint(problem)
    isnothing(problem.lcons) && return Sambo.Unconstrained()
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
    algorithm::_SamboAlgorithm;
    maxiters=100,
    maximum_evaluations=maxiters,
    kwargs...,
)
    isnothing(problem.lb) &&
        throw(ArgumentError("Sambo algorithms require lower and upper box bounds"))
    problem.sense == OptimizationBase.MaxSense &&
        throw(ArgumentError("Sambo's OptimizationBase adapter currently supports minimization only"))
    native_problem = Sambo.Problem(
        point -> problem.f(point, problem.p),
        Sambo.Box(vec(problem.lb), vec(problem.ub));
        constraint=_constraint(problem),
    )
    native_result = Sambo.solve(
        native_problem,
        algorithm;
        maximum_evaluations,
        initial_points=vec(problem.u0),
        kwargs...,
    )
    statistics = OptimizationBase.OptimizationStats(
        iterations=Sambo.iteration_count(native_result),
        time=isempty(Sambo.trace(native_result).elapsed_seconds) ? 0.0 :
            Sambo.trace(native_result).elapsed_seconds[Sambo.trace(native_result).count],
        fevals=Sambo.evaluation_count(native_result),
    )
    cache = OptimizationBase.SciMLBase.DefaultOptimizationCache(problem.f, problem.p)
    return OptimizationBase.SciMLBase.build_solution(
        cache,
        algorithm,
        Sambo.minimizer(native_result),
        Sambo.minimum(native_result);
        retcode=Sambo.retcode(native_result) in (:success, :evaluation_limit) ?
            OptimizationBase.ReturnCode.Success : OptimizationBase.ReturnCode.Terminated,
        original=native_result,
        stats=statistics,
    )
end

end
