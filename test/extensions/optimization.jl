using SAMBO
using OptimizationBase
using Random
using Test

@testset "OptimizationBase extension" begin
    problem = OptimizationBase.OptimizationProblem(
        (point, parameters) -> sum(abs2, point),
        zeros(2);
        lb=fill(-1.0, 2),
        ub=fill(1.0, 2),
    )
    solution = OptimizationBase.solve(
        problem,
        SAMBO.SMBO(candidate_pool=32);
        maxiters=8,
        rng=MersenneTwister(6),
    )
    @test solution.stats.fevals == 8
    @test solution.original isa SAMBO.Result
    @test solution.u == SAMBO.minimizer(solution.original)

    maximization = OptimizationBase.OptimizationProblem(
        (point, parameters) -> -(point[1] - 0.7)^2,
        nothing;
        lb=[0.0],
        ub=[1.0],
        sense=OptimizationBase.MaxSense,
    )
    maximized = OptimizationBase.solve(
        maximization,
        SAMBO.SMBO(candidate_pool=32);
        maxiters=12,
        rng=MersenneTwister(60),
    )
    @test maximized.objective > -0.05
    @test maximized.original.sense isa SAMBO.Maximize

end
