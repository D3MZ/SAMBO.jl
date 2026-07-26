using BenchmarkTools
using Random
using SAMBO

const BUDGET = 40
const SEED = 42
const SAMPLES = 20

rosenbrock(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2

function benchmark_suite()
    space = Box([-2.0, -1.0], [2.0, 3.0])
    problem = Problem(rosenbrock, space)
    configurations = (
        ("SCE-UA", SCEUA()),
        ("SMBO-GP", SMBO(candidate_pool=1024)),
    )
    suite = BenchmarkGroup()
    for (name, algorithm) in configurations
        suite[name] = @benchmarkable solve(
            $problem,
            $algorithm;
            maximum_evaluations=$BUDGET,
            rng=MersenneTwister($SEED),
        ) evals=1 samples=SAMPLES
    end
    return suite, problem, configurations
end

function main(io=stdout)
    suite, problem, configurations = benchmark_suite()
    trials = run(suite; verbose=false)
    println(
        io,
        "julia_version,algorithm,time_ms,bytes,allocations,minimum,objective_calls,requested_budget,timing_samples,seed",
    )
    for (name, algorithm) in configurations
        estimate = median(trials[name])
        result = solve(
            problem,
            algorithm;
            maximum_evaluations=BUDGET,
            rng=MersenneTwister(SEED),
        )
        println(
            io,
            VERSION, ',', name, ',', estimate.time / 1e6, ',', estimate.memory, ',',
            estimate.allocs, ',', minimum(result), ',', evaluation_count(result), ',',
            BUDGET, ',', SAMPLES, ',', SEED,
        )
    end
end

main()
