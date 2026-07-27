module SMBOPythonAlignmentTests

using LinearAlgebra
using Random
using SAMBO
using Statistics
using Test

@testset "Python SAMBO-compatible SMBO defaults" begin
    algorithm = SMBO()
    @test algorithm.candidate_pool == 80_000
    @test algorithm.batch_size == 1
    @test algorithm.refit_schedule == FixedRefit(1)
    @test algorithm.acquisition isa SAMBO.RandomizedLowerConfidenceBound
    @test algorithm.candidate_sampler isa SAMBO.AdaptiveDensityCandidates
    @test algorithm.improvement_tolerance == 1e-6
    @test algorithm.no_change_iterations == 5
    @test algorithm.surrogate.kernel isa SAMBO.SquaredExponentialKernel
    @test algorithm.surrogate.length_scale isa SAMBO.AutomaticARDLengthScale
    @test algorithm.surrogate.noise == 1e-14
    @test algorithm.surrogate.jitter isa SAMBO.NoJitter
    @test algorithm.surrogate.optimize_hyperparameters

    for dimensions in (5, 6)
        state = init(
            Problem(Box(zeros(dimensions), ones(dimensions))),
            SMBO(candidate_pool=16);
            initial_points=[fill(0.5, dimensions)],
            initial_values=[1.0],
            maximum_evaluations=300,
            rng=Xoshiro(19),
        )
        @test size(state.initial_design, 2) == 279
    end

    density_state = init(
        Problem(x -> sum(abs2, x), Box(zeros(2), ones(2))),
        SMBO(initial_points=1, candidate_pool=16);
        initial_points=[[0.2, 0.8]],
        initial_values=[1.0],
        maximum_evaluations=4,
        rng=Xoshiro(20),
    )
    pool = Matrix{Float64}(undef, 2, 16)
    @test SAMBO.generate_candidates!(
        pool,
        density_state,
        density_state.algorithm.candidate_sampler,
    ) == 16
    @test all(0 .<= pool .<= 1)

    weights = [0.2, 0.3, 0.5]
    @test SAMBO._silverman_bandwidth(weights, 2) ≈
        (inv(sum(abs2, weights)))^(-1 / 6)

    good = [
        [0.18 + 0.002index, 0.22 - 0.001index]
        for index in 1:20
    ]
    bad = [
        [0.78 + 0.002index, 0.82 - 0.001index]
        for index in 1:20
    ]
    kde_state = init(
        Problem(Box(zeros(2), ones(2))),
        SMBO(initial_points=1, candidate_pool=2_000);
        initial_points=vcat(good, bad),
        initial_values=vcat(zeros(20), ones(20)),
        maximum_evaluations=4,
        rng=Xoshiro(24),
    )
    kde_pool = Matrix{Float64}(undef, 2, 2_000)
    @test SAMBO.generate_candidates!(
        kde_pool,
        kde_state,
        kde_state.algorithm.candidate_sampler,
    ) == 2_000
    @test sum(kde_pool) / length(kde_pool) < 0.35

    points = rand(Xoshiro(22), 2, 12)
    values = @. (points[1, :] - 0.3)^2 + 4(points[2, :] - 0.7)^2
    specification = algorithm.surrogate
    model = SAMBO.fitmodel(specification, points, values, Xoshiro(23))
    @test 0.1 <= model.amplitude <= 10
    @test all(0.01 .<= model.length_scale .<= 100)
    reconstructed = model.factor.L * model.factor.L'
    @test all(
        diag(reconstructed) .≈
        fill(model.amplitude + specification.noise, size(points, 2))
    )
    standardized = (values .- sum(values) / length(values)) ./
        std(values; corrected=false)
    initial_nll = SAMBO._rbf_negative_log_likelihood(
        zeros(3),
        points,
        standardized,
        specification.noise,
    )
    fitted_nll = SAMBO._rbf_negative_log_likelihood(
        log.([model.amplitude; model.length_scale]),
        points,
        standardized,
        specification.noise,
    )
    @test fitted_nll <= initial_nll
    probe = [0.2, -0.4, 0.6]
    _, analytic = SAMBO._rbf_nll_gradient(
        probe,
        points,
        standardized,
        specification.noise,
    )
    finite_difference = similar(analytic)
    step = 1e-5
    for index in eachindex(probe)
        left, right = copy(probe), copy(probe)
        left[index] -= step
        right[index] += step
        finite_difference[index] = (
            SAMBO._rbf_negative_log_likelihood(
                right,
                points,
                standardized,
                specification.noise,
            ) -
            SAMBO._rbf_negative_log_likelihood(
                left,
                points,
                standardized,
                specification.noise,
            )
        ) / (2step)
    end
    @test analytic ≈ finite_difference rtol=2e-4

    stopped = solve(
        Problem(_ -> 1.0, Box(zeros(2), ones(2))),
        SMBO(
            initial_points=1,
            candidate_pool=16,
            no_change_iterations=2,
        );
        maximum_evaluations=30,
        rng=Xoshiro(21),
    )
    @test evaluation_count(stopped) == 4
    @test retcode(stopped) == :stalled
end

end
