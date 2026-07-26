@testset "Gaussian-process posterior" begin
    function dense_variance(model, point)
        kernel = [
            SAMBO._matern52(
                sum(abs2, point .- @view(model.points[:, observation])),
                model.length_scale,
            )
            for observation in axes(model.points, 2)
        ]
        latent = max(
            one(eltype(kernel)) - sum(kernel .* (model.factor \ kernel)),
            zero(eltype(kernel)),
        )
        return model.output_scale^2 * latent
    end

    points = reshape([-1.0, -0.1, 0.7, 1.0], 1, :)
    values = [1.2, -0.3, 0.4, 1.8]
    model = SAMBO.fitmodel(
        GaussianProcessSurrogate(length_scale=0.45, noise=1e-6, jitter=1e-10),
        points,
        values,
    )
    queries = reshape([-1.0, -0.5, 0.2, 1.0], 1, :)
    means = zeros(size(queries, 2))
    variances = similar(means)
    SAMBO.predictmeanvariance!(means, variances, model, queries)
    @test variances ≈ [dense_variance(model, @view queries[:, i]) for i in axes(queries, 2)]
    @test all(>=(0), variances)
    @test variances[1] < variances[2]
    @test variances[end] < variances[3]

    means_only = fill(NaN, length(means))
    SAMBO.predictmean!(means_only, model, queries)
    @test means_only ≈ means

    constant_model = SAMBO.fitmodel(
        GaussianProcessSurrogate(length_scale=0.3),
        reshape([0.0, 0.5, 1.0], 1, :),
        fill(7.0, 3),
    )
    @test constant_model.output_scale == 1.0
    constant_variance = zeros(1)
    SAMBO.predictmeanvariance!(
        zeros(1),
        constant_variance,
        constant_model,
        reshape([0.25], 1, :),
    )
    @test constant_variance[1] > 0

    repeated_model = SAMBO.fitmodel(
        GaussianProcessSurrogate(length_scale=0.2, noise=0.0, jitter=1e-12),
        reshape([0.25, 0.25, 0.25], 1, :),
        [1.0, 1.0, 1.0],
    )
    repeated_variance = zeros(1)
    SAMBO.predictmeanvariance!(
        zeros(1),
        repeated_variance,
        repeated_model,
        reshape([0.25], 1, :),
    )
    @test 0 <= repeated_variance[1] < 1e-8

    for T in (Float64, BigFloat)
        typed_model = SAMBO.fitmodel(
            GaussianProcessSurrogate(
                length_scale=T(0.3),
                noise=T(1e-6),
                jitter=T(1e-10),
            ),
            reshape(T[0, 0.5, 1], 1, :),
            T[0, 1, 0],
        )
        typed_variance = zeros(T, 2)
        SAMBO.predictmeanvariance!(
            zeros(T, 2),
            typed_variance,
            typed_model,
            reshape(T[0.25, 0.75], 1, :),
        )
        @test eltype(typed_variance) == T
        @test all(isfinite, typed_variance)
        @test all(>=(0), typed_variance)
    end

    anisotropic = SAMBO.fitmodel(
        GaussianProcessSurrogate(length_scale=[0.2, 0.8]),
        [0.0 0.5 1.0; 0.0 0.25 1.0],
        [0.0, 1.0, 0.0],
    )
    @test anisotropic.length_scale == [0.2, 0.8]
    @test_throws DimensionMismatch SAMBO.fitmodel(
        GaussianProcessSurrogate(length_scale=[0.2]),
        [0.0 1.0; 0.0 1.0],
        [0.0, 1.0],
    )

    ensemble = SAMBO.fitmodel(
        EnsembleSurrogate(
            base=GaussianProcessSurrogate(length_scale=0.3),
            count=3,
        ),
        reshape([0.0, 0.5, 1.0], 1, :),
        [0.0, 1.0, 0.0],
        MersenneTwister(91),
    )
    ensemble_mean = zeros(2)
    ensemble_variance = zeros(2)
    SAMBO.predictmeanvariance!(
        ensemble_mean,
        ensemble_variance,
        ensemble,
        reshape([0.25, 0.75], 1, :),
    )
    @test all(isfinite, ensemble_mean)
    @test all(>=(0), ensemble_variance)
    @test_throws ArgumentError EnsembleSurrogate(count=0)

    @test_throws ArgumentError GaussianProcessSurrogate(length_scale=-1)
    @test_throws ArgumentError GaussianProcessSurrogate(length_scale=Inf)
    @test_throws ArgumentError GaussianProcessSurrogate(noise=-1)
    @test_throws ArgumentError GaussianProcessSurrogate(jitter=0)
    @test_throws ArgumentError GaussianProcessSurrogate(length_scale=[0.2, -0.1])
end
