struct GaussianProcessSurrogate{T<:Real}
    length_scale::T
    noise::T
    jitter::T
end
GaussianProcessSurrogate(; length_scale=0.0, noise=1e-8, jitter=1e-10) =
    GaussianProcessSurrogate(promote(length_scale, noise, jitter)...)

struct GaussianProcessModel{T,M<:AbstractMatrix{T},V<:AbstractVector{T},F}
    points::M
    standardized_values::V
    factor::F
    alpha::V
    output_mean::T
    output_scale::T
    length_scale::T
    noise::T
end

@inline function _matern52(distance_squared, length_scale)
    radius = sqrt(max(distance_squared, zero(distance_squared))) / length_scale
    scaled = sqrt(typeof(radius)(5)) * radius
    return (one(radius) + scaled + scaled^2 / typeof(radius)(3)) * exp(-scaled)
end

function _default_length_scale(points)
    d, n = size(points)
    n <= 1 && return one(eltype(points)) / sqrt(eltype(points)(max(d, 1)))
    nearest_sum = zero(eltype(points))
    for i in 1:n
        nearest = eltype(points)(Inf)
        for j in 1:n
            i == j && continue
            distance = zero(eltype(points))
            @inbounds for row in 1:d
                delta = points[row, i] - points[row, j]
                distance += delta * delta
            end
            nearest = min(nearest, distance)
        end
        nearest_sum += sqrt(nearest)
    end
    return clamp(nearest_sum / n, eltype(points)(0.03), eltype(points)(1.0))
end

function fitmodel(specification::GaussianProcessSurrogate, points, values, rng=Random.default_rng())
    T = promote_type(eltype(points), eltype(values), typeof(float(specification.noise)))
    X = Matrix{T}(points)
    y = Vector{T}(values)
    n = length(y)
    n > 0 || throw(ArgumentError("at least one observation is required"))
    output_mean = mean(y)
    output_scale = max(std(y; corrected=false), sqrt(eps(T)))
    standardized = (y .- output_mean) ./ output_scale
    length_scale = specification.length_scale > 0 ?
        T(specification.length_scale) : _default_length_scale(X)
    covariance = Matrix{T}(undef, n, n)
    for column in 1:n
        covariance[column, column] = one(T) + T(specification.noise)
        for row in column+1:n
            distance = zero(T)
            @inbounds for axis in axes(X, 1)
                delta = X[axis, row] - X[axis, column]
                distance += delta * delta
            end
            value = _matern52(distance, length_scale)
            covariance[row, column] = value
            covariance[column, row] = value
        end
    end
    jitter = T(specification.jitter)
    factor = nothing
    for _ in 1:8
        trial = copy(covariance)
        @inbounds for i in 1:n
            trial[i, i] += jitter
        end
        factor = cholesky(Symmetric(trial); check=false)
        isposdef(factor) && break
        jitter *= T(10)
    end
    !isnothing(factor) && isposdef(factor) ||
        throw(ArgumentError("unable to factor surrogate covariance matrix"))
    alpha = factor \ standardized
    return GaussianProcessModel(
        X,
        standardized,
        factor,
        alpha,
        output_mean,
        output_scale,
        length_scale,
        T(specification.noise),
    )
end

function predictmeanvariance!(means, variances, model::GaussianProcessModel, points)
    n = size(model.points, 2)
    length(means) == size(points, 2) == length(variances) ||
        throw(DimensionMismatch("one prediction per candidate required"))
    kernel = Vector{eltype(model.points)}(undef, n)
    solved = similar(kernel)
    for column in axes(points, 2)
        for observation in 1:n
            distance = zero(eltype(kernel))
            @inbounds for row in axes(points, 1)
                delta = points[row, column] - model.points[row, observation]
                distance += delta * delta
            end
            kernel[observation] = _matern52(distance, model.length_scale)
        end
        means[column] = model.output_mean + model.output_scale * dot(kernel, model.alpha)
        norm_squared = zero(eltype(kernel))
        factors = model.factor.factors
        for row in 1:n
            value = kernel[row]
            @inbounds for previous in 1:row-1
                value -= factors[row, previous] * solved[previous]
            end
            solved[row] = value / factors[row, row]
            norm_squared += solved[row]^2
        end
        latent_variance = max(one(eltype(kernel)) - norm_squared, eps(eltype(kernel)))
        variances[column] = model.output_scale^2 * latent_variance
    end
    return means, variances
end

function predictmean!(means, model::GaussianProcessModel, points)
    variances = similar(means)
    predictmeanvariance!(means, variances, model, points)
    return means
end

struct LowerConfidenceBound{T<:Real}
    exploration::T
end
LowerConfidenceBound(exploration=2.0) = LowerConfidenceBound{typeof(exploration)}(exploration)

function acquisitionvalues!(destination, acquisition::LowerConfidenceBound, mean, variance, current_best)
    @. destination = mean -
        acquisition.exploration * sqrt(max(variance, zero(eltype(variance))))
    return destination
end
