struct GaussianProcessSurrogate{L,N<:Real,J<:Real}
    length_scale::L
    noise::N
    jitter::J
end
function GaussianProcessSurrogate(; length_scale=0.0, noise=1e-8, jitter=1e-10)
    valid_length_scale = length_scale isa AbstractVector ?
        !isempty(length_scale) && all(value -> isfinite(value) && value > 0, length_scale) :
        length_scale == 0 || isfinite(length_scale) && length_scale > 0
    valid_length_scale ||
        throw(ArgumentError("length_scale must be zero (automatic) or finite and positive"))
    isfinite(noise) && noise >= 0 ||
        throw(ArgumentError("noise must be finite and nonnegative"))
    isfinite(jitter) && jitter > 0 ||
        throw(ArgumentError("jitter must be finite and positive"))
    promoted_noise, promoted_jitter = promote(noise, jitter)
    stored_scale = length_scale isa AbstractVector ?
        collect(length_scale) : length_scale
    return GaussianProcessSurrogate(
        stored_scale,
        promoted_noise,
        promoted_jitter,
    )
end

struct GaussianProcessModel{T,M<:AbstractMatrix{T},V<:AbstractVector{T},F,L}
    points::M
    factor::F
    alpha::V
    output_mean::T
    output_scale::T
    length_scale::L
    noise::T
end

@inline function _scaled_distance(left, right, length_scale::Real)
    return sum(abs2, left .- right) / length_scale^2
end
@inline function _scaled_distance(left, right, length_scale::AbstractVector)
    length(left) == length(length_scale) ||
        throw(DimensionMismatch("one length scale per coordinate required"))
    distance = zero(promote_type(eltype(left), eltype(length_scale)))
    @inbounds for index in eachindex(left, right, length_scale)
        distance += ((left[index] - right[index]) / length_scale[index])^2
    end
    return distance
end
@inline function _matern52_scaled(distance_squared)
    scaled = sqrt(typeof(distance_squared)(5) * max(
        distance_squared,
        zero(distance_squared),
    ))
    return (
        one(scaled) + scaled + scaled^2 / typeof(scaled)(3)
    ) * exp(-scaled)
end

@inline function _matern52(distance_squared, length_scale)
    return _matern52_scaled(distance_squared / length_scale^2)
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
    valid_length_scale = specification.length_scale isa AbstractVector ?
        !isempty(specification.length_scale) &&
            all(value -> isfinite(value) && value > 0, specification.length_scale) :
        specification.length_scale == 0 ||
            isfinite(specification.length_scale) && specification.length_scale > 0
    valid_length_scale ||
        throw(ArgumentError("length_scale must be zero (automatic) or finite and positive"))
    isfinite(specification.noise) && specification.noise >= 0 ||
        throw(ArgumentError("noise must be finite and nonnegative"))
    isfinite(specification.jitter) && specification.jitter > 0 ||
        throw(ArgumentError("jitter must be finite and positive"))
    T = promote_type(eltype(points), eltype(values), typeof(float(specification.noise)))
    X = Matrix{T}(points)
    y = Vector{T}(values)
    n = length(y)
    n > 0 || throw(ArgumentError("at least one observation is required"))
    output_mean = mean(y)
    output_scale = std(y; corrected=false)
    iszero(output_scale) && (output_scale = one(T))
    standardized = (y .- output_mean) ./ output_scale
    length_scale = if specification.length_scale isa AbstractVector
        length(specification.length_scale) == size(X, 1) ||
            throw(DimensionMismatch("one length scale per coordinate required"))
        T.(specification.length_scale)
    elseif specification.length_scale > 0
        T(specification.length_scale)
    else
        _default_length_scale(X)
    end
    covariance = Matrix{T}(undef, n, n)
    for column in 1:n
        covariance[column, column] = one(T) + T(specification.noise)
        for row in column+1:n
            value = _matern52_scaled(_scaled_distance(
                @view(X[:, row]),
                @view(X[:, column]),
                length_scale,
            ))
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
        throw(NumericalFailureError(
            "unable to factor surrogate covariance matrix",
        ))
    alpha = factor \ standardized
    return GaussianProcessModel(
        X,
        factor,
        alpha,
        output_mean,
        output_scale,
        length_scale,
        T(specification.noise),
    )
end

function _kernelmatrix(model::GaussianProcessModel, points)
    size(points, 1) == size(model.points, 1) ||
        throw(DimensionMismatch("candidate and model dimensions differ"))
    T = eltype(model.points)
    kernel = Matrix{T}(undef, size(model.points, 2), size(points, 2))
    for column in axes(points, 2), observation in axes(model.points, 2)
        kernel[observation, column] = _matern52_scaled(_scaled_distance(
            @view(points[:, column]),
            @view(model.points[:, observation]),
            model.length_scale,
        ))
    end
    return kernel
end

struct EnsembleSurrogate{S}
    base::S
    count::Int
    function EnsembleSurrogate(base::S, count::Int) where {S}
        count > 0 || throw(ArgumentError("ensemble count must be positive"))
        new{S}(base, count)
    end
end
EnsembleSurrogate(base) = EnsembleSurrogate(base, 5)
EnsembleSurrogate(; base=GaussianProcessSurrogate(), count=5) =
    EnsembleSurrogate(base, count)

struct EnsembleModel{M}
    models::Vector{M}
end

function fitmodel(specification::EnsembleSurrogate, points, values, rng=Random.default_rng())
    count = size(points, 2)
    first_indices = Random.rand(rng, 1:count, count)
    first_model = fitmodel(
        specification.base,
        points[:, first_indices],
        values[first_indices],
        rng,
    )
    models = Vector{typeof(first_model)}(undef, specification.count)
    models[1] = first_model
    for member in 2:specification.count
        indices = Random.rand(rng, 1:count, count)
        models[member] = fitmodel(
            specification.base,
            points[:, indices],
            values[indices],
            rng,
        )
    end
    return EnsembleModel(models)
end

function predictmeanvariance!(means, variances, model::EnsembleModel, points)
    fill!(means, zero(eltype(means)))
    fill!(variances, zero(eltype(variances)))
    member_mean = similar(means)
    member_variance = similar(variances)
    for member in model.models
        predictmeanvariance!(member_mean, member_variance, member, points)
        @. means += member_mean
        @. variances += member_variance + member_mean^2
    end
    count = length(model.models)
    @. means /= count
    @. variances = max(variances / count - means^2, zero(eltype(variances)))
    return means, variances
end

function predictmean!(means, model::EnsembleModel, points)
    fill!(means, zero(eltype(means)))
    member_mean = similar(means)
    for member in model.models
        predictmean!(member_mean, member, points)
        @. means += member_mean
    end
    means ./= length(model.models)
    return means
end

"""
Predict posterior means and latent-function variances.

The returned variance excludes observation noise.
"""
function predictmeanvariance!(means, variances, model::GaussianProcessModel, points)
    length(means) == size(points, 2) == length(variances) ||
        throw(DimensionMismatch("one prediction per candidate required"))
    kernel = _kernelmatrix(model, points)
    mul!(means, transpose(kernel), model.alpha)
    @. means = model.output_mean + model.output_scale * means
    solved_kernel = copy(kernel)
    ldiv!(model.factor.L, solved_kernel)
    T = eltype(solved_kernel)
    for column in axes(solved_kernel, 2)
        latent_variance = max(
            one(T) - sum(abs2, @view solved_kernel[:, column]),
            zero(T),
        )
        variances[column] = model.output_scale^2 * latent_variance
    end
    return means, variances
end

function predictmean!(means, model::GaussianProcessModel, points)
    length(means) == size(points, 2) ||
        throw(DimensionMismatch("one prediction per candidate required"))
    kernel = _kernelmatrix(model, points)
    mul!(means, transpose(kernel), model.alpha)
    @. means = model.output_mean + model.output_scale * means
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
