struct AutomaticLengthScale end
struct IsotropicLengthScale{T<:Real}
    value::T
end
struct ARDLengthScale{V<:AbstractVector}
    values::V
    function ARDLengthScale(values::AbstractVector)
        !isempty(values) && all(value -> isfinite(value) && value > 0, values) ||
            throw(ArgumentError("ARD length scales must be finite and positive"))
        copied = collect(values)
        new{typeof(copied)}(copied)
    end
end
struct GeometricJitter{T<:Real}
    initial::T
    factor::T
    attempts::Int
    function GeometricJitter(
        initial::T,
        factor::T,
        attempts::Int,
    ) where {T<:Real}
        isfinite(initial) && initial > 0 ||
            throw(ArgumentError("initial jitter must be finite and positive"))
        isfinite(factor) && factor > 1 ||
            throw(ArgumentError("jitter factor must be finite and greater than one"))
        attempts > 0 || throw(ArgumentError("jitter attempts must be positive"))
        new{T}(initial, factor, attempts)
    end
end
function GeometricJitter(initial=1e-10, factor=10.0, attempts=8)
    promoted = promote(initial, factor)
    return GeometricJitter(promoted[1], promoted[2], attempts)
end

_length_scale(::AutomaticLengthScale) = AutomaticLengthScale()
function _length_scale(scale::Real)
    scale == 0 && return AutomaticLengthScale()
    isfinite(scale) && scale > 0 ||
        throw(ArgumentError("length_scale must be automatic or finite and positive"))
    return IsotropicLengthScale(scale)
end
function _length_scale(scale::AbstractVector)
    return ARDLengthScale(scale)
end
_jitter_policy(policy::GeometricJitter) = policy
_jitter_policy(initial::Real) = GeometricJitter(initial)

struct GaussianProcessSurrogate{L,N<:Real,J}
    length_scale::L
    noise::N
    jitter::J
end
function GaussianProcessSurrogate(;
    length_scale=AutomaticLengthScale(),
    noise=1e-8,
    jitter=GeometricJitter(),
)
    isfinite(noise) && noise >= 0 ||
        throw(ArgumentError("noise must be finite and nonnegative"))
    return GaussianProcessSurrogate(
        _length_scale(length_scale),
        noise,
        _jitter_policy(jitter),
    )
end

struct GaussianProcessModel{T,M<:AbstractMatrix{T},V<:AbstractVector{T},F,LM,L}
    points::M
    factor::F
    lower::LM
    alpha::V
    output_mean::T
    output_scale::T
    length_scale::L
    noise::T
end

resolved_length_scale_type(::AutomaticLengthScale, ::Type{T}) where {T} = T
resolved_length_scale_type(::IsotropicLengthScale, ::Type{T}) where {T} = T
resolved_length_scale_type(::ARDLengthScale, ::Type{T}) where {T} = Vector{T}
fittedmodeltype(specification, ::Type{TX}, ::Type{TY}) where {TX,TY} = Any
function fittedmodeltype(
    specification::GaussianProcessSurrogate{L,N},
    ::Type{TX},
    ::Type{TY},
) where {L,N,TX,TY}
    T = promote_type(TX, TY, typeof(float(specification.noise)))
    LS = resolved_length_scale_type(specification.length_scale, T)
    return GaussianProcessModel{
        T,
        Matrix{T},
        Vector{T},
        Cholesky{T,Matrix{T}},
        Matrix{T},
        LS,
    }
end

@inline function _scaled_distance(left, right, length_scale::Real)
    inverse_scale = inv(length_scale)
    distance = zero(promote_type(
        eltype(left),
        eltype(right),
        typeof(length_scale),
    ))
    @inbounds for index in eachindex(left, right)
        delta = (left[index] - right[index]) * inverse_scale
        distance += delta * delta
    end
    return distance
end
@inline function _distance_squared(left, right)
    distance = zero(promote_type(eltype(left), eltype(right)))
    @inbounds for index in eachindex(left, right)
        delta = left[index] - right[index]
        distance += delta * delta
    end
    return distance
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

resolve_length_scale(::AutomaticLengthScale, points, ::Type{T}) where {T} =
    _default_length_scale(points)
resolve_length_scale(scale::IsotropicLengthScale, points, ::Type{T}) where {T} =
    T(scale.value)
function resolve_length_scale(scale::ARDLengthScale, points, ::Type{T}) where {T}
    length(scale.values) == size(points, 1) ||
        throw(DimensionMismatch("one length scale per coordinate required"))
    return T.(scale.values)
end

function _factor_covariance(covariance, policy::GeometricJitter, ::Type{T}) where {T}
    jitter = T(policy.initial)
    factor = nothing
    for _ in 1:policy.attempts
        trial = copy(covariance)
        @inbounds for index in axes(trial, 1)
            trial[index, index] += jitter
        end
        factor = cholesky(Symmetric(trial); check=false)
        isposdef(factor) && return factor
        jitter *= T(policy.factor)
    end
    throw(NumericalFailureError(
        "unable to factor surrogate covariance matrix",
    ))
end

function fitmodel(specification::GaussianProcessSurrogate, points, values, rng=Random.default_rng())
    size(points, 2) == length(values) ||
        throw(DimensionMismatch("one value per training point required"))
    all(isfinite, points) ||
        throw(ArgumentError("training points must be finite"))
    all(isfinite, values) ||
        throw(ArgumentError("training values must be finite"))
    isfinite(specification.noise) && specification.noise >= 0 ||
        throw(ArgumentError("noise must be finite and nonnegative"))
    T = promote_type(eltype(points), eltype(values), typeof(float(specification.noise)))
    X = Matrix{T}(points)
    y = Vector{T}(values)
    n = length(y)
    n > 0 || throw(ArgumentError("at least one observation is required"))
    output_mean = mean(y)
    output_scale = std(y; corrected=false)
    iszero(output_scale) && (output_scale = one(T))
    standardized = (y .- output_mean) ./ output_scale
    length_scale = resolve_length_scale(specification.length_scale, X, T)
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
    factor = _factor_covariance(covariance, specification.jitter, T)
    alpha = factor \ standardized
    return GaussianProcessModel(
        X,
        factor,
        Matrix(factor.L),
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
function kernelmatrix!(kernel, model::GaussianProcessModel, points)
    size(kernel) == (size(model.points, 2), size(points, 2)) ||
        throw(DimensionMismatch("kernel destination has the wrong size"))
    for column in axes(points, 2), observation in axes(model.points, 2)
        kernel[observation, column] = _matern52_scaled(_scaled_distance(
            @view(points[:, column]),
            @view(model.points[:, observation]),
            model.length_scale,
        ))
    end
    return kernel
end

mutable struct GPPredictionWorkspace{T}
    kernel::Matrix{T}
    solved_kernel::Matrix{T}
end
GPPredictionWorkspace(::Type{T}) where {T} =
    GPPredictionWorkspace(Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0))
predictionworkspace(specification, ::Type{T}) where {T} = nothing
predictionworkspace(::GaussianProcessSurrogate, ::Type{T}) where {T} =
    GPPredictionWorkspace(T)
function ensure_capacity!(workspace::GPPredictionWorkspace{T}, observations, candidates) where {T}
    if size(workspace.kernel, 1) < observations ||
            size(workspace.kernel, 2) < candidates
        rows = max(observations, size(workspace.kernel, 1))
        columns = max(candidates, size(workspace.kernel, 2))
        workspace.kernel = Matrix{T}(undef, rows, columns)
        workspace.solved_kernel = Matrix{T}(undef, rows, columns)
    end
    return workspace
end
function _forward_solve!(destination, lower)
    @inbounds for column in axes(destination, 2)
        for row in axes(destination, 1)
            value = destination[row, column]
            for previous in 1:row-1
                value -= lower[row, previous] * destination[previous, column]
            end
            destination[row, column] = value / lower[row, row]
        end
    end
    return destination
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

mutable struct EnsemblePredictionWorkspace{T,W}
    member_mean::Vector{T}
    member_variance::Vector{T}
    base_workspace::W
end
predictionworkspace(specification::EnsembleSurrogate, ::Type{T}) where {T} =
    EnsemblePredictionWorkspace(
        T[],
        T[],
        predictionworkspace(specification.base, T),
    )
function _ensure_ensemble_capacity!(
    workspace::EnsemblePredictionWorkspace,
    candidates,
)
    resize!(workspace.member_mean, candidates)
    resize!(workspace.member_variance, candidates)
    return workspace
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
    workspace = EnsemblePredictionWorkspace(
        similar(means, 0),
        similar(variances, 0),
        nothing,
    )
    return predictmeanvariance!(means, variances, model, points, workspace)
end
function predictmeanvariance!(
    means,
    variances,
    model::EnsembleModel,
    points,
    workspace::EnsemblePredictionWorkspace,
)
    length(means) == size(points, 2) == length(variances) ||
        throw(DimensionMismatch("one prediction per candidate required"))
    _ensure_ensemble_capacity!(workspace, length(means))
    fill!(means, zero(eltype(means)))
    fill!(variances, zero(eltype(variances)))
    member_mean = workspace.member_mean
    member_variance = workspace.member_variance
    for member in model.models
        predictmeanvariance!(
            member_mean,
            member_variance,
            member,
            points,
            workspace.base_workspace,
        )
        @. means += member_mean
        @. variances += member_variance + member_mean^2
    end
    count = length(model.models)
    @. means /= count
    @. variances = max(variances / count - means^2, zero(eltype(variances)))
    return means, variances
end

function predictmean!(means, model::EnsembleModel, points)
    workspace = EnsemblePredictionWorkspace(
        similar(means, 0),
        similar(means, 0),
        nothing,
    )
    return predictmean!(means, model, points, workspace)
end
function predictmean!(
    means,
    model::EnsembleModel,
    points,
    workspace::EnsemblePredictionWorkspace,
)
    length(means) == size(points, 2) ||
        throw(DimensionMismatch("one prediction per candidate required"))
    _ensure_ensemble_capacity!(workspace, length(means))
    fill!(means, zero(eltype(means)))
    member_mean = workspace.member_mean
    for member in model.models
        predictmean!(
            member_mean,
            member,
            points,
            workspace.base_workspace,
        )
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
    workspace = GPPredictionWorkspace(eltype(model.points))
    return predictmeanvariance!(means, variances, model, points, workspace)
end
function predictmeanvariance!(
    means,
    variances,
    model::GaussianProcessModel,
    points,
    workspace::GPPredictionWorkspace,
)
    length(means) == size(points, 2) == length(variances) ||
        throw(DimensionMismatch("one prediction per candidate required"))
    observations = size(model.points, 2)
    candidates = size(points, 2)
    ensure_capacity!(workspace, observations, candidates)
    kernel = @view workspace.kernel[1:observations, 1:candidates]
    kernelmatrix!(kernel, model, points)
    mul!(means, transpose(kernel), model.alpha)
    @. means = model.output_mean + model.output_scale * means
    solved_kernel =
        @view workspace.solved_kernel[1:observations, 1:candidates]
    copyto!(solved_kernel, kernel)
    _forward_solve!(solved_kernel, model.lower)
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
    workspace = GPPredictionWorkspace(eltype(model.points))
    return predictmean!(means, model, points, workspace)
end
function predictmean!(
    means,
    model::GaussianProcessModel,
    points,
    workspace::GPPredictionWorkspace,
)
    length(means) == size(points, 2) ||
        throw(DimensionMismatch("one prediction per candidate required"))
    observations = size(model.points, 2)
    candidates = size(points, 2)
    ensure_capacity!(workspace, observations, candidates)
    kernel = @view workspace.kernel[1:observations, 1:candidates]
    kernelmatrix!(kernel, model, points)
    mul!(means, transpose(kernel), model.alpha)
    @. means = model.output_mean + model.output_scale * means
    return means
end

predictmeanvariance!(means, variances, model, points, workspace) =
    predictmeanvariance!(means, variances, model, points)
predictmean!(means, model, points, workspace) =
    predictmean!(means, model, points)

struct LowerConfidenceBound{T<:Real}
    exploration::T
end
LowerConfidenceBound(exploration=2.0) = LowerConfidenceBound{typeof(exploration)}(exploration)
struct GreedyMean end
struct DistanceUncertainty{S}
    surrogate::S
end
clone_surrogate(surrogate) = deepcopy(surrogate)

function acquisitionvalues!(destination, acquisition::LowerConfidenceBound, mean, variance, current_best)
    @. destination = mean -
        acquisition.exploration * sqrt(max(variance, zero(eltype(variance))))
    return destination
end
function acquisitionvalues!(destination, ::GreedyMean, mean, variance, current_best)
    copyto!(destination, mean)
    return destination
end
