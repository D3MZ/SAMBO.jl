abstract type AbstractDesign end
struct UniformDesign <: AbstractDesign end
struct LatinHypercubeDesign <: AbstractDesign end
struct HaltonDesign <: AbstractDesign
    skip::Int
end
HaltonDesign(; skip=20) = HaltonDesign(skip)
struct SobolDesign <: AbstractDesign
    skip::Int
end
SobolDesign(; skip=0) = SobolDesign(skip)

function sample!(rng, destination::AbstractMatrix, ::UniformDesign, space)
    size(destination, 1) == dimension(space) ||
        throw(DimensionMismatch("sample matrix and space differ in dimension"))
    Random.rand!(rng, destination)
    return destination
end

function sample!(rng, destination::AbstractMatrix{T}, ::LatinHypercubeDesign, space) where {T}
    d, n = size(destination)
    d == dimension(space) || throw(DimensionMismatch("sample matrix and space differ in dimension"))
    n == 0 && return destination
    destination .= QuasiMonteCarlo.sample(
        n,
        d,
        QuasiMonteCarlo.LatinHypercubeSample(rng=rng),
        T,
    )
    return destination
end

function sample!(rng, destination::AbstractMatrix{T}, design::HaltonDesign, space) where {T}
    d, n = size(destination)
    d == dimension(space) || throw(DimensionMismatch("sample matrix and space differ in dimension"))
    n == 0 && return destination
    sequence = QuasiMonteCarlo.sample(
        n + design.skip,
        d,
        QuasiMonteCarlo.HaltonSample(
            R=QuasiMonteCarlo.Shift(rng=rng),
        ),
        T,
    )
    destination .= @view sequence[:, design.skip+1:end]
    return destination
end

function sample!(rng, destination::AbstractMatrix{T}, design::SobolDesign, space) where {T}
    d, n = size(destination)
    d == dimension(space) || throw(DimensionMismatch("sample matrix and space differ in dimension"))
    n == 0 && return destination
    sequence = QuasiMonteCarlo.sample(
        n + design.skip,
        d,
        QuasiMonteCarlo.SobolSample(
            R=QuasiMonteCarlo.Shift(rng=rng),
        ),
        T,
    )
    destination .= @view sequence[:, design.skip+1:end]
    return destination
end

function _sample_feasible!(rng, destination, sampler, problem; maximum_attempts=max(1000, 100size(destination, 2)))
    d, requested = size(destination)
    d == dimension(problem.space) || throw(DimensionMismatch("sample matrix and space differ in dimension"))
    requested == 0 && return destination
    buffer = similar(destination, d, min(max(requested, 8), 256))
    count = 0
    attempts = 0
    while count < requested && attempts < maximum_attempts
        sample!(rng, buffer, sampler, problem.space)
        for column in axes(buffer, 2)
            attempts += 1
            candidate = @view buffer[:, column]
            _canonicalize!(candidate, problem.space)
            point = decode(problem.space, candidate)
            if isfeasible(problem, point)
                count += 1
                destination[:, count] .= candidate
                count == requested && return destination
            end
            attempts >= maximum_attempts && break
        end
    end
    throw(ArgumentError("unable to sample $requested feasible points in $maximum_attempts attempts"))
end
