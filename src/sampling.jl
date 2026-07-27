abstract type AbstractDesign end
struct InfeasibleSpaceError <: Exception
    requested::Int
    attempts::Int
end
function Base.showerror(io::IO, error::InfeasibleSpaceError)
    print(
        io,
        "unable to sample ",
        error.requested,
        " feasible points in ",
        error.attempts,
        " attempts",
    )
end
struct UniformDesign <: AbstractDesign end
struct LatinHypercubeDesign <: AbstractDesign end
struct HaltonDesign <: AbstractDesign
    skip::Int
end
function HaltonDesign(; skip=20)
    skip >= 0 || throw(ArgumentError("Halton skip must be nonnegative"))
    return HaltonDesign(skip)
end
struct ScrambledHaltonDesign <: AbstractDesign
    skip::Int
    seed::UInt64
end
ScrambledHaltonDesign(skip::Int) = ScrambledHaltonDesign(skip, UInt64(0))
function ScrambledHaltonDesign(; skip=0, seed=0)
    skip >= 0 || throw(ArgumentError("scrambled Halton skip must be nonnegative"))
    seed isa Integer ||
        throw(ArgumentError("scrambled Halton seed must be an integer"))
    0 <= seed <= typemax(UInt64) ||
        throw(ArgumentError("scrambled Halton seed must fit in UInt64"))
    return ScrambledHaltonDesign(skip, UInt64(seed))
end
struct SobolDesign <: AbstractDesign
    skip::Int
end
advance(design::SobolDesign, count) =
    SobolDesign(skip=design.skip + count)
advance(design::HaltonDesign, count) =
    HaltonDesign(skip=design.skip + count)
advance(design::ScrambledHaltonDesign, count) =
    ScrambledHaltonDesign(skip=design.skip + count, seed=design.seed)
advance(design::UniformDesign, count) = design
advance(design::LatinHypercubeDesign, count) = design
function SobolDesign(; skip=0)
    skip >= 0 || throw(ArgumentError("Sobol skip must be nonnegative"))
    return SobolDesign(skip)
end

function _canonicalize_samples!(destination, space)
    for column in axes(destination, 2)
        _canonicalize!(@view(destination[:, column]), space)
    end
    return destination
end

function sample!(rng, destination::AbstractMatrix, ::UniformDesign, space)
    size(destination, 1) == dimension(space) ||
        throw(DimensionMismatch("sample matrix and space differ in dimension"))
    Random.rand!(rng, destination)
    return _canonicalize_samples!(destination, space)
end

function sample!(rng, destination::AbstractMatrix{T}, ::LatinHypercubeDesign, space) where {T}
    d, n = size(destination)
    d == dimension(space) || throw(DimensionMismatch("sample matrix and space differ in dimension"))
    n == 0 && return destination
    sequence = QuasiMonteCarlo.sample(
        n,
        d,
        QuasiMonteCarlo.LatinHypercubeSample(rng=rng),
        T,
    )::Matrix{T}
    copyto!(destination, sequence)
    return _canonicalize_samples!(destination, space)
end

function sample!(rng, destination::AbstractMatrix{T}, design::HaltonDesign, space) where {T}
    design.skip >= 0 || throw(ArgumentError("Halton skip must be nonnegative"))
    d, n = size(destination)
    d == dimension(space) || throw(DimensionMismatch("sample matrix and space differ in dimension"))
    n == 0 && return destination
    bases = QuasiMonteCarlo.nextprimes(1, d)
    for column in axes(destination, 2), axis in axes(destination, 1)
        index = design.skip + column
        inverse_base = inv(T(bases[axis]))
        factor = inverse_base
        value = zero(T)
        while index > 0
            index, digit = divrem(index, bases[axis])
            value += T(digit) * factor
            factor *= inverse_base
        end
        destination[axis, column] = value
    end
    return _canonicalize_samples!(destination, space)
end

function _scrambled_halton!(
    destination::AbstractMatrix{T},
    skip,
    seed::UInt64,
) where {T}
    d, n = size(destination)
    bases = QuasiMonteCarlo.nextprimes(1, d)
    for axis in 1:d
        base = bases[axis]
        digits = max(1, ceil(Int, 54 / log2(base)) - 1)
        generator = Random.Xoshiro(
            seed ⊻ (UInt64(axis) * 0x9e3779b97f4a7c15),
        )
        permutations = Matrix{Int}(undef, base, digits)
        for digit in 1:digits
            permutation = @view permutations[:, digit]
            permutation .= 0:base-1
            Random.shuffle!(generator, permutation)
        end
        for column in 1:n
            index = skip + column - 1
            factor = inv(T(base))
            value = zero(T)
            for digit in 1:digits
                index, remainder = divrem(index, base)
                value += T(permutations[remainder + 1, digit]) * factor
                factor /= T(base)
            end
            destination[axis, column] = value
        end
    end
    return destination
end

function sample!(
    rng,
    destination::AbstractMatrix{T},
    design::ScrambledHaltonDesign,
    space,
) where {T}
    size(destination, 1) == dimension(space) ||
        throw(DimensionMismatch("sample matrix and space differ in dimension"))
    _scrambled_halton!(destination, design.skip, design.seed)
    return _canonicalize_samples!(destination, space)
end

function sample!(rng, destination::AbstractMatrix{T}, design::SobolDesign, space) where {T}
    design.skip >= 0 || throw(ArgumentError("Sobol skip must be nonnegative"))
    d, n = size(destination)
    d == dimension(space) || throw(DimensionMismatch("sample matrix and space differ in dimension"))
    n == 0 && return destination
    sequence = QuasiMonteCarlo.Sobol.SobolSeq(d)
    first = @view destination[:, 1]
    QuasiMonteCarlo.Sobol.skip!(
        sequence,
        design.skip,
        first;
        exact=true,
    )
    for point in eachcol(destination)
        QuasiMonteCarlo.Sobol.next!(sequence, point)
    end
    return _canonicalize_samples!(destination, space)
end

@inline _isfeasible_latent(
    ::Problem{F,S,Unconstrained},
    latent,
) where {F,S} = true
@inline _isfeasible_latent(problem::Problem, latent) =
    isfeasible(problem, decode(problem.space, latent))

function _sample_feasible!(
    rng,
    destination,
    sampler,
    problem::Problem{F,S,Unconstrained};
    maximum_attempts=max(1000, 100size(destination, 2)),
) where {F,S}
    size(destination, 1) == dimension(problem.space) ||
        throw(DimensionMismatch("sample matrix and space differ in dimension"))
    return sample!(rng, destination, sampler, problem.space)
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
            if _isfeasible_latent(problem, candidate)
                count += 1
                destination[:, count] .= candidate
                count == requested && return destination
            end
            attempts >= maximum_attempts && break
        end
    end
    throw(InfeasibleSpaceError(requested, maximum_attempts))
end

function _finite_feasible_indices(problem; maximum_cardinality=100_000)
    cardinality = space_cardinality(problem.space)
    isnothing(cardinality) && return nothing
    cardinality <= maximum_cardinality || return nothing
    latent = Vector{latenttype(problem.space)}(
        undef,
        dimension(problem.space),
    )
    feasible = Int[]
    sizehint!(feasible, cardinality)
    for index in 1:cardinality
        canonical_latent!(latent, problem.space, index)
        isfeasible(problem, decode(problem.space, latent)) &&
            push!(feasible, index)
    end
    return feasible
end

function _finite_points(problem, indices)
    points = Matrix{latenttype(problem.space)}(
        undef,
        dimension(problem.space),
        length(indices),
    )
    for (column, index) in pairs(indices)
        canonical_latent!(
            @view(points[:, column]),
            problem.space,
            index,
        )
    end
    return points
end
