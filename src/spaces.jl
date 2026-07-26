"A bounded continuous search dimension."
struct Continuous{T<:Real}
    lower::T
    upper::T
    function Continuous(lower::T, upper::T) where {T<:Real}
        isfinite(lower) && isfinite(upper) ||
            throw(ArgumentError("continuous bounds must be finite"))
        lower <= upper || throw(ArgumentError("lower must not exceed upper"))
        new{T}(lower, upper)
    end
end
Continuous(a::Real, b::Real) = Continuous(promote(a, b)...)

"An unordered categorical search dimension."
struct Choices{V<:Tuple}
    values::V
    function Choices(values::Tuple)
        isempty(values) && throw(ArgumentError("Choices cannot be empty"))
        allunique(values) || throw(ArgumentError("Choices must be unique"))
        new{typeof(values)}(values)
    end
end
Choices(values...) = Choices(values)

"A homogeneous, continuous hyperrectangle."
struct Box{T<:AbstractFloat,L<:AbstractVector{T},U<:AbstractVector{T}}
    lower::L
    upper::U
    function Box(
        lower::L,
        upper::U,
    ) where {T<:AbstractFloat,L<:AbstractVector{T},U<:AbstractVector{T}}
        length(lower) == length(upper) || throw(DimensionMismatch("bounds differ in length"))
        isempty(lower) && throw(ArgumentError("Box cannot be empty"))
        all(isfinite, lower) && all(isfinite, upper) ||
            throw(ArgumentError("box bounds must be finite"))
        all(i -> lower[i] <= upper[i], eachindex(lower, upper)) ||
            throw(ArgumentError("invalid bounds"))
        copied_lower = collect(lower)
        copied_upper = collect(upper)
        new{T,typeof(copied_lower),typeof(copied_upper)}(copied_lower, copied_upper)
    end
end
Box(lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real}) = begin
    length(lower) == length(upper) || throw(DimensionMismatch("bounds differ in length"))
    isempty(lower) && throw(ArgumentError("Box cannot be empty"))
    T = float(promote_type(eltype(lower), eltype(upper)))
    copied_lower = T.(collect(lower))
    copied_upper = T.(collect(upper))
    all(isfinite, copied_lower) && all(isfinite, copied_upper) ||
        throw(ArgumentError("box bounds must be finite"))
    all(i -> copied_lower[i] <= copied_upper[i], eachindex(copied_lower, copied_upper)) ||
        throw(ArgumentError("invalid bounds"))
    Box{T,typeof(copied_lower),typeof(copied_upper)}(copied_lower, copied_upper)
end
Box(bounds::AbstractVector{<:Tuple}) = Box(first.(bounds), last.(bounds))

"A heterogeneous search space. Keyword construction decodes points as named tuples."
struct SearchSpace{D<:Union{Tuple,NamedTuple}}
    dimensions::D
    function SearchSpace(dimensions::D) where {D<:Union{Tuple,NamedTuple}}
        isempty(dimensions) && throw(ArgumentError("SearchSpace cannot be empty"))
        for descriptor in dimensions
            descriptor isa Union{Continuous,AbstractRange,Choices} ||
                throw(ArgumentError("unsupported search-space dimension $(typeof(descriptor))"))
            descriptor isa AbstractRange && isempty(descriptor) &&
                throw(ArgumentError("search-space ranges cannot be empty"))
        end
        new{D}(dimensions)
    end
end
SearchSpace(dimensions...) = SearchSpace(dimensions)
SearchSpace(; kwargs...) = SearchSpace((; kwargs...))

dimension(space::Box) = length(space.lower)
dimension(space::SearchSpace) = length(space.dimensions)
latenttype(space::Box{T}) where {T} = T
latenttype(::SearchSpace) = Float64
dimensionnames(::Box) = nothing
dimensionnames(space::SearchSpace{<:NamedTuple}) = propertynames(space.dimensions)
dimensionnames(::SearchSpace) = nothing

@inline _decode(d::Continuous, z) = d.lower + z * (d.upper - d.lower)
@inline function _decode(d::AbstractRange, z)
    d[min(floor(Int, z * length(d)) + 1, length(d))]
end
@inline _decode(d::Choices, z) =
    d.values[min(floor(Int, z * length(d.values)) + 1, length(d.values))]

@inline function _encode(d::Continuous, x)
    d.lower <= x <= d.upper || throw(ArgumentError("value is outside its continuous dimension"))
    d.upper == d.lower ? 0.0 : (x - d.lower) / (d.upper - d.lower)
end
@inline _encode(d::AbstractRange, x) = _encode_discrete(d, x)
@inline _encode(d::Choices, x) = _encode_discrete(d.values, x)
function _encode_discrete(values::AbstractRange, x)
    x in values || throw(ArgumentError("value is not in its discrete dimension"))
    index = length(values) == 1 ? 1 :
        round(Int, (x - first(values)) / step(values)) + 1
    1 <= index <= length(values) && isequal(values[index], x) ||
        throw(ArgumentError("value is not in its discrete dimension"))
    return length(values) == 1 ? 0.0 : (index - 1) / (length(values) - 1)
end
function _encode_discrete(values, x)
    i = findfirst(isequal(x), values)
    isnothing(i) && throw(ArgumentError("value is not in its discrete dimension"))
    length(values) == 1 ? 0.0 : (i - 1) / (length(values) - 1)
end

function _checklatent(space, z)
    length(z) == dimension(space) || throw(DimensionMismatch("point and search space differ in dimension"))
    all(x -> 0 <= x <= 1, z) || throw(ArgumentError("latent coordinates must lie in [0, 1]"))
    return z
end

function decode(space::Box, z::AbstractVector)
    _checklatent(space, z)
    map((lower, upper, x) -> lower + x * (upper - lower), space.lower, space.upper, z)
end
@generated function _decodedimensions(dimensions::D, z) where {D<:Union{Tuple,NamedTuple}}
    count = D <: NamedTuple ? length(D.parameters[1]) : length(D.parameters)
    return Expr(
        :tuple,
        [:( _decode(getfield(dimensions, $index), z[$index]) ) for index in 1:count]...,
    )
end
function decode(space::SearchSpace{D}, z::AbstractVector) where {D<:Tuple}
    _checklatent(space, z)
    return _decodedimensions(space.dimensions, z)
end
function decode(
    space::SearchSpace{D},
    z::AbstractVector,
) where {Names,Types,D<:NamedTuple{Names,Types}}
    _checklatent(space, z)
    decoded = _decodedimensions(space.dimensions, z)
    return NamedTuple{Names}(decoded)
end
function encode(space::Box, point)
    length(point) == dimension(space) || throw(DimensionMismatch("point and search space differ in dimension"))
    map(space.lower, space.upper, point) do lower, upper, x
        lower <= x <= upper || throw(ArgumentError("point is outside the box"))
        upper == lower ? zero(eltype(space.lower)) : (x - lower) / (upper - lower)
    end
end
function encode(space::SearchSpace, point)
    names = dimensionnames(space)
    values = if point isa NamedTuple && !isnothing(names)
        all(name -> hasproperty(point, name), names) ||
            throw(ArgumentError("named point is missing a search-space dimension"))
        ntuple(i -> getproperty(point, names[i]), dimension(space))
    else
        Tuple(point)
    end
    length(values) == dimension(space) || throw(DimensionMismatch("point and search space differ in dimension"))
    [latenttype(space)(_encode(space.dimensions[i], values[i])) for i in 1:dimension(space)]
end
function encode!(destination, space, point)
    length(destination) == dimension(space) ||
        throw(DimensionMismatch("destination and search space differ in dimension"))
    destination .= encode(space, point)
    return destination
end
function project!(z, space=nothing)
    clamp!(z, zero(eltype(z)), one(eltype(z)))
    !isnothing(space) && _canonicalize!(z, space)
    return z
end
function _canonicalize!(z, space::Box)
    for index in eachindex(space.lower, space.upper)
        space.lower[index] == space.upper[index] && (z[index] = zero(eltype(z)))
    end
    return z
end
function _canonicalize!(z, space::SearchSpace)
    for index in 1:dimension(space)
        dimension = space.dimensions[index]
        if dimension isa Continuous && dimension.lower == dimension.upper
            z[index] = zero(eltype(z))
        elseif dimension isa AbstractRange || dimension isa Choices
            z[index] = _encode(dimension, _decode(dimension, z[index]))
        end
    end
    return z
end

function modelmatrix!(destination, space, points)
    size(destination) == size(points) ||
        throw(DimensionMismatch("model matrix and latent points differ in size"))
    destination .= points
    return destination
end
active_dimensions(space::Box) = findall(i -> space.lower[i] != space.upper[i], eachindex(space.lower))
function active_dimensions(space::SearchSpace)
    findall(1:dimension(space)) do i
        d = space.dimensions[i]
        d isa Continuous ? d.lower != d.upper :
        d isa AbstractRange ? length(d) > 1 :
        d isa Choices ? length(d.values) > 1 : true
    end
end

function space_cardinality(space::Box)
    return all(space.lower .== space.upper) ? 1 : nothing
end
function space_cardinality(space::SearchSpace)
    cardinality = 1
    for descriptor in space.dimensions
        count = if descriptor isa Continuous
            descriptor.lower == descriptor.upper ? 1 : return nothing
        elseif descriptor isa AbstractRange
            length(descriptor)
        else
            length(descriptor.values)
        end
        cardinality > typemax(Int) ÷ count && return nothing
        cardinality *= count
    end
    return cardinality
end
dimensionlabel(space, i) = _dimensionlabel(space, i)
dimensionticks(space, i) = _dimensionticks(space, i)

_dimensionlabel(::Box, i) = "x$i"
function _dimensionlabel(space::SearchSpace, i)
    names = dimensionnames(space)
    return isnothing(names) ? "x$i" : string(names[i])
end
function _dimensionticks(space::SearchSpace, i)
    d = space.dimensions[i]
    vals = d isa Choices ? d.values : d isa AbstractRange && length(d) <= 20 ? Tuple(d) : nothing
    isnothing(vals) ? nothing : (collect(range(0, 1; length=length(vals))), string.(collect(vals)))
end
_dimensionticks(::Box, i) = nothing
