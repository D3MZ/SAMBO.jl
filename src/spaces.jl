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

"An ordinal categorical search dimension encoded in the supplied value order."
struct Choices{V<:Tuple}
    values::V
    function Choices(values::Tuple)
        isempty(values) && throw(ArgumentError("Choices cannot be empty"))
        allunique(values) || throw(ArgumentError("Choices must be unique"))
        new{typeof(values)}(values)
    end
end
Choices(values::AbstractVector) = Choices(Tuple(values))
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
struct SearchSpace{T<:AbstractFloat,D<:Union{Tuple,NamedTuple}}
    dimensions::D
    function SearchSpace(
        ::Type{T},
        dimensions::D,
    ) where {T<:AbstractFloat,D<:Union{Tuple,NamedTuple}}
        isempty(dimensions) && throw(ArgumentError("SearchSpace cannot be empty"))
        for descriptor in dimensions
            descriptor isa Union{Continuous,AbstractRange,Choices} ||
                throw(ArgumentError("unsupported search-space dimension $(typeof(descriptor))"))
            descriptor isa AbstractRange && isempty(descriptor) &&
                throw(ArgumentError("search-space ranges cannot be empty"))
        end
        if dimensions isa NamedTuple
            reserved = (
                :evaluation,
                :source,
                :iteration,
                :objective,
                :elapsed_seconds,
            )
            collision = findfirst(name -> name in reserved, propertynames(dimensions))
            isnothing(collision) || throw(ArgumentError(
                "search-space dimension name $(propertynames(dimensions)[collision]) is reserved",
            ))
        end
        new{T,D}(dimensions)
    end
end
SearchSpace(::Type{T}, dimensions...) where {T<:AbstractFloat} =
    SearchSpace(T, dimensions)
SearchSpace(dimensions::D) where {D<:Union{Tuple,NamedTuple}} =
    SearchSpace(Float64, dimensions)
SearchSpace(dimensions...) = SearchSpace(Float64, dimensions)
SearchSpace(::Type{T}; kwargs...) where {T<:AbstractFloat} =
    SearchSpace(T, (; kwargs...))
SearchSpace(; kwargs...) = SearchSpace(Float64, (; kwargs...))

dimension(space::Box) = length(space.lower)
dimension(space::SearchSpace) = length(space.dimensions)
latenttype(space::Box{T}) where {T} = T
latenttype(::SearchSpace{T}) where {T} = T
dimensionnames(::Box) = nothing
_dimensionnames(dimensions::NamedTuple) = propertynames(dimensions)
_dimensionnames(dimensions::Tuple) = nothing
dimensionnames(space::SearchSpace) = _dimensionnames(space.dimensions)

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
function decode(space::SearchSpace{T,D}, z::AbstractVector) where {T,D<:Tuple}
    _checklatent(space, z)
    return _decodedimensions(space.dimensions, z)
end
function decode(
    space::SearchSpace{T,D},
    z::AbstractVector,
) where {T,Names,Types,D<:NamedTuple{Names,Types}}
    _checklatent(space, z)
    decoded = _decodedimensions(space.dimensions, z)
    return NamedTuple{Names}(decoded)
end
function encode!(destination, space::Box, point)
    length(destination) == dimension(space) ||
        throw(DimensionMismatch("destination and search space differ in dimension"))
    length(point) == dimension(space) || throw(DimensionMismatch("point and search space differ in dimension"))
    @inbounds for index in eachindex(destination, space.lower, space.upper, point)
        lower = space.lower[index]
        upper = space.upper[index]
        x = point[index]
        lower <= x <= upper || throw(ArgumentError("point is outside the box"))
        destination[index] = upper == lower ?
            zero(eltype(destination)) :
            (x - lower) / (upper - lower)
    end
    return destination
end

@inline _encode_dimensions!(destination, ::Tuple{}, values, index) =
    destination
@inline function _encode_dimensions!(
    destination,
    dimensions::Tuple,
    values,
    index,
)
    destination[index] = _encode(first(dimensions), values[index])
    return _encode_dimensions!(
        destination,
        Base.tail(dimensions),
        values,
        index + 1,
    )
end

function encode!(destination, space::SearchSpace, point)
    length(destination) == dimension(space) ||
        throw(DimensionMismatch("destination and search space differ in dimension"))
    names = dimensionnames(space)
    values = if point isa NamedTuple && !isnothing(names)
        all(name -> hasproperty(point, name), names) ||
            throw(ArgumentError("named point is missing a search-space dimension"))
        ntuple(i -> getproperty(point, names[i]), dimension(space))
    else
        Tuple(point)
    end
    length(values) == dimension(space) || throw(DimensionMismatch("point and search space differ in dimension"))
    return _encode_dimensions!(
        destination,
        Tuple(space.dimensions),
        values,
        1,
    )
end

function encode(space, point)
    destination =
        Vector{latenttype(space)}(undef, dimension(space))
    return encode!(destination, space, point)
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
@inline canonicalize_coordinate(dimension::Continuous, value) =
    ifelse(dimension.lower == dimension.upper, zero(value), value)
@inline canonicalize_coordinate(dimension::AbstractRange, value) =
    _encode(dimension, _decode(dimension, value))
@inline canonicalize_coordinate(dimension::Choices, value) =
    _encode(dimension, _decode(dimension, value))
@inline _canonicalize_dimensions!(z, ::Tuple{}, index) = z
@inline function _canonicalize_dimensions!(z, dimensions::Tuple, index)
    z[index] = canonicalize_coordinate(first(dimensions), z[index])
    return _canonicalize_dimensions!(z, Base.tail(dimensions), index + 1)
end
function _canonicalize!(z, space::SearchSpace)
    _canonicalize_dimensions!(z, Tuple(space.dimensions), 1)
    return z
end

function modelmatrix!(destination, space, points)
    size(destination) == size(points) ||
        throw(DimensionMismatch("model matrix and latent points differ in size"))
    destination .= points
    return destination
end
active_dimensions(space::Box) = findall(i -> space.lower[i] != space.upper[i], eachindex(space.lower))
_isactive(dimension::Continuous) = dimension.lower != dimension.upper
_isactive(dimension::AbstractRange) = length(dimension) > 1
_isactive(dimension::Choices) = length(dimension.values) > 1
function active_dimensions(space::SearchSpace)
    return findall(i -> _isactive(space.dimensions[i]), 1:dimension(space))
end

function space_cardinality(space::Box)
    return all(space.lower .== space.upper) ? 1 : nothing
end
_dimension_cardinality(dimension::Continuous) =
    dimension.lower == dimension.upper ? 1 : nothing
_dimension_cardinality(dimension::AbstractRange) = length(dimension)
_dimension_cardinality(dimension::Choices) = length(dimension.values)
function space_cardinality(space::SearchSpace)
    cardinality = 1
    for descriptor in space.dimensions
        count = _dimension_cardinality(descriptor)
        isnothing(count) && return nothing
        cardinality > typemax(Int) ÷ count && return nothing
        cardinality *= count
    end
    return cardinality
end

_canonical_level_index(dimension::Continuous, value) =
    dimension.lower == dimension.upper ? 1 :
    throw(ArgumentError("continuous search spaces do not have canonical indices"))
function _canonical_level_index(dimension::AbstractRange, value)
    index = findfirst(isequal(_decode(dimension, value)), dimension)
    isnothing(index) &&
        throw(ArgumentError("value is not in the discrete search dimension"))
    return index
end
function _canonical_level_index(dimension::Choices, value)
    index = findfirst(
        isequal(_decode(dimension, value)),
        dimension.values,
    )
    isnothing(index) &&
        throw(ArgumentError("categorical value is not in the search space"))
    return index
end

function canonical_index(space::Box, latent)
    space_cardinality(space) == 1 ||
        throw(ArgumentError("continuous boxes do not have canonical indices"))
    _checklatent(space, latent)
    return 1
end
function canonical_index(space::SearchSpace, latent)
    isnothing(space_cardinality(space)) &&
        throw(ArgumentError("continuous search spaces do not have canonical indices"))
    _checklatent(space, latent)
    index = 1
    stride = 1
    for axis in 1:dimension(space)
        descriptor = space.dimensions[axis]
        level = _canonical_level_index(descriptor, latent[axis])
        index += (level - 1) * stride
        stride *= _dimension_cardinality(descriptor)
    end
    return index
end

_canonical_level_value(::Continuous, level, count, ::Type{T}) where {T} =
    zero(T)
_canonical_level_value(
    ::Union{AbstractRange,Choices},
    level,
    count,
    ::Type{T},
) where {T} = count == 1 ? zero(T) : T(level - 1) / T(count - 1)

function canonical_latent!(destination, space::Box, index::Integer)
    space_cardinality(space) == 1 && index == 1 ||
        throw(BoundsError(space, index))
    fill!(destination, zero(eltype(destination)))
    return destination
end
function canonical_latent!(destination, space::SearchSpace, index::Integer)
    cardinality = space_cardinality(space)
    isnothing(cardinality) && throw(ArgumentError(
        "continuous search spaces do not have canonical indices",
    ))
    1 <= index <= cardinality || throw(BoundsError(space, index))
    remainder = index - 1
    for axis in 1:dimension(space)
        descriptor = space.dimensions[axis]
        count = _dimension_cardinality(descriptor)
        level = rem(remainder, count) + 1
        remainder = fld(remainder, count)
        destination[axis] = _canonical_level_value(
            descriptor,
            level,
            count,
            eltype(destination),
        )
    end
    return destination
end
dimensionlabel(space, i) = _dimensionlabel(space, i)
dimensionticks(space, i) = _dimensionticks(space, i)

_dimensionlabel(::Box, i) = "x$i"
function _dimensionlabel(space::SearchSpace, i)
    names = dimensionnames(space)
    return isnothing(names) ? "x$i" : string(names[i])
end
_dimensionticks(::Continuous) = nothing
_dimensionticks(dimension::Choices) = (
    collect(range(0, 1; length=length(dimension.values))),
    string.(collect(dimension.values)),
)
function _dimensionticks(dimension::AbstractRange)
    length(dimension) <= 20 || return nothing
    values = collect(dimension)
    return collect(range(0, 1; length=length(values))), string.(values)
end
_dimensionticks(space::SearchSpace, i) = _dimensionticks(space.dimensions[i])
_dimensionticks(::Box, i) = nothing
