"A bounded continuous search dimension."
struct Continuous{T<:Real}
    lower::T
    upper::T
    function Continuous(lower::T, upper::T) where {T<:Real}
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
        new{typeof(values)}(values)
    end
end
Choices(values...) = Choices(values)

"A homogeneous, continuous hyperrectangle."
struct Box{T<:AbstractFloat,L<:AbstractVector{T},U<:AbstractVector{T}}
    lower::L
    upper::U
    function Box(lower::L, upper::U) where {T<:AbstractFloat,L<:AbstractVector{T},U<:AbstractVector{T}}
        length(lower) == length(upper) || throw(DimensionMismatch("bounds differ in length"))
        all(i -> lower[i] <= upper[i], eachindex(lower, upper)) || throw(ArgumentError("invalid bounds"))
        new{T,L,U}(lower, upper)
    end
end
Box(lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real}) = begin
    T = float(promote_type(eltype(lower), eltype(upper)))
    Box(T.(lower), T.(upper))
end
Box(bounds::AbstractVector{<:Tuple}) = Box(first.(bounds), last.(bounds))

"A heterogeneous search space. Keyword construction decodes points as named tuples."
struct SearchSpace{N,D<:Tuple}
    names::N
    dimensions::D
    function SearchSpace(names::N, dimensions::D) where {N,D<:Tuple}
        if !isnothing(names)
            names isa Tuple{Vararg{Symbol}} || throw(ArgumentError("dimension names must be symbols"))
            length(names) == length(dimensions) || throw(DimensionMismatch("names and dimensions differ in length"))
            allunique(names) || throw(ArgumentError("dimension names must be unique"))
        end
        any(d -> d isa AbstractRange && isempty(d), dimensions) &&
            throw(ArgumentError("search-space ranges cannot be empty"))
        new{N,D}(names, dimensions)
    end
end
SearchSpace(dimensions::Tuple) = SearchSpace(nothing, dimensions)
SearchSpace(dimensions...) = SearchSpace(dimensions)
SearchSpace(; kwargs...) = SearchSpace(Tuple(keys(kwargs)), Tuple(values(kwargs)))

dimension(space::Box) = length(space.lower)
dimension(space::SearchSpace) = length(space.dimensions)

@inline _decode(d::Continuous, z) = d.lower + z * (d.upper - d.lower)
@inline function _decode(d::AbstractRange, z)
    d[clamp(round(Int, 1 + z * (length(d) - 1)), 1, length(d))]
end
@inline _decode(d::Choices, z) = d.values[clamp(round(Int, 1 + z * (length(d.values) - 1)), 1, length(d.values))]

@inline function _encode(d::Continuous, x)
    d.lower <= x <= d.upper || throw(ArgumentError("value is outside its continuous dimension"))
    d.upper == d.lower ? 0.0 : (x - d.lower) / (d.upper - d.lower)
end
@inline _encode(d::AbstractRange, x) = _encode_discrete(d, x)
@inline _encode(d::Choices, x) = _encode_discrete(d.values, x)
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
function decode(space::SearchSpace, z::AbstractVector)
    _checklatent(space, z)
    values = ntuple(i -> _decode(space.dimensions[i], z[i]), dimension(space))
    isnothing(space.names) ? values : NamedTuple{space.names}(values)
end
function encode(space::Box, point)
    length(point) == dimension(space) || throw(DimensionMismatch("point and search space differ in dimension"))
    map(space.lower, space.upper, point) do lower, upper, x
        lower <= x <= upper || throw(ArgumentError("point is outside the box"))
        upper == lower ? zero(eltype(space.lower)) : (x - lower) / (upper - lower)
    end
end
function encode(space::SearchSpace, point)
    values = if point isa NamedTuple && !isnothing(space.names)
        all(name -> hasproperty(point, name), space.names) ||
            throw(ArgumentError("named point is missing a search-space dimension"))
        ntuple(i -> getproperty(point, space.names[i]), dimension(space))
    else
        Tuple(point)
    end
    length(values) == dimension(space) || throw(DimensionMismatch("point and search space differ in dimension"))
    [Float64(_encode(space.dimensions[i], values[i])) for i in eachindex(space.dimensions)]
end
function project!(z)
    clamp!(z, zero(eltype(z)), one(eltype(z)))
    return z
end

_dimensionlabel(::Box, i) = "x$i"
_dimensionlabel(space::SearchSpace, i) = isnothing(space.names) ? "x$i" : string(space.names[i])
function _dimensionticks(space::SearchSpace, i)
    d = space.dimensions[i]
    vals = d isa Choices ? d.values : d isa AbstractRange && length(d) <= 20 ? Tuple(d) : nothing
    isnothing(vals) ? nothing : (collect(range(0, 1; length=length(vals))), string.(collect(vals)))
end
_dimensionticks(::Box, i) = nothing
