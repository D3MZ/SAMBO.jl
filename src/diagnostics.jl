"Return evaluation indices and the running best objective."
function convergencedata(result::Result)
    values = objectivevalues(result.trace)
    return (
        evaluations=collect(@view result.trace.evaluation_numbers[1:result.trace.count]),
        best=result.sense isa Minimize ?
            accumulate(min, values) : accumulate(max, values),
    )
end

"Return cumulative regret relative to `optimum` (the observed minimum by default)."
function regretdata(result::Result; optimum=minimum(result))
    values = objectivevalues(result.trace)
    return (
        evaluations=collect(@view result.trace.evaluation_numbers[1:result.trace.count]),
        regret=cumsum(
            result.sense isa Minimize ? values .- optimum : optimum .- values,
        ),
    )
end

function _checkdimensions(space, dimensions)
    dims = collect(Int, dimensions)
    isempty(dims) && throw(ArgumentError("select at least one dimension"))
    allunique(dims) || throw(ArgumentError("dimensions must be unique"))
    all(i -> 1 <= i <= dimension(space), dims) ||
        throw(ArgumentError("dimension index is outside the search space"))
    return dims
end

"Return decoded observations and latent coordinates for evaluation-matrix plots."
function evaluationsdata(result::Result; dimensions=1:dimension(result.space))
    dims = _checkdimensions(result.space, dimensions)
    return (
        latent=Matrix(latentpoints(result.trace)[dims, :]),
        values=collect(objectivevalues(result.trace)),
        order=collect(@view result.trace.evaluation_numbers[1:result.trace.count]),
        labels=[_dimensionlabel(result.space, i) for i in dims],
        ticks=[_dimensionticks(result.space, i) for i in dims],
    )
end

"Compute one- or two-dimensional partial dependence from a fitted surrogate."
function partialdependence(
    result::Result;
    model=fittedmodel(result),
    dimensions=(1,),
    resolution=40,
    samples=min(128, evaluation_count(result)),
    rng=Random.default_rng(),
    background=nothing,
)
    dims = Tuple(_checkdimensions(result.space, dimensions))
    length(dims) in (1, 2) || throw(ArgumentError("select one or two dimensions"))
    resolution > 1 || throw(ArgumentError("resolution must exceed one"))
    samples > 0 || throw(ArgumentError("samples must be positive and the trace must be nonempty"))
    isnothing(model) && throw(ArgumentError(
        "partialdependence requires an explicit fitted model",
    ))

    TX = eltype(result.trace.latent_points)
    TY = eltype(result.trace.objective_values)
    grids = ntuple(length(dims)) do index
        grid = collect(range(zero(TX), one(TX); length=resolution))
        for value_index in eachindex(grid)
            coordinate = zeros(TX, dimension(result.space))
            coordinate[dims[index]] = grid[value_index]
            _canonicalize!(coordinate, result.space)
            grid[value_index] = coordinate[dims[index]]
        end
        unique!(grid)
    end
    base = if isnothing(background)
        generated = Matrix{TX}(
            undef,
            dimension(result.space),
            samples,
        )
        _sample_feasible!(
            rng,
            generated,
            UniformDesign(),
            result.problem,
        )
    else
        size(background) == (dimension(result.space), samples) ||
            throw(DimensionMismatch("background must be dimensions × samples"))
        generated = Matrix{TX}(background)
        _canonicalize_samples!(generated, result.space)
    end
    grid_sizes = length.(grids)
    values = Array{TY}(undef, grid_sizes...)
    queries = copy(base)
    predictions = Vector{TY}(undef, samples)
    if length(dims) == 1
        for i in eachindex(grids[1])
            queries .= base
            queries[dims[1], :] .= grids[1][i]
            predictmean!(predictions, model, queries)
            values[i] = _loss(result.sense, mean(predictions))
        end
    else
        for j in eachindex(grids[2]), i in eachindex(grids[1])
            queries .= base
            queries[dims[1], :] .= grids[1][i]
            queries[dims[2], :] .= grids[2][j]
            predictmean!(predictions, model, queries)
            values[i, j] = _loss(result.sense, mean(predictions))
        end
    end
    return (grids=grids, values=values, dimensions=dims)
end
