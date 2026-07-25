"Return evaluation indices and the running best objective."
function convergencedata(result::Result)
    values = objectivevalues(result.trace)
    return (evaluations=collect(eachindex(values)), best=accumulate(min, values))
end

"Return cumulative regret relative to `optimum` (the observed minimum by default)."
function regretdata(result::Result; optimum=minimum(result))
    values = objectivevalues(result.trace)
    return (evaluations=collect(eachindex(values)), regret=cumsum(values .- optimum))
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
        order=collect(1:evaluation_count(result)),
        labels=[_dimensionlabel(result.space, i) for i in dims],
        ticks=[_dimensionticks(result.space, i) for i in dims],
    )
end

"Compute one- or two-dimensional RBF partial dependence from the result trace."
function partialdependence(
    result::Result;
    dimensions=(1,),
    resolution=40,
    samples=min(128, evaluation_count(result)),
    rng=Random.default_rng(),
)
    dims = Tuple(_checkdimensions(result.space, dimensions))
    length(dims) in (1, 2) || throw(ArgumentError("select one or two dimensions"))
    resolution > 1 || throw(ArgumentError("resolution must exceed one"))
    samples > 0 || throw(ArgumentError("samples must be positive and the trace must be nonempty"))

    grids = ntuple(_ -> collect(range(0, 1; length=resolution)), length(dims))
    base = rand(rng, dimension(result.space), samples)
    values = Array{Float64}(undef, ntuple(_ -> resolution, length(dims)))
    if length(dims) == 1
        queries = repeat(base, 1, resolution)
        for i in 1:resolution
            block = (i - 1) * samples + 1:i * samples
            queries[dims[1], block] .= grids[1][i]
        end
        predictions, _ = _rbf_predict(result.trace, queries)
        for i in 1:resolution
            block = (i - 1) * samples + 1:i * samples
            values[i] = mean(@view predictions[block])
        end
    else
        queries = repeat(base, 1, resolution^2)
        block_number = 0
        for j in 1:resolution, i in 1:resolution
            block_number += 1
            block = (block_number - 1) * samples + 1:block_number * samples
            queries[dims[1], block] .= grids[1][i]
            queries[dims[2], block] .= grids[2][j]
        end
        predictions, _ = _rbf_predict(result.trace, queries)
        block_number = 0
        for j in 1:resolution, i in 1:resolution
            block_number += 1
            block = (block_number - 1) * samples + 1:block_number * samples
            values[i, j] = mean(@view predictions[block])
        end
    end
    return (grids=grids, values=values, dimensions=dims)
end
