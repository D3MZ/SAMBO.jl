struct EvaluationsOnly end
struct IncludeKnownObservations end
struct FeasibleConditionalDependence end
struct UnconstrainedModelDependence end

mutable struct PartialDependenceWorkspace{TX,TY}
    queries::Matrix{TX}
    feasible_queries::Matrix{TX}
    predictions::Vector{TY}
end
PartialDependenceWorkspace(::Type{TX}, ::Type{TY}, dimensions, samples) where {TX,TY} =
    PartialDependenceWorkspace(
        Matrix{TX}(undef, dimensions, samples),
        Matrix{TX}(undef, dimensions, samples),
        Vector{TY}(undef, samples),
    )

function _ensure_partial_dependence_workspace!(
    workspace::PartialDependenceWorkspace,
    dimensions,
    samples,
)
    if size(workspace.queries, 1) != dimensions ||
            size(workspace.queries, 2) < samples
        workspace.queries = Matrix{eltype(workspace.queries)}(
            undef,
            dimensions,
            samples,
        )
        workspace.feasible_queries = similar(workspace.queries)
    end
    length(workspace.predictions) < samples &&
        resize!(workspace.predictions, samples)
    return workspace
end

_diagnostic_indices(trace, ::IncludeKnownObservations) = 1:trace.count
_diagnostic_indices(trace, ::EvaluationsOnly) =
    findall(!=(KnownObservation), @view trace.source[1:trace.count])
_running_best(::Minimize, values) = accumulate(min, values)
_running_best(::Maximize, values) = accumulate(max, values)
_regret(::Minimize, values, optimum) = values .- optimum
_regret(::Maximize, values, optimum) = optimum .- values

"Return evaluation indices and the running best objective."
function convergencedata(
    result::Result;
    observations=IncludeKnownObservations(),
)
    indices = _diagnostic_indices(result.trace, observations)
    values = objectivevalues(result.trace)[indices]
    return (
        evaluations=collect(result.trace.evaluation_numbers[indices]),
        best=_running_best(result.sense, values),
    )
end

"Return cumulative regret relative to `optimum` (the observed optimum by default)."
function regretdata(
    result::Result;
    optimum=minimum(result),
    observations=EvaluationsOnly(),
)
    indices = _diagnostic_indices(result.trace, observations)
    values = objectivevalues(result.trace)[indices]
    return (
        evaluations=collect(result.trace.evaluation_numbers[indices]),
        regret=cumsum(_regret(result.sense, values, optimum)),
    )
end

function _canonical_grid(
    space::Box,
    dimension_index,
    resolution,
    ::Type{T},
) where {T}
    return space.lower[dimension_index] == space.upper[dimension_index] ?
        T[zero(T)] :
        collect(range(zero(T), one(T); length=resolution))
end
function _canonical_grid(
    space::SearchSpace,
    dimension_index,
    resolution,
    ::Type{T},
) where {T}
    descriptor = space.dimensions[dimension_index]
    return _canonical_grid(descriptor, resolution, T)
end
function _canonical_grid(descriptor::Continuous, resolution, ::Type{T}) where {T}
    return descriptor.lower == descriptor.upper ?
        T[zero(T)] :
        collect(range(zero(T), one(T); length=resolution))
end
function _canonical_grid(descriptor::AbstractRange, resolution, ::Type{T}) where {T}
    count = length(descriptor)
    return count == 1 ?
        T[zero(T)] :
        collect(range(zero(T), one(T); length=count))
end
function _canonical_grid(descriptor::Choices, resolution, ::Type{T}) where {T}
    count = length(descriptor.values)
    return count == 1 ?
        T[zero(T)] :
        collect(range(zero(T), one(T); length=count))
end

_feasible_columns(result, queries, ::UnconstrainedModelDependence) =
    axes(queries, 2)
function _feasible_columns(result, queries, ::FeasibleConditionalDependence)
    return findall(axes(queries, 2)) do column
        isfeasible(
            result.problem,
            decode(result.space, @view queries[:, column]),
        )
    end
end

function _partial_mean!(
    predictions,
    feasible_queries,
    result,
    model,
    queries,
    policy,
)
    columns = _feasible_columns(result, queries, policy)
    isempty(columns) && return eltype(predictions)(NaN)
    selected = if length(columns) == size(queries, 2)
        queries
    else
        compact = @view feasible_queries[:, 1:length(columns)]
        for (destination, source) in enumerate(columns)
            copyto!(@view(compact[:, destination]), @view(queries[:, source]))
        end
        compact
    end
    output = @view predictions[1:size(selected, 2)]
    predictmean!(output, model, selected)
    return _loss(result.sense, mean(output))
end

function _partial_dependence!(
    values,
    result,
    model,
    dimensions::NTuple{1,Int},
    grids,
    base,
    workspace,
    policy,
)
    queries = @view workspace.queries[:, 1:size(base, 2)]
    predictions = workspace.predictions
    for index in eachindex(grids[1])
        copyto!(queries, base)
        queries[dimensions[1], :] .= grids[1][index]
        values[index] = _partial_mean!(
            predictions,
            workspace.feasible_queries,
            result,
            model,
            queries,
            policy,
        )
    end
    return values
end

function _partial_dependence!(
    values,
    result,
    model,
    dimensions::NTuple{2,Int},
    grids,
    base,
    workspace,
    policy,
)
    queries = @view workspace.queries[:, 1:size(base, 2)]
    predictions = workspace.predictions
    for column in eachindex(grids[2]), row in eachindex(grids[1])
        copyto!(queries, base)
        queries[dimensions[1], :] .= grids[1][row]
        queries[dimensions[2], :] .= grids[2][column]
        values[row, column] = _partial_mean!(
            predictions,
            workspace.feasible_queries,
            result,
            model,
            queries,
            policy,
        )
    end
    return values
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
function evaluationsdata(
    result::Result;
    dimensions=1:dimension(result.space),
    observations=EvaluationsOnly(),
)
    dims = _checkdimensions(result.space, dimensions)
    indices = _diagnostic_indices(result.trace, observations)
    return (
        latent=Matrix(latentpoints(result.trace)[dims, indices]),
        values=collect(objectivevalues(result.trace)[indices]),
        order=collect(result.trace.evaluation_numbers[indices]),
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
    dependence=FeasibleConditionalDependence(),
    workspace=nothing,
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
    grids = ntuple(
        index -> _canonical_grid(result.space, dims[index], resolution, TX),
        length(dims),
    )
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
        for column in axes(generated, 2)
            _checklatent(result.space, @view generated[:, column])
        end
        _canonicalize_samples!(generated, result.space)
    end
    grid_sizes = length.(grids)
    values = Array{TY}(undef, grid_sizes...)
    pd_workspace = isnothing(workspace) ?
        PartialDependenceWorkspace(TX, TY, dimension(result.space), samples) :
        _ensure_partial_dependence_workspace!(
            workspace,
            dimension(result.space),
            samples,
        )
    _partial_dependence!(
        values,
        result,
        model,
        dims,
        grids,
        base,
        pd_workspace,
        dependence,
    )
    return (grids=grids, values=values, dimensions=dims)
end
