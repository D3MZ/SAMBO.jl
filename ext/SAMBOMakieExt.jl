module SAMBOMakieExt

using SAMBO
using Makie
using Random

const _PYTHON_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
)
const _MARKERS = (:circle, :rect, :xcross, :diamond, :dtriangle, :cross, :utriangle)
const _FACET_SIZE = 170
const _MATRIX_PANEL_SIZE = 130
const _MATRIX_COLUMN_STEP = 150
const _MATRIX_ROW_STEP = 156

_results(result::SAMBO.Result) = (result,)
_results(results) = Tuple(results)
function _common_sense(results)
    sense = first(results).sense
    all(result -> typeof(result.sense) === typeof(sense), results) ||
        throw(ArgumentError("plotting multiple results requires a common optimization sense"))
    return sense
end
_best_label(::SAMBO.Minimize) = "min f(x) after n evaluations"
_best_label(::SAMBO.Maximize) = "max f(x) after n evaluations"
_optimum_label(::SAMBO.Minimize) = "True minimum"
_optimum_label(::SAMBO.Maximize) = "True maximum"
_resultlabels(labels, count) = isnothing(labels) ? nothing : begin
    collected = collect(labels)
    length(collected) == count ||
        throw(DimensionMismatch("one label per result is required"))
    collected
end
_regret_label(::SAMBO.Minimize) =
    "Cumulative regret after n evaluations:  Σₜⁿ [f(xₜ) − fₒₚₜ]"
_regret_label(::SAMBO.Maximize) =
    "Cumulative regret after n evaluations:  Σₜⁿ [fₒₚₜ − f(xₜ)]"
_plotscale(::Val{:log}) = log10
_plotscale(::Val{:symlog}) = Makie.Symlog10(1)
_plotscale(::Val{:linear}) = identity
_plotscale(scale::Symbol) = _plotscale(Val(scale))
_plotscale(scale) = scale === identity ? identity : scale

function _trace_axis(position; title, xlabel, ylabel, xscale=identity, yscale=identity, axis=(;))
    defaults = (
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xscale=_plotscale(xscale),
        yscale=_plotscale(yscale),
        xgridvisible=true,
        ygridvisible=true,
        xgridcolor=(:gray, 0.45),
        ygridcolor=(:gray, 0.45),
        xgridwidth=0.8,
        ygridwidth=0.8,
        titlesize=16,
        titlefont=:regular,
        xlabelsize=14,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12,
    )
    return Axis(position; merge(defaults, axis)...)
end

function _trace_series!(axis, result, index, label, data)
    color = _PYTHON_COLORS[mod1(index, length(_PYTHON_COLORS))]
    marker = _MARKERS[mod1(index, length(_MARKERS))]
    lines!(
        axis,
        data.evaluations,
        data.values;
        color=(color, 0.7),
        linestyle=:dash,
        linewidth=2,
        label=label,
    )
    count = length(data.evaluations)
    stride = max(1, round(Int, count / 5))
    marker_indices = collect(1:stride:count)
    scatter!(
        axis,
        data.evaluations[marker_indices],
        data.values[marker_indices];
        color=(color, 0.7),
        marker=marker,
        markersize=8,
    )
end

function SAMBO.convergenceplot(
    results;
    labels=nothing,
    optimum=nothing,
    xscale=identity,
    yscale=identity,
    scale=yscale,
    axis=(;),
    figure=(;),
)
    result_tuple = _results(results)
    isempty(result_tuple) && throw(ArgumentError("at least one result is required"))
    sense = _common_sense(result_tuple)
    fig = Figure(; merge((size=(640, 480),), figure)...)
    ax = _trace_axis(
        fig[1, 1];
        title="Convergence",
        xlabel="Number of function evaluations n",
        ylabel=_best_label(sense),
        xscale,
        yscale=scale,
        axis,
    )
    SAMBO.convergenceplot!(ax, results; labels, optimum)
    return fig
end

function SAMBO.convergenceplot!(axis, results; labels=nothing, optimum=nothing)
    result_tuple = _results(results)
    isempty(result_tuple) && throw(ArgumentError("at least one result is required"))
    plot_labels = _resultlabels(labels, length(result_tuple))
    for (index, result) in pairs(result_tuple)
        convergence = SAMBO.convergencedata(result)
        label = isnothing(plot_labels) ?
            (length(result_tuple) > 1 ? "#$index" : nothing) : plot_labels[index]
        _trace_series!(
            axis,
            result,
            index,
            label,
            (evaluations=convergence.evaluations, values=convergence.best),
        )
    end
    if !isnothing(optimum)
        hlines!(
            axis,
            [optimum];
            color=:black,
            linestyle=:dash,
            linewidth=1,
            label=_optimum_label(_common_sense(result_tuple)),
        )
    end
    maximum_evaluation = maximum(
        SAMBO.evaluation_count(result) for result in result_tuple
    )
    tick_step = max(1, round(Int, maximum_evaluation / 9))
    axis.xticks = 0:tick_step:(maximum_evaluation + tick_step)
    (!isnothing(optimum) || !isnothing(plot_labels) || length(result_tuple) > 1) &&
        axislegend(axis; position=:rt)
    return axis
end

function SAMBO.regretplot(
    results;
    labels=nothing,
    optimum=nothing,
    xscale=identity,
    yscale=identity,
    scale=yscale,
    axis=(;),
    figure=(;),
)
    result_tuple = _results(results)
    isempty(result_tuple) && throw(ArgumentError("at least one result is required"))
    sense = _common_sense(result_tuple)
    fig = Figure(; merge((size=(640, 480),), figure)...)
    ax = _trace_axis(
        fig[1, 1];
        title="Cumulative regret",
        xlabel="Number of function evaluations n",
        ylabel=_regret_label(sense),
        xscale,
        yscale=scale,
        axis,
    )
    SAMBO.regretplot!(ax, results; labels, optimum)
    return fig
end

function SAMBO.regretplot!(axis, results; labels=nothing, optimum=nothing)
    result_tuple = _results(results)
    isempty(result_tuple) && throw(ArgumentError("at least one result is required"))
    sense = _common_sense(result_tuple)
    reference = isnothing(optimum) ?
        SAMBO.reference_value(sense, result_tuple) : optimum
    plot_labels = _resultlabels(labels, length(result_tuple))
    for (index, result) in pairs(result_tuple)
        regret = SAMBO.regretdata(result; optimum=reference)
        label = isnothing(plot_labels) ?
            (length(result_tuple) > 1 ? "#$index" : nothing) : plot_labels[index]
        _trace_series!(
            axis,
            result,
            index,
            label,
            (evaluations=regret.evaluations, values=regret.regret),
        )
    end
    maximum_evaluation = maximum(
        SAMBO.evaluation_count(result) for result in result_tuple
    )
    tick_step = max(1, round(Int, maximum_evaluation / 9))
    axis.xticks = 0:tick_step:(maximum_evaluation + tick_step)
    (!isnothing(plot_labels) || length(result_tuple) > 1) &&
        axislegend(axis; position=:rb)
    return axis
end

function _matrix_figure(title, count, figure)
    width = count == 1 ? _FACET_SIZE : _MATRIX_COLUMN_STEP * count + 60
    height = count == 1 ? _FACET_SIZE : _MATRIX_ROW_STEP * count + 80
    fig = Figure(; merge((size=(width, height),), figure)...)
    if count > 1
        Label(fig[0, 1:count], title; fontsize=14)
    end
    return fig, fig.layout
end

function _fill_matrix_grid!(grid, count)
    for index in 1:count
        rowsize!(grid, index, _MATRIX_PANEL_SIZE)
        colsize!(grid, index, _MATRIX_PANEL_SIZE)
    end
    rowgap!(grid, -10)
    colgap!(grid, -8)
    count > 1 && rowgap!(grid, 1, 8)
    return grid
end

function _dimensionlabels(space, dims, names)
    labels = isnothing(names) ?
        [SAMBO._dimensionlabel(space, dimension) for dimension in dims] :
        string.(collect(names))
    length(labels) == length(dims) ||
        throw(DimensionMismatch("one name per selected dimension is required"))
    return labels
end

function _matrix_axis(position, row, column, count, labels, axis)
    diagonal = row == column
    defaults = (
        aspect=AxisAspect(1),
        xlabel=diagonal || row == count ? labels[column] : "",
        ylabel=!diagonal && column == 1 ? labels[row] : "",
        xaxisposition=diagonal ? :top : :bottom,
        yaxisposition=diagonal ? :right : :left,
        xticklabelsvisible=diagonal || row == count,
        yticklabelsvisible=diagonal || column == 1,
        xticklabelrotation=π / 4,
        yticklabelrotation=diagonal ? π / 4 : 0,
        xlabelsize=10,
        ylabelsize=10,
        xticklabelsize=8,
        yticklabelsize=8,
        xticklabelspace=8.0,
        yticklabelspace=8.0,
        xlabelpadding=6,
        ylabelpadding=6,
        xgridvisible=false,
        ygridvisible=false,
    )
    return Axis(position[row, column]; merge(defaults, axis)...)
end

function _apply_dimension_ticks!(axis, space, xdimension, ydimension, diagonal)
    xticks = _plot_ticks(space, xdimension)
    !isnothing(xticks) && (axis.xticks = xticks)
    if !diagonal
        yticks = _plot_ticks(space, ydimension)
        !isnothing(yticks) && (axis.yticks = yticks)
    end
    return axis
end

_ticklabel(value) = string(round(value; sigdigits=4))
function _continuous_ticks(lower, upper)
    positions = collect(0.1:0.2:0.9)
    values = @. lower + positions * (upper - lower)
    return positions, _ticklabel.(values)
end
_plot_ticks(space::SAMBO.Box, dimension) =
    _continuous_ticks(space.lower[dimension], space.upper[dimension])
_plot_ticks(descriptor::SAMBO.Continuous) =
    _continuous_ticks(descriptor.lower, descriptor.upper)
_plot_ticks(descriptor::Union{AbstractRange,SAMBO.Choices}) =
    SAMBO._dimensionticks(descriptor)
_plot_ticks(space::SAMBO.SearchSpace, dimension) =
    _plot_ticks(space.dimensions[dimension])

function _jitter(values, space, dimension, amount, rng)
    _isdiscrete(space, dimension) || return values
    output = collect(values)
    @. output += amount * randn(rng)
    return output
end
_isdiscrete(::SAMBO.Box, dimension) = false
_isdiscrete(::SAMBO.Continuous) = false
_isdiscrete(::Union{AbstractRange,SAMBO.Choices}) = true
_isdiscrete(space::SAMBO.SearchSpace, dimension) =
    _isdiscrete(space.dimensions[dimension])
_discrete_count(::SAMBO.Box, dimension) = nothing
_discrete_count(::SAMBO.Continuous) = nothing
_discrete_count(descriptor::AbstractRange) = length(descriptor)
_discrete_count(descriptor::SAMBO.Choices) = length(descriptor.values)
_discrete_count(space::SAMBO.SearchSpace, dimension) =
    _discrete_count(space.dimensions[dimension])

function SAMBO.evaluationsplot(
    result::SAMBO.Result;
    dimensions=1:size(SAMBO.latentpoints(SAMBO.trace(result)), 1),
    observations=SAMBO.EvaluationsOnly(),
    names=nothing,
    bins=10,
    jitter=0.02,
    rng=Random.default_rng(),
    colormap=:summer,
    colorbar=false,
    axis=(;),
    figure=(;),
)
    dims = SAMBO._checkdimensions(result.space, dimensions)
    fig, grid = _matrix_figure(
        "Sequence & distribution of function evaluations",
        length(dims),
        figure,
    )
    SAMBO.evaluationsplot!(
        grid,
        result;
        dimensions=dims,
        observations,
        names,
        bins,
        jitter,
        rng,
        colormap,
        colorbar,
        axis,
    )
    return fig
end

function SAMBO.evaluationsplot!(
    position,
    result::SAMBO.Result;
    dimensions=1:size(SAMBO.latentpoints(SAMBO.trace(result)), 1),
    observations=SAMBO.EvaluationsOnly(),
    names=nothing,
    bins=10,
    jitter=0.02,
    rng=Random.default_rng(),
    colormap=:summer,
    colorbar=false,
    axis=(;),
)
    dims = SAMBO._checkdimensions(result.space, dimensions)
    data = SAMBO.evaluationsdata(result; dimensions=dims, observations)
    labels = _dimensionlabels(result.space, dims, names)
    count = length(dims)
    best = isempty(data.values) ? nothing :
        SAMBO.argbest(result.sense, data.values)
    diagonal_axes = Axis[]
    colorplot = nothing
    for row in 1:count, column in 1:row
        xdimension = dims[column]
        ydimension = dims[row]
        diagonal = row == column
        ax = _matrix_axis(position, row, column, count, labels, axis)
        if diagonal
            push!(diagonal_axes, ax)
            discrete_count = _discrete_count(result.space, xdimension)
            dimension_bins = isnothing(discrete_count) ? bins :
                clamp(discrete_count, 1, 50)
            hist!(
                ax,
                @view(data.latent[column, :]);
                bins=dimension_bins,
                color=_PYTHON_COLORS[1],
                strokewidth=0,
            )
            xlims!(ax, -0.05, 1.05)
        else
            xvalues = _jitter(
                @view(data.latent[column, :]),
                result.space,
                xdimension,
                jitter,
                rng,
            )
            yvalues = _jitter(
                @view(data.latent[row, :]),
                result.space,
                ydimension,
                jitter,
                rng,
            )
            colorplot = scatter!(
                ax,
                xvalues,
                yvalues;
                color=data.order,
                colormap,
                markersize=7,
                strokecolor=:black,
                strokewidth=0.5,
            )
            if !isnothing(best)
                scatter!(
                    ax,
                    data.latent[column, best],
                    data.latent[row, best];
                    marker=:star5,
                    color=(:red, 0.6),
                    strokecolor=:black,
                    strokewidth=0.5,
                    markersize=22,
                )
            end
            limits!(ax, -0.05, 1.05, -0.05, 1.05)
        end
        _apply_dimension_ticks!(ax, result.space, xdimension, ydimension, diagonal)
    end
    length(diagonal_axes) > 1 && linkyaxes!(diagonal_axes...)
    colorbar && !isnothing(colorplot) &&
        Colorbar(position[1:count, count + 1], colorplot; label="Evaluation")
    _fill_matrix_grid!(position, count)
    return position
end

function SAMBO.objectiveplot(
    result::SAMBO.Result;
    model=SAMBO.fittedmodel(result),
    dimensions=SAMBO.active_dimensions(result.space),
    names=nothing,
    optimum=nothing,
    resolution=16,
    samples=min(250, SAMBO.evaluation_count(result)),
    levels=10,
    plot_max_points=200,
    rng=Random.default_rng(),
    colormap=Reverse(:viridis),
    colorbar=false,
    axis=(;),
    figure=(;),
)
    dims = SAMBO._checkdimensions(result.space, dimensions)
    fig, grid = _matrix_figure("Partial dependence", length(dims), figure)
    SAMBO.objectiveplot!(
        grid,
        result;
        model,
        dimensions=dims,
        names,
        optimum,
        resolution,
        samples,
        levels,
        plot_max_points,
        rng,
        colormap,
        colorbar,
        axis,
    )
    return fig
end

function SAMBO.objectiveplot!(
    position,
    result::SAMBO.Result;
    model=SAMBO.fittedmodel(result),
    dimensions=SAMBO.active_dimensions(result.space),
    names=nothing,
    optimum=nothing,
    resolution=16,
    samples=min(250, SAMBO.evaluation_count(result)),
    levels=10,
    plot_max_points=200,
    rng=Random.default_rng(),
    colormap=Reverse(:viridis),
    colorbar=false,
    axis=(;),
)
    isnothing(model) && throw(ArgumentError(
        "objectiveplot requires a fitted model; pass `model=` for non-model-based results",
    ))
    plot_max_points >= 0 ||
        throw(ArgumentError("plot_max_points must be nonnegative"))
    dims = SAMBO._checkdimensions(result.space, dimensions)
    labels = _dimensionlabels(result.space, dims, names)
    count = length(dims)
    points = SAMBO.latentpoints(SAMBO.trace(result))
    TX = eltype(points)
    background = Matrix{TX}(
        undef,
        SAMBO.dimension(result.space),
        samples,
    )
    SAMBO._sample_feasible!(
        rng,
        background,
        SAMBO.UniformDesign(),
        result.problem,
    )
    best = SAMBO.argbest(result)
    optimum_latent = isnothing(optimum) ? @view(points[:, best]) :
        SAMBO.encode(result.space, optimum)
    point_count = min(plot_max_points, size(points, 2))
    point_indices = unique(round.(Int, range(1, size(points, 2); length=point_count)))
    filter!(index -> index != best, point_indices)
    dependences = Matrix{Any}(undef, count, count)
    contour_minimum = Inf
    contour_maximum = -Inf
    for row in 1:count, column in 1:row
        selected_dimensions = row == column ?
            (dims[column],) : (dims[column], dims[row])
        dependence = SAMBO.partialdependence(
            result;
            model,
            dimensions=selected_dimensions,
            resolution,
            samples,
            rng,
            background,
        )
        dependences[row, column] = dependence
        if row != column
            contour_minimum = min(contour_minimum, minimum(dependence.values))
            contour_maximum = max(contour_maximum, maximum(dependence.values))
        end
    end
    contour_range = if isfinite(contour_minimum)
        upper = contour_minimum < contour_maximum ?
            contour_maximum : contour_minimum + eps(float(contour_minimum))
        (contour_minimum, upper)
    else
        nothing
    end
    diagonal_axes = Axis[]
    colorplot = nothing
    for row in 1:count, column in 1:row
        xdimension = dims[column]
        ydimension = dims[row]
        diagonal = row == column
        ax = _matrix_axis(position, row, column, count, labels, axis)
        if diagonal
            push!(diagonal_axes, ax)
            dependence = dependences[row, column]
            lines!(
                ax,
                dependence.grids[1],
                dependence.values;
                color=_PYTHON_COLORS[1],
                linewidth=2,
            )
            vlines!(
                ax,
                [optimum_latent[xdimension]];
                linestyle=:dash,
                color=:red,
                linewidth=1,
            )
            xlims!(ax, -0.05, 1.05)
        else
            dependence = dependences[row, column]
            colorplot = contourf!(
                ax,
                dependence.grids[1],
                dependence.grids[2],
                dependence.values;
                levels=range(contour_range[1], contour_range[2]; length=levels),
                colormap=isempty(point_indices) ? colormap :
                    [(color, 0.8) for color in to_colormap(colormap)],
            )
            if !isempty(point_indices)
                scatter!(
                    ax,
                    @view(points[xdimension, point_indices]),
                    @view(points[ydimension, point_indices]);
                    color=(:black, 0.5),
                    markersize=4,
                    strokewidth=0,
                )
            end
            scatter!(
                ax,
                optimum_latent[xdimension],
                optimum_latent[ydimension];
                marker=:star5,
                color="#d00",
                strokecolor=:black,
                strokewidth=0.5,
                markersize=18,
            )
            limits!(ax, -0.05, 1.05, -0.05, 1.05)
        end
        _apply_dimension_ticks!(ax, result.space, xdimension, ydimension, diagonal)
    end
    length(diagonal_axes) > 1 && linkyaxes!(diagonal_axes...)
    colorbar && !isnothing(colorplot) &&
        Colorbar(position[1:count, count + 1], colorplot; label="Objective")
    _fill_matrix_grid!(position, count)
    return position
end

end
