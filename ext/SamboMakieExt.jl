module SamboMakieExt

using Sambo
using Makie
using Random

_results(result::Sambo.Result) = (result,)
_results(results) = Tuple(results)

function _labels(labels, n)
    isnothing(labels) && return nothing
    collected = collect(labels)
    length(collected) == n || throw(DimensionMismatch("one label per result is required"))
    return collected
end

function Sambo.convergenceplot(results; labels=nothing, optimum=nothing, axis=(;), figure=(;))
    fig = Figure(; figure...)
    ax = Axis(fig[1, 1]; xlabel="Evaluations", ylabel="Best objective", axis...)
    Sambo.convergenceplot!(ax, results; labels, optimum)
    return fig
end

function Sambo.convergenceplot!(ax, results; labels=nothing, optimum=nothing)
    result_tuple = _results(results)
    plot_labels = _labels(labels, length(result_tuple))
    for (i, result) in pairs(result_tuple)
        data = Sambo.convergencedata(result)
        label = isnothing(plot_labels) ? nothing : plot_labels[i]
        lines!(ax, data.evaluations, data.best; label)
    end
    !isnothing(optimum) && hlines!(ax, [optimum]; linestyle=:dash, color=:black)
    !isnothing(plot_labels) && axislegend(ax)
    return ax
end

function Sambo.regretplot(results; labels=nothing, optimum=nothing, axis=(;), figure=(;))
    fig = Figure(; figure...)
    ax = Axis(fig[1, 1]; xlabel="Evaluations", ylabel="Cumulative regret", axis...)
    Sambo.regretplot!(ax, results; labels, optimum)
    return fig
end

function Sambo.regretplot!(ax, results; labels=nothing, optimum=nothing)
    result_tuple = _results(results)
    isempty(result_tuple) && throw(ArgumentError("at least one result is required"))
    reference = isnothing(optimum) ? minimum(Sambo.minimum, result_tuple) : optimum
    plot_labels = _labels(labels, length(result_tuple))
    for (i, result) in pairs(result_tuple)
        data = Sambo.regretdata(result; optimum=reference)
        label = isnothing(plot_labels) ? nothing : plot_labels[i]
        lines!(ax, data.evaluations, data.regret; label)
    end
    !isnothing(plot_labels) && axislegend(ax)
    return ax
end

function Sambo.evaluationsplot(
    result::Sambo.Result;
    dimensions=1:size(Sambo.latentpoints(Sambo.trace(result)), 1),
    axis=(;),
    figure=(;),
)
    fig = Figure(; figure...)
    Sambo.evaluationsplot!(fig, result; dimensions, axis)
    return fig
end

function Sambo.evaluationsplot!(position, result::Sambo.Result; dimensions=1:size(Sambo.latentpoints(Sambo.trace(result)), 1), axis=(;))
    data = Sambo.evaluationsdata(result; dimensions)
    n = length(data.labels)
    for row in 1:n, column in 1:row
        ax = Axis(
            position[row, column];
            xlabel=data.labels[column],
            ylabel=row == column ? "Count" : data.labels[row],
            axis...,
        )
        if row == column
            hist!(ax, @view data.latent[column, :])
        else
            scatter!(
                ax,
                @view(data.latent[column, :]),
                @view(data.latent[row, :]);
                color=data.order,
                colormap=:viridis,
            )
        end
        column_ticks = data.ticks[column]
        !isnothing(column_ticks) && (ax.xticks = column_ticks)
        if row != column
            row_ticks = data.ticks[row]
            !isnothing(row_ticks) && (ax.yticks = row_ticks)
        end
    end
    return position
end

function Sambo.objectiveplot(
    result::Sambo.Result;
    dimensions=1:min(3, size(Sambo.latentpoints(Sambo.trace(result)), 1)),
    resolution=30,
    samples=min(128, Sambo.evaluation_count(result)),
    rng=Random.default_rng(),
    axis=(;),
    figure=(;),
)
    fig = Figure(; figure...)
    Sambo.objectiveplot!(fig, result; dimensions, resolution, samples, rng, axis)
    return fig
end

function Sambo.objectiveplot!(
    position,
    result::Sambo.Result;
    dimensions=1:min(3, size(Sambo.latentpoints(Sambo.trace(result)), 1)),
    resolution=30,
    samples=min(128, Sambo.evaluation_count(result)),
    rng=Random.default_rng(),
    axis=(;),
)
    dims = Sambo._checkdimensions(result.space, dimensions)
    n = length(dims)
    for row in 1:n, column in 1:row
        xdimension = dims[column]
        ydimension = dims[row]
        ax = Axis(
            position[row, column];
            xlabel=Sambo._dimensionlabel(result.space, xdimension),
            ylabel=row == column ? "Objective" : Sambo._dimensionlabel(result.space, ydimension),
            axis...,
        )
        if row == column
            dependence = Sambo.partialdependence(
                result; dimensions=(xdimension,), resolution, samples, rng,
            )
            lines!(ax, dependence.grids[1], dependence.values)
        else
            dependence = Sambo.partialdependence(
                result; dimensions=(xdimension, ydimension), resolution, samples, rng,
            )
            contourf!(ax, dependence.grids[1], dependence.grids[2], dependence.values)
        end
        xticks = Sambo._dimensionticks(result.space, xdimension)
        !isnothing(xticks) && (ax.xticks = xticks)
        if row != column
            yticks = Sambo._dimensionticks(result.space, ydimension)
            !isnothing(yticks) && (ax.yticks = yticks)
        end
    end
    return position
end

end
