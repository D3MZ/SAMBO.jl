using CairoMakie
using Statistics
using Test

temporary = mktempdir()
ENV["SAMBO_PLOT_OUTPUT"] = temporary
include("generate_julia_plots.jl")

function luminance(pixel)
    return 0.2126Float64(getproperty(pixel, :r)) +
        0.7152Float64(getproperty(pixel, :g)) +
        0.0722Float64(getproperty(pixel, :b))
end

function resampled_luminance(image, rows, columns)
    row_indices = round.(Int, range(1, size(image, 1); length=rows))
    column_indices = round.(Int, range(1, size(image, 2); length=columns))
    output = Matrix{Float64}(undef, rows, columns)
    for column in axes(output, 2), row in axes(output, 1)
        output[row, column] =
            luminance(image[row_indices[row], column_indices[column]])
    end
    return output
end

function style_features(image)
    luminance = resampled_luminance(image, 256, 256)
    foreground = filter(<(0.97), vec(luminance))
    horizontal_edges = abs.(diff(luminance; dims=2))
    vertical_edges = abs.(diff(luminance; dims=1))
    return (
        aspect_ratio=size(image, 2) / size(image, 1),
        ink_fraction=length(foreground) / length(luminance),
        foreground_luminance=isempty(foreground) ? 1.0 : mean(foreground),
        edge_fraction=(
            count(>(0.12), horizontal_edges) +
            count(>(0.12), vertical_edges)
        ) / (length(horizontal_edges) + length(vertical_edges)),
    )
end

@testset "Python/Julia plot parity" begin
    for name in ("convergence", "regret", "objective", "evaluations")
        python = CairoMakie.load(
            joinpath(@__DIR__, "plots", "python", "$name.png"),
        )
        julia = CairoMakie.load(joinpath(temporary, "$name.png"))
        python_features = style_features(python)
        julia_features = style_features(julia)
        @info "plot parity" name python_features julia_features
        @test abs(
            python_features.aspect_ratio - julia_features.aspect_ratio,
        ) <= 0.15
        @test abs(
            python_features.ink_fraction - julia_features.ink_fraction,
        ) <= 0.15
        @test abs(
            python_features.foreground_luminance -
            julia_features.foreground_luminance,
        ) <= 0.2
        @test abs(
            python_features.edge_fraction - julia_features.edge_fraction,
        ) <= 0.08
    end
end
