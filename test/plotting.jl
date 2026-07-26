using CairoMakie

@testset "Makie extension" begin
    plot_result = SAMBO.minimize(
        x -> sum(abs2, x),
        SAMBO.Box([-1.0, -1.0], [1.0, 1.0]);
        algorithm=SAMBO.SMBO(candidate_pool=64),
        maximum_evaluations=10,
        rng=MersenneTwister(11),
    )
    figures = (
        SAMBO.convergenceplot(plot_result),
        SAMBO.regretplot(plot_result),
        SAMBO.evaluationsplot(plot_result),
        SAMBO.objectiveplot(plot_result; resolution=8, samples=8, rng=MersenneTwister(2)),
    )
    @test all(figure -> figure isa CairoMakie.Figure, figures)
    @test all(figure -> !isempty(figure.content), figures)
    mktempdir() do directory
        destination = joinpath(directory, "objective.svg")
        CairoMakie.save(destination, figures[end])
        @test filesize(destination) > 0
    end
end

@testset "Maximization plot semantics" begin
    maximized = SAMBO.minimize(
        x -> -(x[1] - 0.75)^2,
        SAMBO.Box([0.0], [1.0]);
        sense=SAMBO.Maximize(),
        algorithm=SAMBO.SMBO(candidate_pool=32),
        maximum_evaluations=8,
        rng=MersenneTwister(300),
    )
    convergence = SAMBO.convergenceplot(maximized; optimum=0.0)
    convergence_axis = only(filter(
        block -> block isa CairoMakie.Axis,
        convergence.content,
    ))
    @test convergence_axis.ylabel[] == "max f(x) after n evaluations"
    @test SAMBO.argbest(maximized) ==
        argmax(SAMBO.objectivevalues(SAMBO.trace(maximized)))

    minimized = SAMBO.minimize(
        x -> x[1]^2,
        SAMBO.Box([0.0], [1.0]);
        maximum_evaluations=4,
        rng=MersenneTwister(301),
    )
    @test_throws ArgumentError SAMBO.regretplot((maximized, minimized))
end

@testset "Matrix plot layout fuzz" begin
    function check_matrix_layout(figure, count)
        axes = filter(block -> block isa CairoMakie.Axis, figure.content)
        @test length(axes) == count * (count + 1) ÷ 2
        @test all(
            axis -> all(width -> 125 <= width <= 131, axis.scene.viewport[].widths),
            axes,
        )

        viewport = figure.scene.viewport[]
        @test all(axes) do axis
            box = axis.scene.viewport[]
            all(box.origin .>= 0) &&
                all(box.origin .+ box.widths .<= viewport.widths)
        end

        count == 1 && return
        firstaxis(row) = row * (row - 1) ÷ 2 + 1
        horizontal_gaps = Float64[]
        vertical_gaps = Float64[]
        for row in 2:count, column in 1:(row - 1)
            left = axes[firstaxis(row) + column - 1].scene.viewport[]
            right = axes[firstaxis(row) + column].scene.viewport[]
            push!(
                horizontal_gaps,
                right.origin[1] - (left.origin[1] + left.widths[1]),
            )
        end
        for row in 1:(count - 1)
            upper = axes[firstaxis(row)].scene.viewport[]
            lower = axes[firstaxis(row + 1)].scene.viewport[]
            push!(
                vertical_gaps,
                upper.origin[2] - (lower.origin[2] + lower.widths[2]),
            )
        end
        @test 0 <= minimum(horizontal_gaps) <= maximum(horizontal_gaps) <= 15
        @test 0 <= minimum(vertical_gaps) <= maximum(vertical_gaps) <= 30
    end

    for dimensions in shuffle(MersenneTwister(731), collect(1:8))
        space = SAMBO.Box(fill(-1.0, dimensions), fill(1.0, dimensions))
        result = SAMBO.minimize(
            x -> sum(abs2, x),
            space;
            algorithm=SAMBO.SMBO(candidate_pool=32),
            maximum_evaluations=max(8, 2dimensions + 1),
            rng=MersenneTwister(dimensions),
        )
        names = ["long_variable_name_$index" for index in 1:dimensions]
        evaluation_figure = SAMBO.evaluationsplot(
            result;
            names,
            rng=MersenneTwister(100 + dimensions),
        )
        objective_figure = SAMBO.objectiveplot(
            result;
            dimensions=1:dimensions,
            names,
            resolution=4,
            samples=4,
            rng=MersenneTwister(200 + dimensions),
        )
        check_matrix_layout(evaluation_figure, dimensions)
        check_matrix_layout(objective_figure, dimensions)
    end
end
