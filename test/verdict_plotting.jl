using Test
using Random
using SAMBO
import CairoMakie

struct CoordinateLoss end
function SAMBO.predictmean!(output, ::CoordinateLoss, points)
    @. output = -points[1, :]
    return output
end

@testset "verdict plotting and diagnostic semantics" begin
    makie_extension = Base.get_extension(SAMBO, :SAMBOMakieExt)

    @test makie_extension._plotscale(:log) === log10
    @test makie_extension._plotscale(:symlog) !== log10

    maximized = minimize(
        x -> x[1],
        SAMBO.Box([0.0], [1.0]);
        sense=Maximize(),
        algorithm=SMBO(candidate_pool=32),
        maximum_evaluations=8,
        rng=MersenneTwister(701),
    )
    @test SAMBO.argbest(maximized) ==
        argmax(objectivevalues(trace(maximized)))

    convergence = convergenceplot(maximized; optimum=1.0)
    convergence_axis = only(filter(
        block -> block isa CairoMakie.Axis,
        convergence.content,
    ))
    @test convergence_axis.ylabel[] == "max f(x) after n evaluations"

    regret = regretplot(maximized; optimum=1.0)
    regret_axis = only(filter(
        block -> block isa CairoMakie.Axis,
        regret.content,
    ))
    @test occursin("fₒₚₜ − f(xₜ)", regret_axis.ylabel[])

    known = minimize(
        x -> x[1]^2,
        SAMBO.Box([0.0], [1.0]);
        initial_points=[[0.25]],
        initial_values=[0.0625],
        maximum_evaluations=3,
        rng=MersenneTwister(702),
    )
    convergence_known = convergencedata(known)
    regret_known = regretdata(known)
    @test first(convergence_known.evaluations) == 0
    @test all(>(0), regret_known.evaluations)
    @test all(>(0), evaluationsdata(known).order)
    @test first(evaluationsdata(
        known;
        observations=IncludeKnownObservations(),
    ).order) == 0

    maximized_dependence = partialdependence(
        maximized;
        model=CoordinateLoss(),
        dimensions=(1,),
        resolution=5,
        samples=4,
        background=fill(0.5, 1, 4),
    )
    @test maximized_dependence.values ≈ maximized_dependence.grids[1]
    native_maximized_dependence = partialdependence(
        maximized;
        dimensions=(1,),
        resolution=5,
        samples=4,
        background=fill(0.5, 1, 4),
    )
    @test native_maximized_dependence.values[end] >
        native_maximized_dependence.values[1]

    mixed = minimize(
        x -> x.x + (x.kind == :b),
        SearchSpace(x=Continuous(0.0, 1.0), kind=Choices(:a, :b));
        algorithm=SMBO(candidate_pool=32),
        maximum_evaluations=8,
        rng=MersenneTwister(703),
    )
    mixed_dependence = partialdependence(
        mixed;
        dimensions=(1, 2),
        resolution=5,
        samples=4,
        rng=MersenneTwister(704),
    )
    @test size(mixed_dependence.values) == (5, 2)
    @test length(mixed_dependence.grids[2]) == 2

    constrained = minimize(
        x -> x[1]^2,
        SAMBO.Box([0.0], [1.0]);
        constraint=x -> x[1] - 0.5,
        algorithm=SMBO(candidate_pool=32),
        maximum_evaluations=8,
        rng=MersenneTwister(705),
    )
    conditional = partialdependence(
        constrained;
        dimensions=(1,),
        resolution=5,
        samples=4,
        background=fill(0.25, 1, 4),
    )
    @test all(isfinite, conditional.values[1:3])
    @test all(isnan, conditional.values[4:5])

    fixed = minimize(
        x -> sum(abs2, x),
        SAMBO.Box([0.0, 0.0], [0.0, 1.0]);
        algorithm=SMBO(candidate_pool=32),
        maximum_evaluations=8,
        rng=MersenneTwister(706),
    )
    fixed_dependence = partialdependence(
        fixed;
        dimensions=(1,),
        resolution=8,
        samples=4,
        rng=MersenneTwister(707),
    )
    @test fixed_dependence.grids[1] == [0.0]
    @test size(fixed_dependence.values) == (1,)

    @test_throws ArgumentError objectiveplot(
        maximized;
        resolution=5,
        samples=4,
        plot_max_points=-1,
    )
    one_dimensional = objectiveplot(
        maximized;
        resolution=5,
        samples=4,
        colorbar=true,
        rng=MersenneTwister(708),
    )
    @test count(block -> block isa CairoMakie.Axis, one_dimensional.content) == 1
    @test !any(block -> block isa CairoMakie.Colorbar, one_dimensional.content)
end
