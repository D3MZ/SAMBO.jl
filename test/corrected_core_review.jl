module CorrectedCoreReviewTests

using Random
using SAMBO
using Test

struct CorrectedRejectAllRepairs end
SAMBO.repair!(
    rng,
    proposal,
    ::CorrectedRejectAllRepairs,
    problem,
    centroid,
) = false

@testset "corrected core review" begin
    @testset "proposal failure and low topological budget" begin
        state = init(
            Problem(
                x -> sum(abs2, x),
                Box(fill(-1.0, 2), fill(1.0, 2)),
            ),
            SCEUA(
                complexes=2,
                complex_size=3,
                repair=CorrectedRejectAllRepairs(),
            );
            maximum_evaluations=20,
            rng=Xoshiro(1),
        )
        step!(state)
        step!(state)
        @test retcode(result(state)) == :stalled

        solved = solve(
            Problem(
                x -> sum(abs2, x),
                Box(fill(-1.0, 2), fill(1.0, 2)),
            ),
            SAMBO.TopologicalMultistart();
            maximum_evaluations=2,
            rng=Xoshiro(2),
        )
        @test retcode(solved) == :evaluation_limit
        @test evaluation_count(solved) <= 2
    end

    @testset "checkpoint compatibility" begin
        problem = Problem(SearchSpace(x=Continuous(0.0, 1.0)))
        state = init(
            problem,
            SMBO(initial_points=1);
            maximum_evaluations=4,
            rng=Xoshiro(3),
        )
        batch = ask!(state, 1)
        saved = checkpoint(state)
        @test_throws ArgumentError restore(
            Problem(SearchSpace(x=Continuous(0.0, 2.0))),
            saved,
        )
        @test_throws ArgumentError restore(
            Problem(
                SearchSpace(x=Continuous(0.0, 1.0));
                sense=Maximize(),
            ),
            saved,
        )
        restored = restore(problem, saved)
        @test restored.pending[batch.identifier].points ==
            latentpoints(batch)
    end

    @testset "finite exhaustion is shared" begin
        singleton = SearchSpace(choice=Choices(:only))
        for algorithm in (
            SCEUA(complexes=1, complex_size=2),
            SMBO(initial_points=1, candidate_pool=8),
        )
            solved = solve(
                Problem(_ -> 0.0, singleton),
                algorithm;
                maximum_evaluations=4,
                rng=Xoshiro(4),
            )
            @test evaluation_count(solved) == 1
            @test retcode(solved) == :space_exhausted
        end

        finite = SearchSpace(kind=Choices(:a, :b), count=1:2)
        state = init(
            Problem(
                finite;
                constraint=point -> point.kind == :a,
            ),
            SMBO(initial_points=0, candidate_pool=32);
            initial_points=[
                (kind=:a, count=1),
                (kind=:a, count=2),
            ],
            initial_values=[1.0, 0.0],
            maximum_evaluations=4,
            rng=Xoshiro(5),
        )
        @test isempty(ask!(state, 1))
        @test retcode(result(state)) == :space_exhausted

        constrained_finite = SearchSpace(choice=1:100)
        approximate = solve(
            Problem(
                point -> Float64(point.choice),
                constrained_finite;
                constraint=point -> point.choice <= 10,
            ),
            SMBO(
                initial_points=1,
                candidate_pool=32,
                candidate_equality=SAMBO.ApproximateCandidateEquality(0.001),
            );
            maximum_evaluations=100,
            rng=Xoshiro(6),
        )
        @test evaluation_count(approximate) == 10
        @test retcode(approximate) == :space_exhausted
        @test Set(
            decode(
                constrained_finite,
                @view(latentpoints(trace(approximate))[:, column]),
            ).choice for column in axes(latentpoints(trace(approximate)), 2)
        ) == Set(1:10)

        merged_approximate = solve(
            Problem(
                point -> Float64(point.choice),
                constrained_finite;
                constraint=point -> point.choice <= 10,
            ),
            SMBO(
                initial_points=1,
                candidate_pool=32,
                candidate_equality=SAMBO.ApproximateCandidateEquality(0.02),
            );
            maximum_evaluations=100,
            rng=Xoshiro(6),
        )
        @test 0 < evaluation_count(merged_approximate) < 10
        @test retcode(merged_approximate) == :space_exhausted
    end

    @testset "categorical construction and contract" begin
        @test Choices([:a, :b]).values == (:a, :b)
        vector_space = SearchSpace(kind=Choices([:a, :b]))
        @test decode(vector_space, [1.0]) == (kind=:b,)

        categorical = SearchSpace(kind=Choices(:a, :b, :c))
        za = encode(categorical, (kind=:a,))
        zb = encode(categorical, (kind=:b,))
        zc = encode(categorical, (kind=:c,))
        @test sum(abs2, za - zb) < sum(abs2, za - zc)
        choices_documentation = Base.Docs.meta(SAMBO)[
            Base.Docs.Binding(SAMBO, :Choices)
        ]
        @test occursin(
            "ordinal",
            lowercase(repr(choices_documentation)),
        )
    end

    @testset "QMC continuation" begin
        space = Box(zeros(2), ones(2))
        for design in (
            SAMBO.SobolDesign(),
            SAMBO.HaltonDesign(skip=0),
            SAMBO.ScrambledHaltonDesign(seed=0x1234),
        )
            whole = Matrix{Float64}(undef, 2, 16)
            left = Matrix{Float64}(undef, 2, 8)
            right = Matrix{Float64}(undef, 2, 8)
            SAMBO.sample!(Xoshiro(6), whole, design, space)
            split_rng = Xoshiro(6)
            SAMBO.sample!(split_rng, left, design, space)
            SAMBO.sample!(
                split_rng,
                right,
                SAMBO.advance(design, 8),
                space,
            )
            @test hcat(left, right) == whole
        end
    end

    @testset "partial-dependence boundaries" begin
        space = SearchSpace(kind=Choices(:a, :b, :c, :d, :e))
        grid = SAMBO._canonical_grid(space, 1, 2, Float64)
        @test length(grid) == 5
        @test [
            decode(space, [value])
            for value in grid
        ] == [(kind=value,) for value in (:a, :b, :c, :d, :e)]

        fitted = minimize(
            x -> sum(abs2, x),
            Box(fill(-1.0, 2), fill(1.0, 2));
            algorithm=SMBO(candidate_pool=32),
            maximum_evaluations=8,
            rng=Xoshiro(7),
        )
        @test_throws ArgumentError partialdependence(
            fitted;
            samples=8,
            background=fill(2.0, 2, 8),
        )
    end

    @testset "inner-constructor invariants" begin
        @test_throws ArgumentError SAMBO.PatternSearch(-1.0, 1e-5)
        @test_throws ArgumentError SAMBO.QuasiNewtonSearch(
            finite_difference_step=0,
        )
        @test_throws ArgumentError SAMBO.QuasiNewtonSearch(
            gradient_tolerance=0,
        )
        @test_throws ArgumentError SAMBO.QuasiNewtonSearch(minimum_step=0)
        @test_throws ArgumentError GeometricJitter(-1.0, 10.0, 8)
        @test_throws ArgumentError SAMBO.LocalPenalization(-1.0, 0.1)
        @test_throws ArgumentError SAMBO.MixtureCandidates(
            SAMBO.UniformCandidates(),
            SAMBO.EliteGaussianCandidates(),
            2.0,
        )
        values = [0.2, 0.8]
        scale = ARDLengthScale(values)
        values[1] = 100.0
        @test scale.values[1] == 0.2
    end

    @testset "numeric boundaries" begin
        tolerance = big"1e-500"
        algorithm = SCEUA(population_tolerance=tolerance)
        @test algorithm.population_tolerance isa BigFloat
        @test algorithm.population_tolerance == tolerance
        @test SAMBO.automatic_sampling_count(64, 100) == 100
        @test_throws ArgumentError SAMBO.buildcomplex(
            BigFloat[0 1 0 1; 0 0 1 1],
            SAMBO.DelaunayTopology(),
        )
        @test_nowarn SAMBO.buildcomplex(
            BigFloat[0 1 0; 0 0 1],
            SAMBO.KNearestTopology(neighbors=1),
        )
    end

    @testset "GP training boundary" begin
        specification = GaussianProcessSurrogate()
        @test_throws DimensionMismatch SAMBO.fitmodel(
            specification,
            rand(2, 3),
            rand(2),
        )
        @test_throws ArgumentError SAMBO.fitmodel(
            specification,
            [0.0 NaN; 0.0 1.0],
            [0.0, 1.0],
        )
        @test_throws ArgumentError SAMBO.fitmodel(
            specification,
            rand(2, 2),
            [0.0, Inf],
        )
    end

    @testset "sense-neutral result accessors" begin
        maximized = minimize(
            x -> -(x[1] - 0.7)^2,
            Box([0.0], [1.0]);
            sense=Maximize(),
            maximum_evaluations=20,
            rng=Xoshiro(11),
        )
        @test SAMBO.bestvalue(maximized) ==
            maximum(objectivevalues(trace(maximized)))
        @test SAMBO.bestpoint(maximized) == minimizer(maximized)
        @test SAMBO.optimizationsense(maximized) isa Maximize
    end
end

end
