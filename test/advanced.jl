struct BrokenSurrogate end
SAMBO.fitmodel(::BrokenSurrogate, points, values, rng) =
    throw(NumericalFailureError("intentional test failure"))
struct FillCandidates
    value::Float64
    calls::Base.RefValue{Int}
end
function SAMBO.generate_candidates!(destination, state, sampler::FillCandidates)
    sampler.calls[] += 1
    fill!(destination, sampler.value)
    return size(destination, 2)
end

@testset "advanced solver contracts" begin
    @testset "iterative SHGO" begin
        @test_throws SAMBO.ComplexConstructionError SAMBO.buildcomplex(
            [0.0 0.3 0.6 1.0; 0.0 0.3 0.6 1.0],
            SAMBO.DelaunayTopology(),
        )
        knn = SAMBO.buildcomplex(
            [0.0 1.0; 0.0 1.0],
            SAMBO.KNearestTopology(neighbors=1),
        )
        @test knn.adjacency == [[2], [1]]

        problem = Problem(
            x -> sum(abs2, x),
            Box(fill(-2.0, 2), fill(2.0, 2)),
        )
        state = init(
            problem,
            SHGO(sampling_points=8, homology_patience=100);
            maximum_evaluations=100,
            rng=MersenneTwister(81),
        )
        step!(state)
        storage = state.workspace.sample_points
        first_samples = state.workspace.sample_count
        @test state.workspace.refinements == 1
        @test first_samples > 0
        @test !isempty(state.workspace.complex.adjacency)
        step!(state)
        @test state.workspace.sample_points === storage
        @test state.workspace.refinements == 2
        @test state.workspace.sample_count > first_samples
        @test SAMBO.homology_rank(state) >= 1
        @test SAMBO.homology_rank(state) == SAMBO.minimizer_count(state)
        @test SAMBO.topographical_candidate_count(state) ==
            length(state.workspace.candidate_indices)
        @test SAMBO.homology_rank_differential(state) >= 0

        result = solve!(state)
        @test minimum(result) < 0.05
        @test result.statistics.refinements >= 2
        @test result.statistics.local_minima >= 1
        @test result.statistics.homology_rank == SAMBO.homology_rank(state)

        too_small = solve(
            problem,
            SHGO();
            maximum_evaluations=2,
            rng=MersenneTwister(810),
        )
        @test retcode(too_small) == :evaluation_limit
        @test evaluation_count(too_small) == 0
    end

    @testset "external evaluation lifecycle" begin
        problem = Problem(SearchSpace(x=Continuous(0.0, 1.0)))
        state = init(
            problem,
            SMBO(initial_points=4);
            maximum_evaluations=8,
            rng=MersenneTwister(82),
        )
        batch = ask!(state, 4)
        tell!(state, batch, [1, 3], [0.4, 0.2])
        fail!(state, batch, [2], [ErrorException("worker failed")])
        cancel!(state, batch, [4])
        @test evaluation_count(result(state)) == 2
        @test result(state).statistics.failed_candidates == 1
        @test isempty(state.pending)

        replacement = ask!(state, 2)
        tell!(state, replacement[1], replacement[1].x^2)
        cancel!(state, replacement, [2])
        @test evaluation_count(result(state)) == 3

        pending = ask!(state, 2)
        saved = checkpoint(state)
        restored = restore(problem, saved)
        @test latentpoints(ask!(restored, 0)) == zeros(1, 0)
        tell!(restored, pending, [point.x^2 for point in pending])
        @test evaluation_count(result(restored)) == 5
        @test trace(result(restored)).count == 5

        finite = init(
            Problem(SearchSpace(choice=Choices(:a))),
            SMBO(initial_points=1);
            maximum_evaluations=4,
            rng=MersenneTwister(821),
        )
        finite_batch = ask!(finite, 1)
        @test length(finite.occupied) == 1
        cancel!(finite, finite_batch, [1])
        @test isempty(finite.occupied)
        replacement = ask!(finite, 1)
        @test length(replacement) == 1
        @test length(finite.occupied) == 1
        tell!(finite, replacement, [0.0])
        @test isempty(ask!(finite))
        @test finite.core.retcode == :space_exhausted
    end

    @testset "candidate mixtures and refit schedules" begin
        state = init(
            Problem(SearchSpace(x=Continuous(0.0, 1.0))),
            SMBO();
            maximum_evaluations=4,
            rng=MersenneTwister(820),
        )
        global_calls = Ref(0)
        local_calls = Ref(0)
        destination = zeros(1, 7)
        all_local = SAMBO.MixtureCandidates(
            FillCandidates(0.1, global_calls),
            FillCandidates(0.9, local_calls);
            global_fraction=0,
        )
        @test generate_candidates!(destination, state, all_local) == 7
        @test global_calls[] == 0
        @test local_calls[] == 1
        @test all(==(0.9), destination)

        global_calls[] = 0
        local_calls[] = 0
        all_global = SAMBO.MixtureCandidates(
            FillCandidates(0.1, global_calls),
            FillCandidates(0.9, local_calls);
            global_fraction=1,
        )
        @test generate_candidates!(destination, state, all_global) == 7
        @test global_calls[] == 1
        @test local_calls[] == 0
        @test all(==(0.1), destination)

        @test SAMBO.refit_interval(SAMBO.FixedRefit(3), 100) == 3
        @test SAMBO.refit_interval(SAMBO.SquareRootRefit(2), 3) == 2
        @test SAMBO.refit_interval(SAMBO.SquareRootRefit(2), 100) == 10
    end

    @testset "sense, failures, constraints, and return codes" begin
        for algorithm in (
            SCEUA(),
            SMBO(candidate_pool=32),
            SHGO(sampling_points=8, homology_patience=20),
        )
            result = minimize(
                x -> -(x[1] - 0.7)^2,
                Box([0.0], [1.0]);
                sense=Maximize(),
                algorithm,
                maximum_evaluations=40,
                rng=MersenneTwister(83),
            )
            @test minimum(result) > -0.02
            @test abs(minimizer(result)[1] - 0.7) < 0.15
        end

        violation_problem = Problem(
            x -> x[1],
            Box([0.0], [1.0]);
            constraint=x -> x[1] - 0.5,
        )
        @test constraint_violation(violation_problem, [0.25]) == -0.25
        @test isfeasible(violation_problem, [0.25])
        @test !isfeasible(violation_problem, [0.75])

        penalized = minimize(
            x -> x[1] < 0.5 ? NaN : x[1],
            Box([0.0], [1.0]);
            algorithm=SCEUA(),
            nonfinite=PenalizeNonfinite(),
            maximum_evaluations=10,
            rng=MersenneTwister(84),
        )
        @test isfinite(minimum(penalized))
        @test_throws ArgumentError minimize(
            _ -> Inf,
            Box([0.0], [1.0]);
            maximum_evaluations=1,
        )
        accepted = minimize(
            _ -> Inf,
            Box([0.0], [1.0]);
            algorithm=SCEUA(),
            nonfinite=AllowInfinite(),
            maximum_evaluations=1,
        )
        @test minimum(accepted) == Inf

        impossible = Problem(
            x -> x[1],
            Box([0.0], [1.0]);
            constraint=_ -> false,
        )
        for algorithm in (SCEUA(), SMBO(), SHGO(sampling_points=4))
            @test retcode(solve(
                impossible,
                algorithm;
                maximum_evaluations=5,
                rng=MersenneTwister(85),
            )) == :infeasible_space
        end

        stalled = minimize(
            _ -> 1.0,
            Box([0.0], [1.0]);
            algorithm=SMBO(candidate_pool=16),
            maximum_evaluations=10,
            stall_evaluations=1,
            rng=MersenneTwister(86),
        )
        @test retcode(stalled) == :stalled

        timed = minimize(
            x -> (sleep(0.001); x[1]),
            Box([0.0], [1.0]);
            algorithm=SCEUA(),
            maximum_evaluations=10,
            time_limit=1e-6,
            rng=MersenneTwister(87),
        )
        @test retcode(timed) == :time_limit

        numerical = minimize(
            x -> x[1]^2,
            Box([0.0], [1.0]);
            algorithm=SMBO(
                surrogate=BrokenSurrogate(),
                initial_points=2,
                candidate_pool=16,
            ),
            maximum_evaluations=4,
            rng=MersenneTwister(88),
        )
        @test retcode(numerical) == :numerical_failure

        finite = init(
            Problem(SearchSpace(x=Choices(:a, :b))),
            SMBO(initial_points=2);
            maximum_evaluations=4,
            rng=MersenneTwister(89),
        )
        candidates = ask!(finite, 2)
        tell!(finite, candidates, [1.0, 2.0])
        @test isempty(ask!(finite, 1))
        @test retcode(result(finite)) == :space_exhausted
        step!(finite)
        @test retcode(result(finite)) == :space_exhausted
    end
end
