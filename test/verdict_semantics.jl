using Random
using SAMBO
using Serialization
using Test

struct AuditFillCandidates{T}
    value::T
    generated::Int
    calls::Base.RefValue{Int}
end
function SAMBO.generate_candidates!(
    destination,
    state::SAMBO.SMBOState,
    sampler::AuditFillCandidates,
)
    sampler.calls[] += 1
    count = min(sampler.generated, size(destination, 2))
    count > 0 && fill!(@view(destination[:, 1:count]), sampler.value)
    return count
end

mutable struct AuditAlternatingRepair
    calls::Int
end
function SAMBO.repair!(
    rng,
    proposal,
    policy::AuditAlternatingRepair,
    problem,
    centroid,
)
    policy.calls += 1
    isodd(policy.calls) || return false
    SAMBO.project!(proposal, problem.space)
    return true
end

struct AuditCountingSurrogate
    fits::Base.RefValue{Int}
end
struct AuditPositionModel end
function SAMBO.fitmodel(specification::AuditCountingSurrogate, points, values, rng)
    specification.fits[] += 1
    return AuditPositionModel()
end
function SAMBO.predictmeanvariance!(means, variances, ::AuditPositionModel, points)
    means .= @view points[1, :]
    fill!(variances, zero(eltype(variances)))
    return means, variances
end
function SAMBO.predictmean!(means, ::AuditPositionModel, points)
    means .= @view points[1, :]
    return means
end

@testset "verdict semantic audit" begin
    @testset "topology is explicit" begin
        degenerate = [0.0 0.3 0.6 1.0; 0.0 0.3 0.6 1.0]
        @test_throws ComplexConstructionError SAMBO.buildcomplex(
            degenerate,
            DelaunayTopology(),
        )
        knn = SAMBO.buildcomplex(degenerate, KNearestTopology(neighbors=1))
        @test all(!isempty, knn.adjacency)
    end

    @testset "candidate generation does not substitute" begin
        calls = Ref(0)
        sampler = AuditFillCandidates(0.777, 1, calls)
        state = init(
            Problem(SearchSpace(x=Continuous(0.0, 1.0))),
            SMBO(
                initial_points=1,
                candidate_pool=8,
                candidate_sampler=sampler,
                repeat_policy=AllowRepeatedEvaluations(),
            );
            maximum_evaluations=4,
            rng=MersenneTwister(901),
        )
        seed = ask!(state, 1)
        tell!(state, seed, [1.0])
        generated = ask!(state, 3)
        @test calls[] == 1
        @test length(generated) == 1
        @test latentpoints(generated) == reshape([0.777], 1, 1)
    end

    @testset "space exhaustion survives step!" begin
        state = init(
            Problem(SearchSpace(choice=Choices(:only))),
            SMBO(initial_points=1);
            maximum_evaluations=3,
            rng=MersenneTwister(902),
        )
        batch = ask!(state, 1)
        tell!(state, batch, [0.0])
        step!(state)
        @test retcode(result(state)) == :space_exhausted
    end

    @testset "SHGO rank and no-progress reason" begin
        state = init(
            Problem(x -> sum(abs2, x), SAMBO.Box(fill(-1.0, 2), fill(1.0, 2))),
            SHGO(sampling_points=8, homology_patience=100);
            maximum_evaluations=50,
            rng=MersenneTwister(903),
        )
        step!(state)
        @test homology_rank(state) == minimizer_count(state)
        @test state.workspace.homology_rank == homology_rank(state)
        @test result(state).statistics.homology_rank == homology_rank(state)
        @test state.workspace.homology_rank_differential ==
            homology_rank_differential(state)

        budget_limited = solve(
            Problem(x -> sum(abs2, x), SAMBO.Box(fill(-1.0, 2), fill(1.0, 2))),
            SHGO(sampling_points=8);
            maximum_evaluations=2,
            rng=MersenneTwister(904),
        )
        @test retcode(budget_limited) == :evaluation_limit
        @test evaluation_count(budget_limited) == 0
    end

    @testset "blocked topological local start is exhausted, not infeasible" begin
        state = init(
            Problem(
                x -> x[1]^2,
                SAMBO.Box([0.0], [1.0]);
                constraint=x -> abs(x[1] - 0.5) <= eps(),
            ),
            TopologicalMultistart(samples=4, local_starts=1);
            initial_points=[[0.5]],
            initial_values=[0.25],
            maximum_evaluations=4,
            rng=MersenneTwister(913),
        )
        state.workspace.sample_points[:, 1] .= 0.5
        state.workspace.sample_values[1] = 0.25
        state.workspace.local_indices = [1]
        state.workspace.local_minima_count = 1
        state.workspace.initialized = true
        SAMBO._begin_local_start!(state, 1)
        step!(state)
        @test retcode(result(state)) == :stalled
        @test evaluation_count(result(state)) == 0
    end

    @testset "failed SCE contraction is not evaluated" begin
        policy = AuditAlternatingRepair(0)
        state = init(
            Problem(_ -> 1.0, SAMBO.Box([-1.0], [1.0])),
            SCEUA(complexes=1, complex_size=2, repair=policy);
            maximum_evaluations=10,
            rng=MersenneTwister(905),
        )
        step!(state)
        before = evaluation_count(result(state))
        step!(state)
        @test evaluation_count(result(state)) - before == 2
        @test policy.calls == 2
    end

    @testset "callback observes an atomic batch" begin
        events = Any[]
        state = init(
            Problem(SearchSpace(x=Continuous(0.0, 1.0))),
            SMBO(initial_points=3);
            maximum_evaluations=3,
            callback=event -> (push!(events, event); false),
            rng=MersenneTwister(906),
        )
        batch = ask!(state, 3)
        tell!(state, batch, [point.x^2 for point in batch])
        @test length(events) == 1
        @test only(events).batch_size == 3
        @test only(events).evaluation == 3
        @test evaluation_count(result(state)) == 3
    end

    @testset "mixture quotas are exact" begin
        state = init(
            Problem(SearchSpace(x=Continuous(0.0, 1.0))),
            SMBO();
            maximum_evaluations=2,
            rng=MersenneTwister(907),
        )
        global_calls, local_calls = Ref(0), Ref(0)
        global_sampler = AuditFillCandidates(0.1, typemax(Int), global_calls)
        local_sampler = AuditFillCandidates(0.9, typemax(Int), local_calls)
        destination = zeros(1, 7)

        all_local = MixtureCandidates(
            global_sampler,
            local_sampler;
            global_fraction=0,
        )
        @test generate_candidates!(destination, state, all_local) == 7
        @test (global_calls[], local_calls[]) == (0, 1)
        @test all(==(0.9), destination)

        global_calls[] = 0
        local_calls[] = 0
        all_global = MixtureCandidates(
            global_sampler,
            local_sampler;
            global_fraction=1,
        )
        @test generate_candidates!(destination, state, all_global) == 7
        @test (global_calls[], local_calls[]) == (1, 0)
        @test all(==(0.1), destination)
    end

    @testset "refit schedules control actual fits" begin
        function fit_count(schedule, evaluations)
            fits = Ref(0)
            sampler = AuditFillCandidates(0.75, typemax(Int), Ref(0))
            state = init(
                Problem(SearchSpace(x=Continuous(0.0, 1.0))),
                SMBO(
                    surrogate=AuditCountingSurrogate(fits),
                    acquisition=GreedyMean(),
                    initial_points=1,
                    candidate_pool=8,
                    candidate_sampler=sampler,
                    repeat_policy=AllowRepeatedEvaluations(),
                    refit_schedule=schedule,
                );
                maximum_evaluations=evaluations,
                rng=MersenneTwister(908),
            )
            while evaluation_count(result(state)) < evaluations
                batch = ask!(state, 1)
                tell!(state, batch, [batch[1].x])
            end
            return fits[]
        end
        @test fit_count(FixedRefit(3), 5) == 2
        @test fit_count(SquareRootRefit(2), 5) == 2
    end

    @testset "repeat policies and equality" begin
        function repeated_state(policy, equality=ExactCandidateEquality())
            sampler = AuditFillCandidates(0.5, typemax(Int), Ref(0))
            state = init(
                Problem(SearchSpace(x=Continuous(0.0, 1.0))),
                SMBO(
                    initial_points=1,
                    candidate_pool=8,
                    candidate_sampler=sampler,
                    repeat_policy=policy,
                    candidate_equality=equality,
                );
                initial_points=[(x=0.5,)],
                initial_values=[0.0],
                maximum_evaluations=3,
                rng=MersenneTwister(909),
            )
            return state
        end
        @test_throws CandidateGenerationError ask!(
            repeated_state(AvoidRepeatedEvaluations()),
            1,
        )
        @test length(ask!(repeated_state(AllowRepeatedEvaluations()), 1)) == 1

        approximate_sampler = AuditFillCandidates(0.5001, typemax(Int), Ref(0))
        approximate = init(
            Problem(SearchSpace(x=Continuous(0.0, 1.0))),
            SMBO(
                initial_points=1,
                candidate_pool=8,
                candidate_sampler=approximate_sampler,
                repeat_policy=AvoidRepeatedEvaluations(),
                candidate_equality=ApproximateCandidateEquality(0.001),
            );
            initial_points=[(x=0.5,)],
            initial_values=[0.0],
            maximum_evaluations=3,
            rng=MersenneTwister(910),
        )
        @test_throws CandidateGenerationError ask!(approximate, 1)
    end

    @testset "batch strategies are distinct" begin
        candidates = reshape([0.0, 0.01, 1.0], 1, :)
        function selected(strategy)
            state = init(
                Problem(SearchSpace(x=Continuous(0.0, 1.0))),
                SMBO(
                    surrogate=AuditCountingSurrogate(Ref(0)),
                    acquisition=GreedyMean(),
                    batch_strategy=strategy,
                    repeat_policy=AllowRepeatedEvaluations(),
                );
                initial_points=[(x=0.5,)],
                initial_values=[0.5],
                maximum_evaluations=4,
                rng=MersenneTwister(911),
            )
            return SAMBO._select_candidates(state, candidates, 2)
        end
        @test selected(GreedyBatch()) == reshape([0.0, 0.01], 1, :)
        @test selected(LocalPenalization(strength=10.0, radius=0.1)) ==
            reshape([0.0, 1.0], 1, :)
    end

    @testset "parametric heterogeneous spaces" begin
        point = (continuous=0.25, integer=2, category=:b)
        for T in (Float32, BigFloat)
            space = SearchSpace(
                T;
                continuous=Continuous(0, 1),
                integer=1:3,
                category=Choices(:a, :b),
            )
            latent = encode(space, point)
            @test eltype(latent) == T
            @test decode(space, latent) == point
        end
    end

    @testset "checkpoint snapshot and serialization contract" begin
        problem = Problem(SearchSpace(x=Continuous(0.0, 1.0)))
        state = init(
            problem,
            SMBO(initial_points=2);
            maximum_evaluations=4,
            rng=MersenneTwister(912),
        )
        pending = ask!(state, 2)
        saved = checkpoint(state)
        tell!(state, pending, [1.0, 0.5])
        @test saved.trace.count == 0

        io = IOBuffer()
        serialize(io, saved)
        seekstart(io)
        roundtrip = deserialize(io)
        restored = restore(problem, roundtrip)
        @test length(restored.pending) == 1
        tell!(restored, pending, [1.0, 0.5])
        @test evaluation_count(result(restored)) == 2
    end
end
