using Test
using Random
using SAMBO

struct EmptyCoverageCandidates end
SAMBO.generate_candidates!(destination, state::SAMBO.SMBOState, ::EmptyCoverageCandidates) = 0

struct TerminatingCoverageCandidates end
function SAMBO.generate_candidates!(
    destination,
    state::SAMBO.SMBOState,
    ::TerminatingCoverageCandidates,
)
    state.core.retcode = :stalled
    return 0
end

struct InfeasibleCoverageCandidates end
function SAMBO.generate_candidates!(
    destination,
    state::SAMBO.SMBOState,
    ::InfeasibleCoverageCandidates,
)
    throw(SAMBO.InfeasibleSpaceError(size(destination, 2), 0))
end

@testset "SMBO and GP residual coverage" begin
    legacy = SMBO(exploration=3.5)
    @test legacy.acquisition == LowerConfidenceBound(3.5)

    generation_error = CandidateGenerationError("empty pool")
    @test generation_error.message == "empty pool"
    @test sprint(showerror, generation_error) == "empty pool"

    discrete = SearchSpace(x=Choices(:a, :b))
    seeded_discrete = init(
        Problem(discrete),
        SMBO(initial_points=0);
        initial_points=[(x=:a,)],
        initial_values=[1.0],
        maximum_evaluations=3,
        rng=MersenneTwister(1),
    )
    @test seeded_discrete.occupied == BitSet([1])

    SAMBO._ensure_candidate_capacity!(
        seeded_discrete.workspace,
        2,
        size(seeded_discrete.workspace.candidates, 2) + 1,
    )
    @test size(seeded_discrete.workspace.candidates, 1) == 2

    continuous = init(
        Problem(SearchSpace(x=Continuous(0.0, 1.0))),
        SMBO(initial_points=1, repeat_policy=SAMBO.AllowRepeatedEvaluations());
        initial_points=[(x=0.2,)],
        initial_values=[0.04],
        maximum_evaluations=5,
        rng=MersenneTwister(2),
    )
    pending = ask!(continuous, 1)
    @test SAMBO._occupied_count_scan(continuous) == 2
    cancel!(continuous, pending)

    unobserved = init(
        Problem(SearchSpace(x=Continuous(0.0, 1.0))),
        SMBO(initial_points=1);
        maximum_evaluations=3,
        rng=MersenneTwister(3),
    )
    selected = SAMBO._select_candidates(unobserved, reshape([0.1, 0.9], 1, :), 2)
    @test selected == reshape([0.1, 0.9], 1, :)

    partial_design = init(
        Problem(SearchSpace(x=Continuous(0.0, 1.0))),
        SMBO(initial_points=1, repeat_policy=SAMBO.AllowRepeatedEvaluations());
        maximum_evaluations=3,
        rng=MersenneTwister(4),
    )
    @test length(ask!(partial_design, 3)) == 3

    empty_state = init(
        Problem(x -> x.x^2, SearchSpace(x=Continuous(0.0, 1.0))),
        SMBO(
            initial_points=1,
            candidate_pool=4,
            candidate_sampler=EmptyCoverageCandidates(),
        );
        initial_points=[(x=0.25,)],
        initial_values=[0.0625],
        maximum_evaluations=3,
        rng=MersenneTwister(5),
    )
    empty_state.initial_design_cursor = size(empty_state.initial_design, 2) + 1
    @test_throws CandidateGenerationError ask!(empty_state, 1)

    exhausted_state = init(
        Problem(discrete),
        SMBO(
            initial_points=1,
            candidate_pool=4,
        );
        initial_points=[(x=:a,), (x=:b,)],
        initial_values=[1.0, 2.0],
        maximum_evaluations=3,
        rng=MersenneTwister(6),
    )
    exhausted_state.initial_design_cursor = size(exhausted_state.initial_design, 2) + 1
    @test isempty(ask!(exhausted_state, 1))
    @test retcode(result(exhausted_state)) == :space_exhausted

    explicit_state = init(
        Problem(SearchSpace(x=Continuous(0.0, 1.0))),
        SMBO(initial_points=1);
        maximum_evaluations=3,
        rng=MersenneTwister(7),
    )
    explicit_batch = ask!(explicit_state, 1)
    unmatched = mod(latentpoints(explicit_batch)[1, 1] + 0.5, 1.0)
    tell!(explicit_state, [(x=unmatched,)], [0.75])
    @test !isempty(explicit_state.pending)
    tell!(explicit_state, latentpoints(explicit_batch), [0.5])
    @test isempty(explicit_state.pending)

    infeasible_state = init(
        Problem(x -> x.x, SearchSpace(x=Continuous(0.0, 1.0))),
        SMBO(
            initial_points=1,
            candidate_pool=4,
            candidate_sampler=InfeasibleCoverageCandidates(),
        );
        initial_points=[(x=0.5,)],
        initial_values=[0.5],
        maximum_evaluations=2,
        rng=MersenneTwister(8),
    )
    infeasible_state.initial_design_cursor =
        size(infeasible_state.initial_design, 2) + 1
    @test retcode(solve!(infeasible_state)) == :infeasible_space

    @test_throws NumericalFailureError SAMBO._factor_covariance(
        fill(-1.0, 1, 1),
        GeometricJitter(1e-10, 10.0, 2),
        Float64,
    )

    generic_occupied = Set([[0.5]])
    SAMBO._release_occupied!(
        generic_occupied,
        SearchSpace(x=Continuous(0.0, 1.0)),
        [0.5],
    )
    @test isempty(generic_occupied)

    reservoir = init(
        Problem(SearchSpace(x=Choices(1, 2, 3, 4, 5))),
        SMBO(initial_points=0, candidate_pool=2);
        maximum_evaluations=5,
        rng=MersenneTwister(9),
    )
    @test size(SAMBO._unused_finite_candidates(reservoir, 2)) == (1, 2)

    terminating = init(
        Problem(x -> x.x^2, SearchSpace(x=Continuous(0.0, 1.0))),
        SMBO(
            initial_points=1,
            candidate_pool=4,
            candidate_sampler=TerminatingCoverageCandidates(),
        );
        initial_points=[(x=0.5,)],
        initial_values=[0.25],
        maximum_evaluations=2,
        rng=MersenneTwister(10),
    )
    terminating.initial_design_cursor =
        size(terminating.initial_design, 2) + 1
    @test isempty(ask!(terminating, 1))
    @test retcode(result(terminating)) == :stalled

    partial = init(
        Problem(SearchSpace(x=Continuous(0.0, 1.0))),
        SMBO(initial_points=2, batch_size=2);
        maximum_evaluations=3,
        rng=MersenneTwister(11),
    )
    partial_batch = ask!(partial, 2)
    tell!(partial, partial_batch, [1], [0.5])
    tell!(partial, partial_batch, [0.25])
    @test isempty(partial.pending)
end
