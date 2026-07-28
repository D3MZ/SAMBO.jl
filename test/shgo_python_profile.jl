using Random
using SAMBO
using Test

struct RecordingLocalStarts
    starts::Vector{Vector{Float64}}
end
function SAMBO.local_minimize!(
    state::SAMBO.SHGOState,
    solver::RecordingLocalStarts,
    start,
    start_value,
    lower,
    upper,
    budget,
)
    push!(solver.starts, collect(start))
    return collect(start), start_value
end

@testset "Python SAMBO SHGO profile" begin
    profile = SHGO(SAMBO.PythonSAMBOProfile())
    @test profile.sampling isa SAMBO.ScrambledHaltonDesign
    @test profile.topology isa SAMBO.PythonIncrementalDelaunayTopology
    @test profile.sampling_points == 80
    @test profile.local_starts == 4
    @test profile.local_start_policy isa SAMBO.FarthestFromLatestMinimum
    @test profile.minimum_local_reserve == 0
    @test !profile.divide_automatic_local_budget
    @test profile.minimum_homology_growth == 0
    @test profile.homology_patience == 1
    @test profile.convergence_tolerance == 1e-6
    @test profile.convergence_window == 30
    @test profile.local_solver isa SAMBO.QuasiNewtonSearch

    adjacency = [Int[] for _ in 1:8]
    insertion_order = Int[]
    inserted = falses(8)
    SAMBO._connect_delaunay_simplex!(
        adjacency,
        [5, 2, 7, 1, 3, 4],
        8;
        scipy_shgo_adjacency=true,
        vertex_order=insertion_order,
        inserted=inserted,
    )
    @test insertion_order == [5, 2, 7]
    sparse_complex = SAMBO.NeighborComplex(adjacency, insertion_order)
    @test SAMBO.localcandidates(sparse_complex, zeros(8)) ==
        insertion_order
    @test 6 ∉ SAMBO.localcandidates(sparse_complex, zeros(8))

    sparse_candidates = init(
        Problem(x -> sum(abs2, x), Box(zeros(2), ones(2))),
        SHGO(
            SAMBO.PythonSAMBOProfile();
            sampling_points=8,
            convergence_tolerance=0.0,
            convergence_window=0,
        );
        maximum_evaluations=20,
        rng=MersenneTwister(40),
    )
    SAMBO._append_samples!(
        sparse_candidates.workspace,
        [
            0.1 0.2 0.4 0.5 0.9
            0.1 0.2 0.4 0.5 0.9
        ],
        [3.0, 4.0, 1.0, 2.0, 0.0],
    )
    sparse_candidates.workspace.complex =
        SAMBO.NeighborComplex([[2], [1], [4], [3], Int[]])
    @test SAMBO.local_minimum_candidates(sparse_candidates) == [1, 3]

    overridden = SHGO(SAMBO.PythonSAMBOProfile(); sampling_points=17)
    @test overridden.sampling_points == 17

    constant_result = solve(
        Problem(_ -> 1.0, Box(fill(0.0, 5), fill(1.0, 5))),
        profile;
        maximum_evaluations=1000,
        rng=MersenneTwister(41),
    )
    @test evaluation_count(constant_result) == 30
    @test retcode(constant_result) == :success

    budget_limited_constant = solve(
        Problem(_ -> 1.0, Box(fill(0.0, 5), fill(1.0, 5))),
        profile;
        maximum_evaluations=30,
        rng=MersenneTwister(41),
    )
    @test evaluation_count(budget_limited_constant) == 30
    @test retcode(budget_limited_constant) == :evaluation_limit

    left = init(
        Problem(x -> x[1], Box(fill(0.0, 2), fill(1.0, 2))),
        profile;
        maximum_evaluations=200,
        rng=MersenneTwister(42),
    )
    right = init(
        Problem(x -> x[1], Box(fill(0.0, 2), fill(1.0, 2))),
        profile;
        maximum_evaluations=200,
        rng=MersenneTwister(42),
    )
    different = init(
        Problem(x -> x[1], Box(fill(0.0, 2), fill(1.0, 2))),
        profile;
        maximum_evaluations=200,
        rng=MersenneTwister(43),
    )
    SAMBO.refine_sampling!(left)
    SAMBO.refine_sampling!(right)
    SAMBO.refine_sampling!(different)
    @test latentpoints(trace(left)) == latentpoints(trace(right))
    @test latentpoints(trace(left)) != latentpoints(trace(different))
    @test evaluation_count(result(left)) == 80

    SAMBO.refine_sampling!(left)
    one_batch = init(
        Problem(x -> x[1], Box(fill(0.0, 2), fill(1.0, 2))),
        SHGO(SAMBO.PythonSAMBOProfile(); sampling_points=160);
        maximum_evaluations=200,
        rng=MersenneTwister(42),
    )
    SAMBO.refine_sampling!(one_batch)
    @test latentpoints(trace(left)) == latentpoints(trace(one_batch))

    recorded_starts = Vector{Float64}[]
    selection_state = init(
        Problem(x -> sum(abs2, x), Box([0.0, 0.0], [100.0, 1.0])),
        SHGO(
            topology=SAMBO.KNearestTopology(neighbors=1),
            local_solver=RecordingLocalStarts(recorded_starts),
            local_starts=4,
            local_budget=1,
            local_start_policy=SAMBO.FarthestFromLatestMinimum(),
        );
        maximum_evaluations=8,
        rng=MersenneTwister(44),
    )
    sample_points = [
        0.0 0.2 0.5 0.4
        0.0 1.0 0.0 1.0
    ]
    SAMBO._append_samples!(
        selection_state.workspace,
        sample_points,
        [0.0, 1.0, 2.0, 3.0],
    )
    selection_state.workspace.complex =
        SAMBO.NeighborComplex([Int[] for _ in 1:4])
    SAMBO.update_minimizer_pool!(selection_state)
    @test recorded_starts == [
        sample_points[:, 1],
        sample_points[:, 3],
        sample_points[:, 2],
        sample_points[:, 4],
    ]

    reevaluation_state = init(
        Problem(x -> sum(abs2, x), Box(zeros(2), ones(2))),
        SHGO(
            SAMBO.PythonSAMBOProfile();
            sampling_points=8,
            local_starts=1,
            local_budget=1,
            convergence_tolerance=0.0,
            convergence_window=0,
        );
        maximum_evaluations=20,
        rng=MersenneTwister(45),
    )
    SAMBO.refine_sampling!(reevaluation_state)
    SAMBO.update_complex!(reevaluation_state)
    before = evaluation_count(result(reevaluation_state))
    SAMBO.update_minimizer_pool!(reevaluation_state)
    @test evaluation_count(result(reevaluation_state)) == before + 1
    last = @view latentpoints(trace(reevaluation_state))[:, end]
    @test any(
        last == @view(latentpoints(trace(reevaluation_state))[:, column])
        for column in 1:before
    )
end
