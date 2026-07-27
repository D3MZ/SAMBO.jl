using SAMBO
using Random
using Test

struct RejectEveryRepair end
SAMBO.repair!(rng, proposal, ::RejectEveryRepair, problem, centroid) = false

struct CoverageLocalSolver end
function SAMBO.local_minimize!(
    state::SAMBO.SHGOState,
    ::CoverageLocalSolver,
    start,
    start_value,
    lower,
    upper,
    budget,
)
    return collect(start), start_value - one(start_value)
end

@testset "SCE-UA uncovered branches" begin
    seeded = init(
        Problem(x -> sum(abs2, x), Box([-1.0], [1.0])),
        SCEUA(complexes=1, complex_size=3);
        initial_points=[[-0.5], [0.25]],
        initial_values=[0.25, 0.0625],
        maximum_evaluations=3,
        rng=MersenneTwister(801),
    )
    step!(seeded)
    @test seeded.workspace.initialized
    @test seeded.workspace.values[1:2] == [0.0625, 0.25]

    impossible = Problem(
        x -> x[1]^2,
        Box([0.0], [1.0]);
        constraint=x -> false,
    )
    proposal = [0.5]
    @test !SAMBO._project_feasible!(proposal, impossible)
    @test !SAMBO.repair!(
        MersenneTwister(802),
        proposal,
        SAMBO.MoveTowardCentroid(),
        impossible,
        [0.5],
    )

    rejected = init(
        Problem(x -> sum(abs2, x), Box([-1.0], [1.0])),
        SCEUA(complexes=1, complex_size=2, repair=RejectEveryRepair());
        maximum_evaluations=8,
        rng=MersenneTwister(803),
    )
    step!(rejected)
    step!(rejected)
    @test retcode(result(rejected)) == :stalled

    converged = init(
        Problem(x -> sum(abs2, x), Box([-1.0], [1.0])),
        SCEUA(
            complexes=1,
            complex_size=2,
            population_tolerance=2.0,
        );
        maximum_evaluations=8,
        rng=MersenneTwister(804),
    )
    step!(converged)
    step!(converged)
    @test retcode(result(converged)) == :success
end

@testset "topological multistart uncovered branches" begin
    seeded = init(
        Problem(x -> sum(abs2, x), Box([-1.0], [1.0])),
        TopologicalMultistart(samples=3, local_starts=1);
        initial_points=[[-0.75], [0.25]],
        initial_values=[0.5625, 0.0625],
        maximum_evaluations=3,
        rng=MersenneTwister(811),
    )
    step!(seeded)
    @test seeded.workspace.initialized
    @test seeded.workspace.sample_values[1:2] == [0.0625, 0.5625]

    restarted = init(
        Problem(_ -> 1.0, Box([0.0], [1.0])),
        TopologicalMultistart(
            samples=2,
            local_starts=1,
            local_solver=SAMBO.PatternSearch(
                initial_step=0.2,
                minimum_step=0.15,
            ),
        );
        maximum_evaluations=5,
        rng=MersenneTwister(812),
    )
    step!(restarted)
    step!(restarted)
    @test restarted.workspace.current_start == 1
    @test restarted.workspace.completed_starts == 1
    @test retcode(result(restarted)) == :stalled

    infeasible = solve(
        Problem(
            x -> x[1]^2,
            Box([0.0], [1.0]);
            constraint=x -> false,
        ),
        TopologicalMultistart(samples=2);
        maximum_evaluations=3,
        rng=MersenneTwister(813),
    )
    @test retcode(infeasible) == :infeasible_space

    huge = [
        0.0 1.0e308 0.0 1.0e308
        0.0 0.0 1.0e308 1.0e308
    ]
    @test_throws SAMBO.ComplexConstructionError SAMBO.buildcomplex(
        huge,
        SAMBO.DelaunayTopology(),
    )
end

@testset "SHGO uncovered branches" begin
    seeded = init(
        Problem(x -> sum(abs2, x), Box([-1.0], [1.0])),
        SHGO(
            topology=SAMBO.KNearestTopology(neighbors=1),
            sampling_points=2,
            minimization_schedule=SAMBO.MinimizeAtTermination(),
        );
        initial_points=[[-0.5], [0.25]],
        initial_values=[0.25, 0.0625],
        maximum_evaluations=8,
        rng=MersenneTwister(821),
    )
    SAMBO._initialize_shgo_observations!(seeded)
    @test seeded.workspace.sample_count == 2
    @test seeded.workspace.sample_values[1:2] == [0.25, 0.0625]

    candidates_state = init(
        Problem(x -> x[1]^2, Box([0.0], [1.0])),
        SHGO(
            topology=SAMBO.KNearestTopology(neighbors=1),
            sampling_points=2,
            minimization_schedule=SAMBO.MinimizeAtTermination(),
        );
        maximum_evaluations=8,
        rng=MersenneTwister(822),
    )
    SAMBO._append_samples!(
        candidates_state.workspace,
        reshape([0.2, 0.8], 1, :),
        [0.04, 0.64],
    )
    candidates_state.workspace.complex = SAMBO.NeighborComplex([Int[], Int[]])
    @test SAMBO.local_minimum_candidates(candidates_state) == [1, 2]

    constrained = init(
        Problem(
            x -> x[1]^2,
            Box([0.0], [1.0]);
            constraint=x -> x[1] == 0.5,
        ),
        SHGO(
            local_solver=SAMBO.PatternSearch(
                initial_step=0.2,
                minimum_step=0.1,
            ),
        );
        maximum_evaluations=8,
        rng=MersenneTwister(823),
    )
    point, value = SAMBO.local_minimize!(
        constrained,
        constrained.algorithm.local_solver,
        [0.5],
        0.25,
        [0.0],
        [1.0],
        8,
    )
    @test point == [0.5]
    @test value == 0.25
    @test evaluation_count(result(constrained)) == 0

    duplicate = init(
        Problem(x -> x[1]^2, Box([0.0], [1.0])),
        SHGO(
            topology=SAMBO.KNearestTopology(neighbors=1),
            local_solver=CoverageLocalSolver(),
            sampling_points=2,
            local_starts=1,
            local_budget=1,
        );
        maximum_evaluations=8,
        rng=MersenneTwister(824),
    )
    SAMBO._append_samples!(
        duplicate.workspace,
        reshape([0.5, 0.8], 1, :),
        [2.0, 3.0],
    )
    duplicate.workspace.complex = SAMBO.NeighborComplex([[2], [1]])
    push!(duplicate.workspace.minimizer_points, [0.5])
    push!(duplicate.workspace.minimizer_values, 2.0)
    SAMBO.update_minimizer_pool!(duplicate)
    @test duplicate.workspace.minimizer_values == [1.0]

    converged = init(
        Problem(x -> x[1]^2, Box([0.0], [1.0])),
        SHGO(
            topology=SAMBO.KNearestTopology(neighbors=1),
            sampling_points=2,
            homology_patience=1,
            minimization_schedule=SAMBO.MinimizeAtTermination(),
        );
        maximum_evaluations=40,
        rng=MersenneTwister(825),
    )
    converged.workspace.refinements = 1
    converged.workspace.stagnant_homology_iterations = 1
    @test SAMBO._step!(converged) isa SAMBO.Refined
    @test retcode(result(converged)) == :success

    finite_seeded = init(
        Problem(_ -> 0.0, SearchSpace(x=Choices(:only))),
        SCEUA(complexes=1, complex_size=2);
        initial_points=[(x=:only,)],
        initial_values=[0.0],
        maximum_evaluations=2,
        rng=MersenneTwister(826),
    )
    step!(finite_seeded)
    @test finite_seeded.workspace.occupied == BitSet([1])
    @test retcode(result(finite_seeded)) == :space_exhausted

    multiple_starts = init(
        Problem(x -> x[1]^2, Box([0.0], [1.0])),
        TopologicalMultistart(samples=4, local_starts=2);
        maximum_evaluations=8,
        rng=MersenneTwister(827),
    )
    multiple_starts.workspace.local_indices = [1, 2]
    multiple_starts.workspace.current_start = 1
    multiple_starts.workspace.sample_points[:, 2] .= 0.75
    multiple_starts.workspace.sample_values[2] = 0.5625
    @test SAMBO._finish_local_start!(multiple_starts) isa
        SAMBO.LocalStartExhausted
    @test multiple_starts.workspace.current_start == 2
    @test SAMBO.automatic_sampling_count(2, 100) == 5
end
