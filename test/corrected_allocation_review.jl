module CorrectedAllocationReview

using SAMBO
using Random
using Test

_allocation_objective(point) = sum(abs2, point)

struct AllocationMeanSurrogate end
struct AllocationMeanModel end
SAMBO.fittedmodeltype(
    ::AllocationMeanSurrogate,
    ::Type{TX},
    ::Type{TY},
) where {TX,TY} = AllocationMeanModel
SAMBO.fitmodel(::AllocationMeanSurrogate, points, values, rng) =
    AllocationMeanModel()
function SAMBO.predictmean!(means, ::AllocationMeanModel, points)
    means .= @view points[1, :]
    return means
end

function _prepared_finite_ask(observations, seed)
    space = SearchSpace(choice=1:(observations + 1))
    return init(
        Problem(_ -> 0.0, space),
        SMBO(
            surrogate=AllocationMeanSurrogate(),
            acquisition=GreedyMean(),
            initial_points=0,
            candidate_pool=1,
        );
        initial_points=[
            (choice=index,) for index in 1:observations
        ],
        initial_values=zeros(observations),
        maximum_evaluations=observations + 1,
        rng=Xoshiro(seed),
    )
end

@testset "native warm allocation coverage" begin
    problem = Problem(
        _allocation_objective,
        Box(fill(-1.0, 2), fill(1.0, 2)),
    )

    smbo = init(
        problem,
        SMBO(initial_points=5, candidate_pool=128);
        maximum_evaluations=30,
        rng=Xoshiro(701),
    )
    initial = ask!(smbo, 5)
    tell!(smbo, initial, [sum(abs2, point) for point in initial])
    pool = @view smbo.workspace.candidates[:, 1:128]
    generate_candidates!(pool, smbo, smbo.algorithm.candidate_sampler)
    generation_bytes = @allocated generate_candidates!(
        pool,
        smbo,
        smbo.algorithm.candidate_sampler,
    )
    SAMBO._select_candidates(smbo, pool, 2)
    selection_bytes =
        @allocated SAMBO._select_candidates(smbo, pool, 2)
    @info "SMBO warm allocations" generation_bytes selection_bytes
    @test generation_bytes <= 65_536
    @test selection_bytes <= 150_000

    sceua = init(
        problem,
        SCEUA();
        maximum_evaluations=100,
        rng=Xoshiro(702),
    )
    step!(sceua)
    step!(sceua)
    sceua_bytes = @allocated step!(sceua)
    @info "SCE-UA warm step allocation" sceua_bytes
    @test sceua_bytes <= 2_048

    shgo = init(
        problem,
        SHGO(sampling_points=16);
        maximum_evaluations=100,
        rng=Xoshiro(703),
    )
    step!(shgo)
    refinement_bytes = @allocated step!(shgo)
    @info "SHGO warm refinement allocation" refinement_bytes
    @test refinement_bytes <= 2_000_000

    topological = init(
        problem,
        TopologicalMultistart(
            samples=16,
            local_starts=4,
            local_solver=SAMBO.PatternSearch(),
        );
        maximum_evaluations=80,
        rng=Xoshiro(704),
    )
    step!(topological)
    step!(topological)
    local_bytes = @allocated step!(topological)
    @info "topological warm local-step allocation" local_bytes
    @test local_bytes <= 4_096

    diagnostic_result = minimize(
        _allocation_objective,
        Box(fill(-1.0, 2), fill(1.0, 2));
        algorithm=SMBO(candidate_pool=32),
        maximum_evaluations=8,
        rng=Xoshiro(705),
    )
    pd_workspace =
        PartialDependenceWorkspace(Float64, Float64, 32, 8)
    partialdependence(
        diagnostic_result;
        dimensions=(1, 2),
        resolution=6,
        samples=8,
        workspace=pd_workspace,
        rng=Xoshiro(706),
    )
    pd_bytes = @allocated partialdependence(
        diagnostic_result;
        dimensions=(1, 2),
        resolution=6,
        samples=8,
        workspace=pd_workspace,
        rng=Xoshiro(706),
    )
    @info "partial-dependence warm chunk allocation" pd_bytes
    @test pd_bytes <= 250_000

    small_finite = _prepared_finite_ask(100, 707)
    large_finite = _prepared_finite_ask(10_000, 708)
    finite_small_bytes = @allocated ask!(small_finite, 1)
    finite_large_bytes = @allocated ask!(large_finite, 1)
    @info "finite ask warm allocations" finite_small_bytes finite_large_bytes
    @test finite_large_bytes <= finite_small_bytes + 4_096

    finite_selection = init(
        Problem(_ -> 0.0, SearchSpace(choice=1:4)),
        SMBO(
            surrogate=AllocationMeanSurrogate(),
            acquisition=GreedyMean(),
            initial_points=0,
            candidate_pool=4,
        );
        initial_points=[(choice=4,)],
        initial_values=[4.0],
        maximum_evaluations=4,
        rng=Xoshiro(709),
    )
    @test only(ask!(finite_selection, 1)).choice == 1
end

end
