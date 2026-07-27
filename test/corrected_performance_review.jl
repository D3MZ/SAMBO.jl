module CorrectedPerformanceReviewTests

using SAMBO
using Random
using Test

_review_objective(point) = sum(abs2, point)

function _prepared_full_tell_state(seed)
    state = init(
        Problem(_review_objective, Box(fill(-1.0, 2), fill(1.0, 2))),
        SMBO(initial_points=64, candidate_pool=128, batch_size=64);
        maximum_evaluations=128,
        rng=Xoshiro(seed),
    )
    batch = ask!(state, 64)
    values = vec(sum(abs2, latentpoints(batch); dims=1))
    return state, batch, values
end

@testset "corrected performance review" begin
    @testset "unconstrained sampling fast path" begin
        problem = Problem(Box(fill(-1.0, 8), fill(1.0, 8)))
        destination = Matrix{Float64}(undef, 8, 4096)
        rng = Xoshiro(8)
        SAMBO._sample_feasible!(
            rng,
            destination,
            UniformDesign(),
            problem,
        )
        bytes = @allocated SAMBO._sample_feasible!(
            rng,
            destination,
            UniformDesign(),
            problem,
        )
        @info "unconstrained sampling warm allocation" bytes
        @test bytes <= 512
    end

    @testset "encode! is the nonallocating primitive" begin
        box = Box([-1.0, -2.0], [1.0, 2.0])
        mixed = SearchSpace(
            count=0:10,
            rate=Continuous(0.1, 2.0),
            kind=Choices(:a, :b),
        )
        box_buffer = zeros(2)
        mixed_buffer = zeros(3)
        box_point = [0.5, -0.5]
        mixed_point = (count=3, rate=0.8, kind=:b)
        SAMBO.encode!(box_buffer, box, box_point)
        SAMBO.encode!(mixed_buffer, mixed, mixed_point)
        box_bytes = @allocated SAMBO.encode!(box_buffer, box, box_point)
        mixed_bytes =
            @allocated SAMBO.encode!(mixed_buffer, mixed, mixed_point)
        @info "encode! warm allocations" box_bytes mixed_bytes
        @test box_bytes == 0
        @test mixed_bytes == 0
        @test encode(box, box_point) == box_buffer
        @test encode(mixed, mixed_point) == mixed_buffer
    end

    @testset "post-fit SMBO function barrier" begin
        state = init(
            Problem(_review_objective, Box(fill(-1.0, 2), fill(1.0, 2))),
            SMBO(
                initial_points=5,
                candidate_pool=64,
                refit_schedule=FixedRefit(100),
            );
            maximum_evaluations=20,
            rng=Xoshiro(9),
        )
        initial = ask!(state, 5)
        tell!(state, initial, [sum(abs2, point) for point in initial])
        warm = ask!(state, 1)
        tell!(state, warm, [sum(abs2, point) for point in warm])
        @test !isnothing(state.model)
        @test length(ask!(state, 1)) == 1
        @test any(method -> method.nargs == 7, methods(SAMBO._predict_candidates!))
    end

    @testset "finite occupancy uses canonical integer identifiers" begin
        space = SearchSpace(choice=Choices(1:100...))
        points = [(choice=index,) for index in 1:100]
        state = init(
            Problem(_ -> 0.0, space),
            SMBO(initial_points=0, candidate_pool=128);
            initial_points=points,
            initial_values=zeros(100),
            maximum_evaluations=101,
            rng=Xoshiro(10),
        )
        @test state.occupied isa Union{BitSet,Set{Int}}
        @test length(state.occupied) == 100
        @test isempty(ask!(state, 1))
        @test retcode(result(state)) == :space_exhausted
    end

    @testset "batch-policy deterministic fixture" begin
        candidates = reshape([0.0, 0.01, 1.0], 1, :)
        function selected(strategy)
            state = init(
                Problem(SearchSpace(x=Continuous(0.0, 1.0))),
                SMBO(
                    acquisition=GreedyMean(),
                    batch_strategy=strategy,
                    repeat_policy=AllowRepeatedEvaluations(),
                );
                initial_points=[(x=0.5,)],
                initial_values=[0.5],
                maximum_evaluations=4,
                rng=Xoshiro(11),
            )
            return SAMBO._select_candidates(state, candidates, 2)
        end
        @test selected(GreedyBatch()) == reshape([0.0, 0.01], 1, :)
        @test selected(LocalPenalization(strength=10.0, radius=0.1)) ==
            reshape([0.0, 1.0], 1, :)
    end

    @testset "ensemble prediction workspace" begin
        rng = Xoshiro(12)
        specification = EnsembleSurrogate(
            GaussianProcessSurrogate(),
            5,
        )
        training_points = rand(rng, 2, 20)
        training_values = vec(sum(abs2, training_points; dims=1))
        queries = rand(rng, 2, 128)
        model = SAMBO.fitmodel(
            specification,
            training_points,
            training_values,
            rng,
        )
        means = zeros(128)
        variances = zeros(128)
        workspace = SAMBO.predictionworkspace(specification, Float64)
        SAMBO.predictmeanvariance!(
            means,
            variances,
            model,
            queries,
            workspace,
        )
        bytes = @allocated SAMBO.predictmeanvariance!(
            means,
            variances,
            model,
            queries,
            workspace,
        )
        @info "ensemble prediction warm allocation" bytes
        @test bytes <= 512
    end

    @testset "topological proposal workspace" begin
        state = init(
            Problem(_review_objective, Box(fill(-1.0, 2), fill(1.0, 2))),
            TopologicalMultistart(samples=16, local_starts=4);
            maximum_evaluations=80,
            rng=Xoshiro(13),
        )
        step!(state)
        step!(state)
        bytes = @allocated step!(state)
        @info "TopologicalMultistart warm step allocation" bytes
        @test bytes <= 4_096
    end

    @testset "full-batch tell! fast path" begin
        warm_state, warm_batch, warm_values =
            _prepared_full_tell_state(14)
        tell!(warm_state, warm_batch, warm_values)

        state, batch, values = _prepared_full_tell_state(15)
        bytes = @allocated tell!(state, batch, values)
        @info "full-batch tell! warm allocation" bytes
        @test objectivevalues(trace(result(state))) == values
        @test isempty(state.pending)
        @test bytes <= 4_096
    end
end

end
