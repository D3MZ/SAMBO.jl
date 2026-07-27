module CorrectedNativeContractTests

using Random
using SAMBO
using Test

struct ContractReplayCandidates{M}
    points::M
end

function SAMBO.generate_candidates!(
    destination,
    state,
    sampler::ContractReplayCandidates,
)
    count = min(size(destination, 2), size(sampler.points, 2))
    copyto!(
        @view(destination[:, 1:count]),
        @view(sampler.points[:, 1:count]),
    )
    return count
end

@testset "corrected native contracts" begin
    @testset "parameterized lifecycle across algorithms" begin
        for algorithm in (
            SCEUA(complexes=1, complex_size=3),
            SMBO(initial_points=3, candidate_pool=32),
            TopologicalMultistart(
                samples=8,
                topology=KNearestTopology(neighbors=2),
            ),
            SHGO(
                sampling_points=4,
                topology=KNearestTopology(neighbors=2),
                local_budget=2,
            ),
        )
            objective_calls = Ref(0)
            constraint_calls = Ref(0)
            callback_calls = Ref(0)
            parameters = (target=0.25, limit=0.9)
            problem = Problem(
                (point, p) -> begin
                    objective_calls[] += 1
                    @test point isa Vector{Float64}
                    abs2(point[1] - p.target)
                end,
                Box([0.0], [1.0]);
                parameters,
                constraint=(point, p) -> begin
                    constraint_calls[] += 1
                    point[1] <= p.limit
                end,
            )
            solved = solve(
                problem,
                algorithm;
                maximum_evaluations=8,
                rng=Xoshiro(101),
                callback=event -> begin
                    callback_calls[] += 1
                    @test event.latest_point isa Vector{Float64}
                    false
                end,
            )
            @test objective_calls[] == evaluation_count(solved)
            @test constraint_calls[] >= evaluation_count(solved)
            @test callback_calls[] >= 1
            @test all(
                point -> point[1] <= parameters.limit,
                (decode(solved.space, latent) for
                 latent in eachcol(latentpoints(trace(solved)))),
            )
        end
    end

    @testset "mixed native values and capability restrictions" begin
        space = SearchSpace(
            rate=Continuous(0.0, 1.0),
            count=0:2,
            kind=Choices(:a, :b),
        )
        for algorithm in (
            SCEUA(complexes=1, complex_size=4),
            SMBO(initial_points=3, candidate_pool=32),
        )
            objective_types = Set{DataType}()
            constraint_types = Set{DataType}()
            callback_types = Set{DataType}()
            parameters = (preferred=:b,)
            solved = solve(
                Problem(
                    (point, p) -> begin
                        push!(objective_types, typeof(point))
                        (point.rate - 0.4)^2 + (point.count - 1)^2 +
                            (point.kind == p.preferred ? 0.0 : 1.0)
                    end,
                    space;
                    parameters,
                    constraint=(point, p) -> begin
                        push!(constraint_types, typeof(point))
                        point.kind == p.preferred
                    end,
                ),
                algorithm;
                maximum_evaluations=8,
                rng=Xoshiro(102),
                callback=event -> begin
                    push!(callback_types, typeof(event.latest_point))
                    false
                end,
            )
            expected = NamedTuple{
                (:rate, :count, :kind),
                Tuple{Float64,Int,Symbol},
            }
            @test objective_types == Set([expected])
            @test constraint_types == Set([expected])
            @test callback_types == Set([expected])
            @test bestpoint(solved) isa expected
        end
        problem = Problem(_ -> 0.0, space)
        @test_throws ArgumentError init(problem, TopologicalMultistart())
        @test_throws ArgumentError init(problem, SHGO())
    end

    @testset "callback stop is atomic across algorithms" begin
        for algorithm in (
            SCEUA(complexes=1, complex_size=3),
            SMBO(initial_points=3, candidate_pool=32),
            TopologicalMultistart(
                samples=8,
                topology=KNearestTopology(neighbors=2),
            ),
            SHGO(
                sampling_points=4,
                topology=KNearestTopology(neighbors=2),
                local_budget=2,
            ),
        )
            events = BatchProgressEvent[]
            solved = solve(
                Problem(x -> sum(abs2, x), Box([-1.0], [1.0])),
                algorithm;
                maximum_evaluations=20,
                rng=Xoshiro(103),
                callback=event -> begin
                    push!(events, event)
                    true
                end,
            )
            @test length(events) == 1
            @test retcode(solved) == :callback_stop
            @test evaluation_count(solved) == events[1].evaluation
            @test events[1].batch_size == evaluation_count(solved)
        end
    end

    @testset "objective failure leaves a batch uncommitted" begin
        calls = Ref(0)
        state = init(
            Problem(
                point -> begin
                    calls[] += 1
                    calls[] == 2 && error("deliberate batch failure")
                    sum(abs2, point)
                end,
                Box([-1.0], [1.0]),
            ),
            SMBO(initial_points=3, batch_size=3);
            maximum_evaluations=3,
            rng=Xoshiro(104),
        )
        @test_throws ErrorException step!(state)
        @test calls[] == 2
        @test evaluation_count(result(state)) == 0
        @test isempty(objectivevalues(trace(state)))
        @test length(state.pending) == 1

        threaded = init(
            Problem(_ -> error("deliberate threaded failure"), Box([-1.0], [1.0])),
            SMBO(initial_points=3, batch_size=3);
            maximum_evaluations=3,
            rng=Xoshiro(104),
            executor=Threaded(),
        )
        @test_throws CompositeException step!(threaded)
        @test evaluation_count(result(threaded)) == 0
        @test isempty(objectivevalues(trace(threaded)))
        @test length(threaded.pending) == 1
    end

    @testset "maximization duality with replayed candidates" begin
        pool = reshape(collect(range(0.0, 1.0; length=17)), 1, :)
        sampler = ContractReplayCandidates(pool)
        algorithm = SMBO(
            initial_points=0,
            candidate_pool=size(pool, 2),
            batch_size=2,
            candidate_sampler=sampler,
        )
        maximum_problem = Problem(
            x -> -(x[1] - 0.7)^2,
            Box([0.0], [1.0]);
            sense=Maximize(),
        )
        minimum_problem = Problem(
            x -> (x[1] - 0.7)^2,
            Box([0.0], [1.0]),
        )
        maximum_state = init(
            maximum_problem,
            algorithm;
            initial_points=[[0.25]],
            initial_values=[-(0.25 - 0.7)^2],
            maximum_evaluations=6,
            rng=Xoshiro(105),
        )
        minimum_state = init(
            minimum_problem,
            algorithm;
            initial_points=[[0.25]],
            initial_values=[(0.25 - 0.7)^2],
            maximum_evaluations=6,
            rng=Xoshiro(105),
        )
        while retcode(result(maximum_state)) == :running
            maximum_batch = ask!(maximum_state, 2)
            minimum_batch = ask!(minimum_state, 2)
            @test latentpoints(maximum_batch) ==
                latentpoints(minimum_batch)
            tell!(
                maximum_state,
                maximum_batch,
                [-(point[1] - 0.7)^2 for point in maximum_batch],
            )
            tell!(
                minimum_state,
                minimum_batch,
                [(point[1] - 0.7)^2 for point in minimum_batch],
            )
        end
        maximum_result = result(maximum_state)
        minimum_result = result(minimum_state)
        @test bestpoint(maximum_result) == bestpoint(minimum_result)
        @test bestvalue(maximum_result) == -bestvalue(minimum_result)
    end

    @testset "mixed sampling covers levels and fixed dimensions" begin
        space = SearchSpace(
            count=0:2,
            kind=Choices(:a, :b, :c),
            fixed=Choices(:only),
        )
        latent = Matrix{Float64}(undef, 3, 6_000)
        SAMBO.sample!(Xoshiro(106), latent, UniformDesign(), space)
        points = decode.(Ref(space), eachcol(latent))
        @test Set(point.count for point in points) == Set(0:2)
        @test Set(point.kind for point in points) == Set((:a, :b, :c))
        @test all(point -> point.fixed == :only, points)
        @test all(iszero, @view latent[3, :])
        for level in 0:2
            level_count = count(point -> point.count == level, points)
            @test 1_700 <= level_count <= 2_300
        end
        for level in (:a, :b, :c)
            level_count = count(point -> point.kind == level, points)
            @test 1_700 <= level_count <= 2_300
        end
    end
end

end
