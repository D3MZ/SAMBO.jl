module NativeDefaultReviewTests

using LinearAlgebra
using Random
using SAMBO
using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
const README = read(joinpath(ROOT, "README.md"), String)
const JULIA_BENCHMARK =
    read(joinpath(ROOT, "benchmark", "benchmarks.jl"), String)
const PYTHON_BENCHMARK =
    read(joinpath(ROOT, "benchmark", "python_reference.py"), String)
const JULIA_CORRECTNESS =
    read(joinpath(ROOT, "benchmark", "correctness_julia.jl"), String)
const PYTHON_CORRECTNESS =
    read(joinpath(ROOT, "benchmark", "correctness_python.py"), String)

function table_algorithms(markdown, heading)
    start = findfirst(heading, markdown)
    isnothing(start) && return Set{String}()
    remainder = markdown[last(start)+1:end]
    next_heading = findfirst("\n### ", remainder)
    section = isnothing(next_heading) ?
        remainder :
        remainder[1:first(next_heading)-1]
    return Set(
        match.captures[1]
        for match in eachmatch(
            r"(?m)^\| (SCE-UA|SMBO|SHGO) \|",
            section,
        )
    )
end

@testset "Native-default benchmark review" begin
    @testset "matched correctness workload is documented" begin
        @test occursin("matched algorithm settings", lowercase(README))
        @test !occursin("equivalent seeded Rosenbrock workloads", README)
        @test occursin("nonseparable rotations", lowercase(README))
    end

    @testset "rotation diagnostic and classification" begin
        rotation = [
            -0.6593804733957869 0.40611875581020845 0.16266300278010432 0.5752843417117289 0.20705946293608232
            -0.39562828403747224 -0.5245384304015962 0.2924634597652826 0.08246149657099422 -0.6899296501722388
            -0.19781414201873612 -0.43686232517528023 -0.8254056738772803 0.26904408514342243 0.12783437659867097
            -0.5934424260562083 0.17345445925725733 -0.1921454429053599 -0.7612651988361476 0.03598698837021451
            -0.13187609467915742 -0.5822300667409905 0.4120576106936707 -0.10167893122409566 0.6807986232723334
        ]
        @test rotation' * rotation ≈ Matrix{Float64}(I, 5, 5) atol=1e-14
        @test det(rotation) ≈ 1.0 atol=1e-14
        @test !all(abs.(rotation * fill(5.12, 5)) .<= 5.12)
        @test occursin("nonseparable", lowercase(JULIA_CORRECTNESS))
        @test occursin("nonseparable", lowercase(PYTHON_CORRECTNESS))
    end

    @testset "documented benchmarks are executable" begin
        documented =
            table_algorithms(README, "### Rosenbrock microbenchmarks")
        @test documented == Set(("SCE-UA", "SMBO", "SHGO"))
        for algorithm in documented
            @test occursin("\"$algorithm\"", JULIA_BENCHMARK)
            @test occursin("\"$algorithm\"", PYTHON_BENCHMARK)
        end

        @test occursin("objective_calls = Ref(0)", JULIA_BENCHMARK)
        @test occursin("objective_calls[] += 1", JULIA_BENCHMARK)
        @test occursin("objective_calls[]", JULIA_BENCHMARK)
        @test occursin("objective_calls += 1", PYTHON_BENCHMARK)
    end

    @testset "public extension protocol" begin
        @test Base.isexported(SAMBO, :TopologicalMultistart)

        protocol_docs = join(
            (
                read(joinpath(ROOT, "spec", "api.md"), String),
                read(joinpath(ROOT, "docs", "src", "index.md"), String),
            ),
            "\n",
        )
        for hook in (
            :fitmodel,
            :predictmean!,
            :predictmeanvariance!,
            :predictionworkspace,
            :clone_surrogate,
            :generate_candidates!,
            :local_minimize!,
        )
            @test Base.isexported(SAMBO, hook)
            @test occursin(string(hook), protocol_docs)
        end

        for accessor in (
            :problem,
            :space,
            :rng,
            :remaining_evaluations,
            :iteration,
            :evaluate!,
            :isfeasible,
        )
            @test Base.isexported(SAMBO, accessor)
        end

        state = init(
            Problem(x -> sum(abs2, x), Box([-1.0], [1.0])),
            SCEUA(complexes=1, complex_size=2);
            maximum_evaluations=4,
            rng=Xoshiro(17),
        )
        candidates = fill(0.5, 1, 1)
        values = zeros(1)
        evaluate!(state, values, candidates)
        @test evaluation_count(result(state)) == 1
        @test remaining_evaluations(state) == 3
        @test problem(state) isa Problem
        @test space(state) === problem(state).space
        @test rng(state) isa Xoshiro
        @test iteration(state) == 0
        @test isfeasible(state, @view candidates[:, 1])
    end

    @testset "documentation contains behavioral contracts" begin
        make_source = read(joinpath(ROOT, "docs", "make.jl"), String)
        for (page, specification) in (
            ("api", "api.md"),
            ("spaces", "spaces.md"),
            ("termination", "termination.md"),
            ("interop", "interop.md"),
        )
            @test occursin(specification, make_source)
            @test isfile(joinpath(ROOT, "docs", "src", page, "index.md"))
        end
    end
end

end
