@testset "spaces" begin
    box = Box([-2.0, 0.0], [2.0, 4.0])
    point = [1.0, 3.0]
    @test decode(box, encode(box, point)) ≈ point
    @test_throws DimensionMismatch decode(box, [0.5])
    @test_throws ArgumentError decode(box, [0.5, 1.1])
    @test_throws ArgumentError encode(box, [3.0, 1.0])

    fixed = Box([1.0], [1.0])
    @test encode(fixed, [1.0]) == [0.0]
    @test decode(fixed, [0.75]) == [1.0]
    fixed_samples = Matrix{Float64}(undef, 1, 8)
    SAMBO.sample!(MersenneTwister(3), fixed_samples, SAMBO.UniformDesign(), fixed)
    @test iszero(fixed_samples)

    space = SearchSpace(count=0:10, rate=Continuous(0.1, 2.0), kind=Choices(:a, :b, :c))
    point = (count=7, rate=1.1, kind=:c)
    decoded = @inferred decode(space, encode(space, point))
    @test decoded.count == point.count
    @test decoded.kind == point.kind
    @test decoded.rate ≈ point.rate
    @test decode(space, [0, 0, 0]) == (count=0, rate=0.1, kind=:a)

    reordered = (kind=:c, count=7, rate=1.1)
    @test decode(space, encode(space, reordered)) == decoded
    @test_throws ArgumentError encode(space, (count=11, rate=1.1, kind=:a))
    @test_throws ArgumentError encode(space, (count=1, rate=1.1))
    @test_throws DimensionMismatch decode(space, [0.0, 0.0, 0.0, 0.0])
    @test_throws ArgumentError SearchSpace(x=1:0)
    @test typeof(space.dimensions) <: NamedTuple

    categories = SearchSpace((Choices(:a, :b, :c),))
    @test decode(categories, [0.0]) == (:a,)
    @test decode(categories, [1 / 3 - eps()]) == (:a,)
    @test decode(categories, [1 / 3]) == (:b,)
    @test decode(categories, [2 / 3]) == (:c,)
    @test decode(categories, [1.0]) == (:c,)
    for coordinate in range(0, 1; length=101)
        encoded = encode(categories, decode(categories, [coordinate]))
        @test encoded[1] in (0.0, 0.5, 1.0)
    end

    fixed_mixed = SearchSpace(
        fixed=Continuous(2.0, 2.0),
        singleton=Choices(:only),
        integer=4:4,
    )
    fixed_mixed_samples = Matrix{Float64}(undef, 3, 8)
    SAMBO.sample!(
        MersenneTwister(4),
        fixed_mixed_samples,
        SAMBO.LatinHypercubeDesign(),
        fixed_mixed,
    )
    @test iszero(fixed_mixed_samples)

    lower = [-1.0]
    upper = [1.0]
    copied_box = Box(lower, upper)
    lower[1] = -10
    upper[1] = 10
    @test copied_box.lower == [-1.0]
    @test copied_box.upper == [1.0]

    @test_throws ArgumentError Continuous(0.0, Inf)
    @test_throws ArgumentError Choices(:a, :a)
    @test_throws ArgumentError Box(Float64[], Float64[])
    @test_throws ArgumentError Box([0.0], [Inf])
    @test_throws ArgumentError SearchSpace(())
    @test_throws ArgumentError SearchSpace((nothing,))
    @test_throws ArgumentError HaltonDesign(skip=-1)
    @test_throws ArgumentError SobolDesign(skip=-1)
    @test_throws ArgumentError SAMBO.GlobalLocalCandidates(global_fraction=1.1)
    @test_throws ArgumentError SAMBO.GlobalLocalCandidates(local_scale=0)

    big_box = Box(BigFloat[-1, 0], BigFloat[1, 2])
    @test SAMBO.latenttype(big_box) == BigFloat
    @test eltype(encode(big_box, BigFloat[0, 1])) == BigFloat
    @test space_cardinality(SearchSpace(a=Choices(:x, :y), b=1:3)) == 6
    @test space_cardinality(SearchSpace(a=Continuous(1.0, 1.0), b=1:3)) == 3
    @test isnothing(space_cardinality(SearchSpace(a=Continuous(0.0, 1.0))))
    @test SAMBO.latenttype(SearchSpace(Float32; x=Continuous(0, 1))) == Float32
    @test SAMBO.latenttype(SearchSpace(BigFloat; x=Continuous(0, 1))) == BigFloat
    @test_throws ArgumentError SearchSpace(evaluation=1:2)

    for design in (
        SAMBO.UniformDesign(),
        SAMBO.LatinHypercubeDesign(),
        SAMBO.HaltonDesign(),
        SAMBO.SobolDesign(),
    )
        samples = Matrix{Float64}(undef, 2, 16)
        SAMBO.sample!(MersenneTwister(1), samples, design, box)
        @test all(0 .<= samples .<= 1)
        @test size(samples) == (2, 16)
    end
end
