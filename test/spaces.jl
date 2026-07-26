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

    space = SearchSpace(count=0:10, rate=Continuous(0.1, 2.0), kind=Choices(:a, :b, :c))
    point = (count=7, rate=1.1, kind=:c)
    decoded = decode(space, encode(space, point))
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

    float32_box = Box(Float32[-1, 0], Float32[1, 2])
    @test Sambo.latenttype(float32_box) == Float32
    @test eltype(encode(float32_box, Float32[0, 1])) == Float32

    for design in (
        Sambo.UniformDesign(),
        Sambo.LatinHypercubeDesign(),
        Sambo.HaltonDesign(),
        Sambo.SobolDesign(),
    )
        samples = Matrix{Float64}(undef, 2, 16)
        Sambo.sample!(MersenneTwister(1), samples, design, box)
        @test all(0 .<= samples .<= 1)
        @test size(samples) == (2, 16)
    end
end
