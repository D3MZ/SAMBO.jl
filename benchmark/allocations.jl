using SAMBO
using Random

function allocation_report()
    rng = MersenneTwister(410)
    points = rand(rng, 3, 32)
    values = vec(sum(abs2, points; dims=1))
    model = SAMBO.fitmodel(GaussianProcessSurrogate(), points, values)
    queries = rand(rng, 3, 128)
    means = zeros(128)
    variances = zeros(128)
    prediction = GPPredictionWorkspace(Float64)
    SAMBO.predictmeanvariance!(
        means,
        variances,
        model,
        queries,
        prediction,
    )

    sce = init(
        Problem(x -> sum(abs2, x), Box(fill(-1.0, 3), fill(1.0, 3))),
        SCEUA();
        maximum_evaluations=200,
        rng=MersenneTwister(411),
    )
    step!(sce)
    step!(sce)

    smbo = init(
        Problem(x -> sum(abs2, x), Box(fill(-1.0, 3), fill(1.0, 3))),
        SMBO(candidate_pool=512);
        maximum_evaluations=50,
        rng=MersenneTwister(412),
    )
    initial = ask!(smbo, 7)
    tell!(smbo, initial, [sum(abs2, point) for point in initial])
    candidate_buffer = @view smbo.workspace.candidates[:, 1:512]

    return (
        gp_prediction=@allocated(SAMBO.predictmeanvariance!(
            means,
            variances,
            model,
            queries,
            prediction,
        )),
        sceua_step=@allocated(step!(sce)),
        smbo_candidate_generation=@allocated(generate_candidates!(
            candidate_buffer,
            smbo,
            smbo.algorithm.candidate_sampler,
        )),
        smbo_ask=@allocated(ask!(smbo, 1)),
    )
end

println(allocation_report())
