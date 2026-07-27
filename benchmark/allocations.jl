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
    generate_candidates!(
        candidate_buffer,
        smbo,
        smbo.algorithm.candidate_sampler,
    )
    SAMBO._select_candidates(smbo, candidate_buffer, 1)
    warm_ask = ask!(smbo, 1)
    cancel!(smbo, warm_ask)

    shgo = init(
        Problem(x -> sum(abs2, x), Box(fill(-1.0, 2), fill(1.0, 2))),
        SHGO(sampling_points=16);
        maximum_evaluations=100,
        rng=MersenneTwister(413),
    )
    step!(shgo)
    SAMBO.local_minimize!(
        shgo,
        shgo.algorithm.local_solver,
        fill(0.5, 2),
        0.0,
        zeros(2),
        ones(2),
        8,
    )

    partial_result = minimize(
        x -> sum(abs2, x),
        Box(fill(-1.0, 2), fill(1.0, 2));
        algorithm=SMBO(candidate_pool=64),
        maximum_evaluations=12,
        rng=MersenneTwister(414),
    )
    partial_workspace =
        PartialDependenceWorkspace(Float64, Float64, 64, 16)
    partialdependence(
        partial_result;
        dimensions=(1, 2),
        resolution=8,
        samples=16,
        workspace=partial_workspace,
        rng=MersenneTwister(415),
    )

    function prepared_finite_ask(observations, seed)
        space = SearchSpace(choice=1:(observations + 1))
        return init(
            Problem(_ -> 0.0, space),
            SMBO(initial_points=0, candidate_pool=8);
            initial_points=[
                (choice=index,) for index in 1:observations
            ],
            initial_values=zeros(observations),
            maximum_evaluations=observations + 1,
            rng=MersenneTwister(seed),
        )
    end
    finite_100 = prepared_finite_ask(100, 416)
    finite_10_000 = prepared_finite_ask(10_000, 417)

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
        smbo_selection=@allocated(SAMBO._select_candidates(
            smbo,
            candidate_buffer,
            1,
        )),
        smbo_ask=@allocated(ask!(smbo, 1)),
        shgo_refinement=@allocated(step!(shgo)),
        shgo_local_step=@allocated(SAMBO.local_minimize!(
            shgo,
            shgo.algorithm.local_solver,
            fill(0.5, 2),
            0.0,
            zeros(2),
            ones(2),
            8,
        )),
        partial_dependence_chunk=@allocated(partialdependence(
            partial_result;
            dimensions=(1, 2),
            resolution=8,
            samples=16,
            workspace=partial_workspace,
            rng=MersenneTwister(415),
        )),
        finite_ask_100=@allocated(ask!(finite_100, 1)),
        finite_ask_10_000=@allocated(ask!(finite_10_000, 1)),
    )
end

println(allocation_report())
