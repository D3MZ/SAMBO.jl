using Random
using SAMBO
using SHA

const DEFAULT_TRIALS = 1:5
const SOURCE_COMMIT = if haskey(ENV, "GITHUB_SHA")
    ENV["GITHUB_SHA"]
else
    try
        root = normpath(joinpath(@__DIR__, ".."))
        commit = readchomp(`git -C $root rev-parse HEAD`)
        dirty = !isempty(readchomp(
            `git -C $root status --porcelain --untracked-files=normal`,
        ))
        dirty ? "$commit-dirty" : commit
    catch
        "unknown"
    end
end

const ROTATION = [
    -0.6593804733957869 0.40611875581020845 0.16266300278010432 0.5752843417117289 0.20705946293608232
    -0.39562828403747224 -0.5245384304015962 0.2924634597652826 0.08246149657099422 -0.6899296501722388
    -0.19781414201873612 -0.43686232517528023 -0.8254056738772803 0.26904408514342243 0.12783437659867097
    -0.5934424260562083 0.17345445925725733 -0.1921454429053599 -0.7612651988361476 0.03598698837021451
    -0.13187609467915742 -0.5822300667409905 0.4120576106936707 -0.10167893122409566 0.6807986232723334
]
const SHIFT = [0.35, -0.55, 0.8, -0.25, 0.6]
const HARTMANN6_MINIMIZER =
    [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]

# These are nonseparable rotated-box robustness cases, not box-invariance tests.
function hartmann6(x)
    alpha = (1.0, 1.2, 3.0, 3.2)
    A = (
        (10.0, 3.0, 17.0, 3.5, 1.7, 8.0),
        (0.05, 10.0, 17.0, 0.1, 8.0, 14.0),
        (3.0, 3.5, 1.7, 10.0, 17.0, 8.0),
        (17.0, 8.0, 0.05, 10.0, 0.1, 14.0),
    )
    P = (
        (0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886),
        (0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991),
        (0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650),
        (0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381),
    )
    return -sum(
        alpha[i] * exp(-sum(A[i][j] * (x[j] - P[i][j])^2 for j in 1:6))
        for i in 1:4
    )
end

function transformed(x)
    return ROTATION * (x .- SHIFT)
end

function rotated_rastrigin(x)
    y = transformed(x)
    return 10length(y) + sum(value^2 - 10cos(2π * value) for value in y)
end

function rotated_rosenbrock(x)
    y = transformed(x) .+ 1
    return sum(
        100(y[index + 1] - y[index]^2)^2 + (1 - y[index])^2
        for index in 1:length(y)-1
    )
end

const CASES = (
    (
        name="hartmann6",
        problem=Problem(hartmann6, Box(zeros(6), ones(6))),
        optimum=-3.322368011415515,
        target=-2.8,
        quality_target=-2.0,
    ),
    (
        name="rotated_rastrigin5",
        problem=Problem(rotated_rastrigin, Box(fill(-5.12, 5), fill(5.12, 5))),
        optimum=0.0,
        target=12.0,
        quality_target=50.0,
    ),
    (
        name="rotated_rosenbrock5",
        problem=Problem(rotated_rosenbrock, Box(fill(-3.0, 5), fill(3.0, 5))),
        optimum=0.0,
        target=12.0,
        quality_target=200.0,
    ),
)

const ALGORITHMS = (
    ("SCE-UA", () -> SCEUA(), 1000),
    ("SMBO", () -> SMBO(candidate_pool=1024), 300),
    ("SHGO", () -> SHGO(sampling_points=128), 1000),
)

const FIXTURE_PATH = joinpath(@__DIR__, "exact_v1_fixture.csv")

function exact_fixture()
    lines = readlines(FIXTURE_PATH)
    streams = Dict{Tuple{String,String,Int},Vector{NamedTuple}}()
    for line in @view lines[2:end]
        fields = split(line, ','; keepempty=true)
        problem, algorithm = fields[1], fields[2]
        trial_id = parse(Int, fields[3])
        dimensions = problem == "hartmann6" ? 6 : 5
        item = (
            evaluation=parse(Int, fields[4]),
            phase=fields[5],
            pool_id=parse(Int, fields[6]),
            acquisition_coefficient=isempty(fields[7]) ?
                nothing : parse(Float64, fields[7]),
            checkpoint=parse(Int, fields[8]),
            latent=parse.(Float64, fields[9:8+dimensions]),
        )
        push!(get!(streams, (problem, algorithm, trial_id), NamedTuple[]), item)
    end
    budgets = Dict(name => budget for (name, _, budget) in ALGORITHMS)
    for ((problem, algorithm, _), stream) in streams
        budget = budgets[algorithm]
        getproperty.(stream, :evaluation) == collect(1:budget) ||
            error("exact-v1 evaluations must be contiguous")
        getproperty.(stream, :checkpoint) == collect(1:budget) ||
            error("exact-v1 checkpoints must be contiguous")
        phases = Set(getproperty.(stream, :phase))
        required = Dict(
            "SCE-UA" => Set(("initial_population", "replacement_sample")),
            "SMBO" => Set(("initial_population", "candidate_pool")),
            "SHGO" => Set(("randomized_shared_sampling",)),
        )[algorithm]
        phases == required ||
            error("invalid exact-v1 phases for $problem/$algorithm")
        algorithm != "SMBO" || all(
            item.acquisition_coefficient !== nothing
            for item in stream if item.phase == "candidate_pool"
        ) || error("exact-v1 SMBO pools require acquisition coefficients")
    end
    return streams
end

fixture_hash() = "sha256:" * bytes2hex(sha256(read(FIXTURE_PATH)))

function shared_initial_point(case, trial_id)
    dimensions = SAMBO.dimension(case.problem.space)
    latent = [
        mod(trial_id * (2axis + 1), 17) / 17
        for axis in 1:dimensions
    ]
    return decode(case.problem.space, latent)
end

initial_design_hash(case, trial_id) =
    "shared-design-v1:$(case.name):$trial_id"

shared_design_capability(algorithm_name) =
    algorithm_name == "SMBO" ?
    "injected-x0-y0" : "not-supported-cross-runtime"

benchmark_trials(algorithm_name, trials) =
    algorithm_name == "SHGO" ? (first(trials),) : trials

normalized_gap(value, optimum) =
    max(abs(value - optimum) / max(1, abs(optimum)), 1e-15)

function main(io=stdout; trials=DEFAULT_TRIALS, profile="native-default-v1")
    profile in ("native-default-v1", "exact-v1") ||
        throw(ArgumentError("unsupported correctness profile: $profile"))
    fixture = profile == "exact-v1" ? exact_fixture() : nothing
    exact_design_hash = profile == "exact-v1" ? fixture_hash() : nothing
    abs(hartmann6(HARTMANN6_MINIMIZER) + 3.322368011415515) < 1e-6 ||
        error("Hartmann-6 reference minimum is inconsistent")
    rotated_rastrigin(SHIFT) == 0 ||
        error("rotated Rastrigin reference minimum is inconsistent")
    rotated_rosenbrock(SHIFT) == 0 ||
        error("rotated Rosenbrock reference minimum is inconsistent")
    println(
        io,
        "runtime,runtime_version,source_commit,python_sambo_version,problem,algorithm,",
        "trial_id,profile,configuration_hash,initial_design_hash,initial_design_capability,",
        "budget,evaluation,evaluations,iteration,best_value,normalized_gap,",
        "minimum,optimum,target,quality_target,required_hit_rate,",
        "noninferiority_margin,feasible,duplicate,retcode,success",
    )
    for case in CASES,
            (algorithm_name, make_algorithm, budget) in ALGORITHMS,
            trial_id in (
                profile == "exact-v1" ?
                Tuple(trials) :
                benchmark_trials(algorithm_name, trials)
            )
        design_capability = shared_design_capability(algorithm_name)
        supports_shared_design = design_capability == "injected-x0-y0"
        if profile == "exact-v1"
            stream = fixture[(case.name, algorithm_name, trial_id)]
            length(stream) == budget ||
                error("exact-v1 fixture does not match budget")
            best = Inf
            seen = Set{Tuple}()
            design_hash = exact_design_hash
            for item in stream
                point = decode(case.problem.space, item.latent)
                value = case.problem.objective(point)
                best = min(best, value)
                key = Tuple(item.latent)
                duplicate = key in seen
                push!(seen, key)
                println(
                    io,
                    "Julia,$VERSION,$SOURCE_COMMIT,n/a,$(case.name),$algorithm_name,",
                    "$trial_id,$profile,exact-v1:$algorithm_name:$budget,",
                    "$design_hash,serialized-exact-replay,$budget,",
                    "$(item.evaluation),$(item.evaluation),$(item.checkpoint),$best,",
                    "$(normalized_gap(best, case.optimum)),$best,$(case.optimum),",
                    "$(case.target),$(case.quality_target),0.8,0.25,true,$duplicate,",
                    "$(item.evaluation == budget ? :exact_replay_completed : :running),",
                    "$(best <= case.target)",
                )
            end
            continue
        end
        result = if supports_shared_design
            initial_point = shared_initial_point(case, trial_id)
            initial_value = case.problem.objective(initial_point)
            solve(
                case.problem,
                make_algorithm();
                initial_points=[initial_point],
                initial_values=[initial_value],
                maximum_evaluations=budget,
                rng=Xoshiro(trial_id),
            )
        else
            solve(
                case.problem,
                make_algorithm();
                maximum_evaluations=budget,
                rng=Xoshiro(trial_id),
            )
        end
        configuration_hash = "$profile:$algorithm_name:$budget"
        design_hash = supports_shared_design ?
            initial_design_hash(case, trial_id) : "none"
        best = Inf
        seen = Set{Tuple}()
        for index in 1:result.trace.count
            result.trace.source[index] == KnownObservation && continue
            evaluation = result.trace.evaluation_numbers[index]
            value = result.trace.objective_values[index]
            best = min(best, value)
            latent = @view result.trace.latent_points[:, index]
            key = Tuple(latent)
            duplicate = key in seen
            push!(seen, key)
            result_code =
                evaluation == evaluation_count(result) ? retcode(result) : :running
            println(
                io,
                "Julia,$VERSION,$SOURCE_COMMIT,n/a,$(case.name),$algorithm_name,",
                "$trial_id,$profile,$configuration_hash,$design_hash,$design_capability,$budget,",
                "$evaluation,$evaluation,$(result.trace.iterations[index]),$best,",
                "$(normalized_gap(best, case.optimum)),$best,$(case.optimum),",
                "$(case.target),$(case.quality_target),0.8,0.25,true,$duplicate,",
                "$result_code,$(best <= case.target)",
            )
        end
    end
end

function requested_profile(args)
    index = findfirst(==("--profile"), args)
    !isnothing(index) && index < length(args) && return args[index + 1]
    return get(ENV, "CORRECTNESS_PROFILE", "native-default-v1")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    main(; profile=requested_profile(ARGS))
