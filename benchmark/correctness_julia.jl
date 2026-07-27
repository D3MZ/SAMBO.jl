using Random
using LinearAlgebra
using SAMBO

const DEFAULT_TRIALS = 1:10
const ROTATION_IDS = 1:6
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

function challenge_rotation(dimensions, rotation_id)
    rotation = Matrix{Float64}(I, dimensions, dimensions)
    offset = mod(rotation_id - 1, dimensions - 1) + 1
    for axis in 1:dimensions
        other = mod(axis + offset - 1, dimensions) + 1
        angle = (mod(rotation_id * (2axis + 1), 19) + 1) * π / 37
        cosine = cos(angle)
        sine = sin(angle)
        for column in 1:dimensions
            left = rotation[axis, column]
            right = rotation[other, column]
            rotation[axis, column] = cosine * left - sine * right
            rotation[other, column] = sine * left + cosine * right
        end
    end
    return rotation
end

function transformed(x, rotation)
    return rotation * (x .- SHIFT)
end

function rotated_rastrigin(x, rotation)
    y = transformed(x, rotation)
    return 10length(y) + sum(value^2 - 10cos(2π * value) for value in y)
end

function rotated_rosenbrock(x, rotation)
    y = transformed(x, rotation) .+ 1
    return sum(
        100(y[index + 1] - y[index]^2)^2 + (1 - y[index])^2
        for index in 1:length(y)-1
    )
end

function oriented_hartmann6(x, rotation_id)
    oriented = similar(x)
    for axis in 1:6
        source = mod(axis + rotation_id - 2, 6) + 1
        reflected = iseven(axis + rotation_id)
        oriented[axis] = reflected ? 1 - x[source] : x[source]
    end
    return hartmann6(oriented)
end

function hartmann_preimage(point, rotation_id)
    preimage = similar(point)
    for axis in 1:6
        source = mod(axis + rotation_id - 2, 6) + 1
        reflected = iseven(axis + rotation_id)
        preimage[source] = reflected ? 1 - point[axis] : point[axis]
    end
    return preimage
end

function benchmark_cases()
    cases = NamedTuple[]
    for rotation_id in ROTATION_IDS
        rotation = challenge_rotation(5, rotation_id)
        push!(
            cases,
            (
                name="hartmann6",
                rotation_id,
                problem=Problem(
                    x -> oriented_hartmann6(x, rotation_id),
                    Box(zeros(6), ones(6)),
                ),
                optimum=-3.322368011415515,
            ),
            (
                name="rotated_rastrigin5",
                rotation_id,
                problem=Problem(
                    x -> rotated_rastrigin(x, rotation),
                    Box(fill(-5.12, 5), fill(5.12, 5)),
                ),
                optimum=0.0,
            ),
            (
                name="rotated_rosenbrock5",
                rotation_id,
                problem=Problem(
                    x -> rotated_rosenbrock(x, rotation),
                    Box(fill(-3.0, 5), fill(3.0, 5)),
                ),
                optimum=0.0,
            ),
        )
    end
    return Tuple(cases)
end

const CASES = benchmark_cases()

function python_sambo_sceua_profile()
    algorithm = SCEUA(
        complex_size=2,
        objective_tolerance=1e-6,
        population_tolerance=1e-6,
        stall_iterations=30,
    )
    @assert algorithm.complexes == 0
    @assert algorithm.complex_size == 2
    @assert algorithm.reflection == 1.0
    @assert algorithm.contraction == 0.5
    @assert algorithm.repair isa SAMBO.MoveTowardCentroid
    @assert algorithm.objective_tolerance == 1e-6
    @assert algorithm.population_tolerance == 1e-6
    @assert algorithm.stall_iterations == 30
    return algorithm
end

const ALGORITHMS = (
    ("SCE-UA", python_sambo_sceua_profile, 1000),
    ("SMBO", () -> SMBO(), 100),
    ("SHGO", () -> SHGO(PythonSAMBOProfile()), 1000),
)

function shared_initial_design(case, count, trial_id)
    dimensions = SAMBO.dimension(case.problem.space)
    design = Vector{Vector{Float64}}(undef, count)
    latent = Matrix{Float64}(undef, dimensions, count)
    for axis in 1:dimensions
        multiplier = 2axis + trial_id + case.rotation_id
        while gcd(multiplier, count) != 1
            multiplier += 1
        end
        offset = mod(
            trial_id * (11 + 2(axis - 1)) +
            case.rotation_id * (17 + axis - 1),
            count,
        )
        for column in 1:count
            stratum = mod(multiplier * (column - 1) + offset, count)
            jitter = mod(
                trial_id * 101 +
                case.rotation_id * 211 +
                axis * 307 +
                column * 401,
                997,
            ) / 997
            latent[axis, column] = (stratum + jitter) / count
        end
    end
    for column in 1:count
        design[column] = decode(
            case.problem.space,
            @view(latent[:, column]),
        )
    end
    return design
end

initial_design_hash(case, trial_id) =
    "shared-counted-lhs-v3:$(case.name):$(case.rotation_id):$trial_id"

shared_design_capability(algorithm_name) =
    algorithm_name in ("SCE-UA", "SMBO") ?
    "injected-counted-lhs" : "not-supported-cross-runtime"

shared_halton_shift(dimensions, rotation_id, trial_id) = [
    mod(
        trial_id * 127 +
        rotation_id * 283 +
        axis * 419,
        997,
    ) / 997
    for axis in 1:dimensions
]

function matched_algorithm(algorithm_name, make_algorithm, case, trial_id)
    algorithm_name == "SHGO" || return make_algorithm()
    dimensions = SAMBO.dimension(case.problem.space)
    return SHGO(
        PythonSAMBOProfile();
        sampling=SAMBO.FixedShiftDesign(
            HaltonDesign(skip=0),
            shared_halton_shift(
                dimensions,
                case.rotation_id,
                trial_id,
            ),
        ),
    )
end

benchmark_trials(algorithm_name, trials) =
    Tuple(trials)

normalized_gap(value, optimum) =
    max(abs(value - optimum) / max(1, abs(optimum)), 1e-15)

function main(io=stdout; trials=DEFAULT_TRIALS)
    abs(hartmann6(HARTMANN6_MINIMIZER) + 3.322368011415515) < 1e-6 ||
        error("Hartmann-6 reference minimum is inconsistent")
    for rotation_id in ROTATION_IDS
        abs(
            oriented_hartmann6(
                hartmann_preimage(HARTMANN6_MINIMIZER, rotation_id),
                rotation_id,
            ) + 3.322368011415515,
        ) < 1e-6 ||
            error("oriented Hartmann-6 reference minimum is inconsistent")
        rotation = challenge_rotation(5, rotation_id)
        rotated_rastrigin(SHIFT, rotation) == 0 ||
            error("rotated Rastrigin reference minimum is inconsistent")
        rotated_rosenbrock(SHIFT, rotation) == 0 ||
            error("rotated Rosenbrock reference minimum is inconsistent")
    end
    println(
        io,
        "runtime,runtime_version,source_commit,python_sambo_version,problem,algorithm,",
        "rotation_id,trial_id,configuration_hash,initial_design_hash,",
        "initial_design_capability,",
        "budget,evaluation,evaluations,iteration,best_value,normalized_gap,",
        "minimum,optimum,noninferiority_margin,feasible,duplicate,retcode",
    )
    for case in CASES,
            (algorithm_name, make_algorithm, budget) in ALGORITHMS,
            trial_id in benchmark_trials(algorithm_name, trials)
        design_capability = shared_design_capability(algorithm_name)
        supports_shared_design = design_capability == "injected-counted-lhs"
        result = if supports_shared_design
            dimensions = SAMBO.dimension(case.problem.space)
            initial_count = if algorithm_name == "SCE-UA"
                profile = make_algorithm()
                complex_size = profile.complex_size == 0 ? 2 : profile.complex_size
                min(
                    max(2, budget ÷ complex_size - 1),
                    max(5, floor(Int, 3log2(dimensions))),
                ) * complex_size
            else
                min(
                    max(1, budget - 20),
                    floor(Int, 40dimensions * max(1, log2(dimensions))),
                )
            end
            solve(
                case.problem,
                matched_algorithm(
                    algorithm_name,
                    make_algorithm,
                    case,
                    trial_id,
                );
                initial_points=shared_initial_design(
                    case,
                    initial_count,
                    trial_id,
                ),
                maximum_evaluations=budget,
                rng=Xoshiro(trial_id),
            )
        else
            solve(
                case.problem,
                matched_algorithm(
                    algorithm_name,
                    make_algorithm,
                    case,
                    trial_id,
                );
                maximum_evaluations=budget,
                rng=Xoshiro(trial_id),
            )
        end
        configuration_hash =
            "python-sambo-1.25.2-matched-v5:$algorithm_name:$budget:6-rotations"
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
                "$(case.rotation_id),$trial_id,$configuration_hash,$design_hash,",
                "$design_capability,$budget,",
                "$evaluation,$evaluation,$(result.trace.iterations[index]),$best,",
                "$(normalized_gap(best, case.optimum)),$best,$(case.optimum),",
                "0.25,true,$duplicate,$result_code",
            )
        end
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
