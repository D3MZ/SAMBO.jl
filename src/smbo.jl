struct UniformCandidates end
struct AdaptiveDensityCandidates end

_silverman_bandwidth(weights, dimensions) =
    (inv(sum(abs2, weights)) * (dimensions + 2) / 4)^
    (-inv(dimensions + 4))
struct AvoidRepeatedEvaluations end
struct AllowRepeatedEvaluations end
struct CandidateGenerationError <: Exception
    message::String
end
struct ExactCandidateEquality end
struct ApproximateCandidateEquality{T<:Real}
    tolerance::T
    function ApproximateCandidateEquality(tolerance::T) where {T<:Real}
        isfinite(tolerance) && tolerance >= 0 ||
            throw(ArgumentError("candidate tolerance must be finite and nonnegative"))
        new{T}(tolerance)
    end
end
ApproximateCandidateEquality(; tolerance=sqrt(eps(Float64))) =
    ApproximateCandidateEquality(tolerance)
Base.showerror(io::IO, error::CandidateGenerationError) =
    print(io, error.message)
struct GreedyBatch end
struct LocalPenalization{T<:Real}
    strength::T
    radius::T
    function LocalPenalization(strength::T, radius::T) where {T<:Real}
        isfinite(strength) && strength >= 0 ||
            throw(ArgumentError("penalization strength must be finite and nonnegative"))
        isfinite(radius) && radius > 0 ||
            throw(ArgumentError("penalization radius must be finite and positive"))
        new{T}(strength, radius)
    end
end
function LocalPenalization(; strength=1.0, radius=0.05)
    return LocalPenalization(promote(strength, radius)...)
end
struct EliteGaussianCandidates{T<:Real}
    scale::T
    function EliteGaussianCandidates(scale::T) where {T<:Real}
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("local candidate scale must be finite and positive"))
        new{T}(scale)
    end
end
EliteGaussianCandidates() = EliteGaussianCandidates(0.15)
struct MixtureCandidates{G,L,T<:Real}
    global_sampler::G
    local_sampler::L
    global_fraction::T
    function MixtureCandidates(
        global_sampler::G,
        local_sampler::L,
        global_fraction::T,
    ) where {G,L,T<:Real}
        isfinite(global_fraction) && 0 <= global_fraction <= 1 ||
            throw(ArgumentError("global_fraction must lie in [0, 1]"))
        new{G,L,T}(global_sampler, local_sampler, global_fraction)
    end
end
function MixtureCandidates(
    global_sampler,
    local_sampler;
    global_fraction=0.35,
)
    return MixtureCandidates(
        global_sampler,
        local_sampler,
        global_fraction,
    )
end
GlobalLocalCandidates(; global_fraction=0.25, local_scale=0.1) =
    MixtureCandidates(
        UniformCandidates(),
        EliteGaussianCandidates(local_scale);
        global_fraction,
    )
struct FixedRefit
    interval::Int
    FixedRefit(interval=1) = interval > 0 ?
        new(interval) : throw(ArgumentError("refit interval must be positive"))
end
struct SquareRootRefit
    minimum_interval::Int
    SquareRootRefit(minimum_interval=1) = minimum_interval > 0 ?
        new(minimum_interval) :
        throw(ArgumentError("minimum refit interval must be positive"))
end
refit_interval(schedule::FixedRefit, observations) = schedule.interval
refit_interval(schedule::SquareRootRefit, observations) =
    max(schedule.minimum_interval, isqrt(observations))

struct SMBO{S,A,I,C,F,R,B,E,T<:Real}
    surrogate::S
    acquisition::A
    initial_design::I
    candidate_sampler::C
    initial_points::Int
    candidate_pool::Int
    batch_size::Int
    refit_schedule::F
    prediction_chunk_size::Int
    repeat_policy::R
    batch_strategy::B
    candidate_equality::E
    improvement_tolerance::T
    no_change_iterations::Int
end
function SMBO(;
    surrogate=GaussianProcessSurrogate(
        length_scale=AutomaticARDLengthScale(),
        noise=1e-14,
        jitter=NoJitter(),
        kernel=SquaredExponentialKernel(),
        optimize_hyperparameters=true,
    ),
    acquisition=RandomizedLowerConfidenceBound(),
    initial_design=LatinHypercubeDesign(),
    candidate_sampler=AdaptiveDensityCandidates(),
    initial_points=0,
    candidate_pool=80_000,
    batch_size=1,
    refit_interval=1,
    refit_schedule=nothing,
    prediction_chunk_size=4096,
    adaptive_refit=false,
    repeat_policy=AvoidRepeatedEvaluations(),
    batch_strategy=LocalPenalization(),
    candidate_equality=ExactCandidateEquality(),
    exploration=nothing,
    improvement_tolerance=1e-6,
    no_change_iterations=5,
)
    if !isnothing(exploration)
        acquisition = LowerConfidenceBound(exploration)
    end
    initial_points >= 0 || throw(ArgumentError("initial_points must be nonnegative"))
    candidate_pool > 0 || throw(ArgumentError("candidate_pool must be positive"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    refit_interval > 0 || throw(ArgumentError("refit_interval must be positive"))
    prediction_chunk_size > 0 ||
        throw(ArgumentError("prediction_chunk_size must be positive"))
    isfinite(improvement_tolerance) && improvement_tolerance >= 0 ||
        throw(ArgumentError("improvement tolerance must be finite and nonnegative"))
    no_change_iterations > 0 ||
        throw(ArgumentError("no-change iterations must be positive"))
    schedule = isnothing(refit_schedule) ?
        (adaptive_refit ? SquareRootRefit(refit_interval) :
            FixedRefit(refit_interval)) : refit_schedule
    return SMBO(
        surrogate,
        acquisition,
        initial_design,
        candidate_sampler,
        initial_points,
        candidate_pool,
        batch_size,
        schedule,
        prediction_chunk_size,
        repeat_policy,
        batch_strategy,
        candidate_equality,
        improvement_tolerance,
        no_change_iterations,
    )
end

struct CandidateBatch{T,S}
    identifier::UInt64
    latent_points::Matrix{T}
    space::S
end
Base.length(batch::CandidateBatch) = size(batch.latent_points, 2)
Base.isempty(batch::CandidateBatch) = length(batch) == 0
Base.getindex(batch::CandidateBatch, index) =
    decode(batch.space, @view batch.latent_points[:, index])
Base.iterate(batch::CandidateBatch, index=1) =
    index > length(batch) ? nothing : (batch[index], index + 1)
latentpoints(batch::CandidateBatch) = batch.latent_points

mutable struct PendingBatch{T}
    points::Matrix{T}
    unresolved::BitVector
end

mutable struct SMBOWorkspace{TX,TY,P}
    means::Vector{TY}
    variances::Vector{TY}
    scores::Vector{TY}
    selected::Matrix{TX}
    prediction::P
    candidates::Matrix{TX}
    order::Vector{Int}
    available::BitVector
end

mutable struct SMBOState{C,A,T,W,M,O,FI}
    core::C
    algorithm::A
    model::M
    pending::Dict{UInt64,PendingBatch{T}}
    next_identifier::UInt64
    observations_at_fit::Int
    failures::Vector{Any}
    initial_design::Matrix{T}
    initial_design_cursor::Int
    workspace::W
    occupied::O
    feasible_indices::FI
end

function init(problem::Problem, algorithm::SMBO; initial_points=nothing, initial_values=nothing, kwargs...)
    core = _makecore(problem; kwargs...)
    _seed_initial!(core, initial_points, initial_values)
    T = eltype(core.trace.latent_points)
    TY = eltype(core.trace.objective_values)
    d = dimension(problem.space)
    initial_count = algorithm.initial_points == 0 ? min(
        max(1, core.criteria.maximum_evaluations - 20),
        floor(Int, 40d * max(1, log2(d))),
    ) :
        algorithm.initial_points
    design_count = min(
        max(0, initial_count - core.trace.count),
        _remaining(core),
    )
    design = try
        design_count == 0 ?
            Matrix{T}(undef, d, 0) :
            _sample_feasible(core, design_count, algorithm.initial_design)
    catch error
        error isa InfeasibleSpaceError || rethrow()
        core.retcode = :infeasible_space
        Matrix{T}(undef, d, 0)
    end
    design = _prepare_initial_design(
        design,
        algorithm.repeat_policy,
        algorithm.candidate_equality,
    )
    workspace = SMBOWorkspace(
        TY[],
        TY[],
        TY[],
        Matrix{T}(undef, d, 0),
        predictionworkspace(algorithm.surrogate, TY),
        Matrix{T}(undef, d, max(algorithm.candidate_pool, 8algorithm.batch_size)),
        Int[],
        BitVector(),
    )
    cardinality = space_cardinality(problem.space)
    occupied = _occupancy(
        algorithm.repeat_policy,
        algorithm.candidate_equality,
        cardinality,
        T,
    )
    feasible_indices = _tracks_occupancy(
        algorithm.repeat_policy,
        algorithm.candidate_equality,
        cardinality,
    ) ? _finite_feasible_indices(problem) : nothing
    _initialize_occupancy!(
        occupied,
        core,
        algorithm.repeat_policy,
        algorithm.candidate_equality,
        cardinality,
    )
    Model = Union{Nothing,fittedmodeltype(algorithm.surrogate, T, TY)}
    return SMBOState{
        typeof(core),
        typeof(algorithm),
        T,
        typeof(workspace),
        Model,
        typeof(occupied),
        typeof(feasible_indices),
    }(
        core,
        algorithm,
        nothing,
        Dict{UInt64,PendingBatch{T}}(),
        UInt64(1),
        0,
        Any[],
        design,
        1,
        workspace,
        occupied,
        feasible_indices,
    )
end

function _fit_smbo!(state::SMBOState; force=false)
    trace = state.core.trace
    trace.count > 0 || return state
    interval = refit_interval(state.algorithm.refit_schedule, trace.count)
    if force || isnothing(state.model) ||
            trace.count - state.observations_at_fit >= interval
        state.model = fitmodel(
            state.algorithm.surrogate,
            latentpoints(trace),
            _surrogate_values(
                state.core.problem.sense,
                objectivevalues(trace),
            ),
            state.core.rng,
        )
        state.observations_at_fit = trace.count
    end
    return state
end

_surrogate_values(::Minimize, values) = values
_surrogate_values(::Maximize, values) = -values

function ask!(state::SMBOState, requested::Integer=state.algorithm.batch_size)
    requested >= 0 || throw(ArgumentError("batch size must be nonnegative"))
    _finished(state.core) && return CandidateBatch(
        UInt64(0),
        Matrix{eltype(state.core.trace.latent_points)}(
            undef,
            dimension(state.core.problem.space),
            0,
        ),
        state.core.problem.space,
    )
    reserved = sum(
        pending -> Base.count(pending.unresolved),
        values(state.pending);
        init=0,
    )
    cardinality = space_cardinality(state.core.problem.space)
    if _space_exhausted(state.algorithm.repeat_policy, state, cardinality)
        state.core.retcode = :space_exhausted
        return CandidateBatch(
            UInt64(0),
            Matrix{eltype(state.core.trace.latent_points)}(
                undef,
                dimension(state.core.problem.space),
                0,
            ),
            state.core.problem.space,
        )
    end
    count = min(Int(requested), max(0, _remaining(state.core) - reserved))
    T = eltype(state.core.trace.latent_points)
    if count == 0
        return CandidateBatch(
            UInt64(0),
            Matrix{T}(undef, dimension(state.core.problem.space), 0),
            state.core.problem.space,
        )
    end
    available_design = size(state.initial_design, 2) -
        state.initial_design_cursor + 1
    finite_pool_count = if isnothing(state.feasible_indices)
        count
    elseif state.core.trace.count == 0
        count
    else
        min(
            length(state.feasible_indices) - length(state.occupied),
            max(count, state.algorithm.candidate_pool),
        )
    end
    finite_candidates =
        _unused_finite_candidates(state, finite_pool_count)
    if !isnothing(finite_candidates)
        candidates = state.core.trace.count == 0 ?
            finite_candidates :
            _select_candidates(state, finite_candidates, count)
    elseif available_design > 0
        design_count = min(count, available_design)
        first = state.initial_design_cursor
        last = first + design_count - 1
        candidates = Matrix(@view state.initial_design[:, first:last])
        state.initial_design_cursor = last + 1
        if design_count < count
            candidates = hcat(
                candidates,
                _sample_feasible(
                    state.core,
                    count - design_count,
                    UniformDesign(),
                ),
            )
        end
    elseif state.core.trace.count == 0
        expanded = _sample_feasible(
            state.core,
            reserved + count,
            state.algorithm.initial_design,
        )
        candidates = Matrix(
            @view(expanded[:, reserved+1:reserved+count]),
        )
    else
        pool_size = max(state.algorithm.candidate_pool, 8count)
        _ensure_candidate_capacity!(
            state.workspace,
            dimension(state.core.problem.space),
            pool_size,
        )
        pool = @view state.workspace.candidates[:, 1:pool_size]
        generated = generate_candidates!(
            pool,
            state,
            state.algorithm.candidate_sampler,
        )
        pool = @view pool[:, 1:generated]
        candidates = _select_candidates(state, pool, count)
    end
    candidates = _deduplicate_candidates(state, candidates, count)
    count = size(candidates, 2)
    count == 0 && begin
        !isnothing(cardinality) && _occupied_count(state) >= cardinality &&
            (state.core.retcode = :space_exhausted)
        state.core.retcode == :running && throw(CandidateGenerationError(
            "candidate generator produced no usable candidates",
        ))
        return CandidateBatch(
            UInt64(0),
            candidates,
            state.core.problem.space,
        )
    end
    identifier = state.next_identifier
    state.next_identifier += UInt64(1)
    state.pending[identifier] = PendingBatch(copy(candidates), trues(count))
    _occupy!(state, candidates)
    return CandidateBatch(identifier, candidates, state.core.problem.space)
end

function step!(state::SMBOState)
    batch = ask!(state)
    isempty(batch) && !_finished(state.core) &&
        throw(AssertionError("ask! returned no candidates without terminating"))
    isempty(batch) && return state
    values = _evaluate_batch(state.core, batch.latent_points)
    return _tell!(
        state,
        batch,
        findall(_pending_batch(state, batch).unresolved),
        values,
        InternalEvaluation,
    )
end

function solve!(state::SMBOState)
    isnothing(state.core.problem.objective) &&
        throw(ArgumentError("solve! requires an objective; use ask!/tell!"))
    try
        previous_best = eltype(state.core.trace.objective_values)(Inf)
        no_change = 0
        while !_finished(state.core)
            guided = state.initial_design_cursor > size(state.initial_design, 2)
            step!(state)
            if guided && !_finished(state.core)
                best = _loss(state.core.problem, state.core.best_value)
                if previous_best == best ||
                        previous_best - best <
                        state.algorithm.improvement_tolerance
                    no_change += 1
                    if no_change == state.algorithm.no_change_iterations
                        state.core.retcode = :stalled
                    end
                else
                    previous_best = best
                    no_change = 0
                end
            end
        end
        _fit_smbo!(state; force=true)
    catch error
        if error isa InfeasibleSpaceError
            state.core.retcode = :infeasible_space
        elseif error isa NumericalFailureError
            state.core.retcode = :numerical_failure
        else
            rethrow()
        end
    end
    return result(state)
end
solve(problem::Problem, algorithm::SMBO; kwargs...) = solve!(init(problem, algorithm; kwargs...))
trace(state::SMBOState) = state.core.trace
result(state::SMBOState) = _result(
    state.core,
    state.algorithm,
    state.model;
    statistics=(
        iterations=state.core.iteration,
        pending_batches=length(state.pending),
        failed_candidates=length(state.failures),
        model_current=state.observations_at_fit == state.core.trace.count,
    ),
)
