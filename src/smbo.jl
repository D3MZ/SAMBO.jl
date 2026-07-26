struct UniformCandidates end
struct AvoidRepeatedEvaluations end
struct AllowRepeatedEvaluations end
struct GreedyBatch end
struct LocalPenalization{T<:Real}
    strength::T
    radius::T
end
function LocalPenalization(; strength=1.0, radius=0.05)
    isfinite(strength) && strength >= 0 ||
        throw(ArgumentError("penalization strength must be finite and nonnegative"))
    isfinite(radius) && radius > 0 ||
        throw(ArgumentError("penalization radius must be finite and positive"))
    return LocalPenalization(promote(strength, radius)...)
end
struct GlobalLocalCandidates{T<:Real}
    global_fraction::T
    local_scale::T
end
function GlobalLocalCandidates(; global_fraction=0.35, local_scale=0.15)
    isfinite(global_fraction) && 0 <= global_fraction <= 1 ||
        throw(ArgumentError("global_fraction must lie in [0, 1]"))
    isfinite(local_scale) && local_scale > 0 ||
        throw(ArgumentError("local_scale must be finite and positive"))
    return GlobalLocalCandidates(promote(global_fraction, local_scale)...)
end

struct SMBO{S,A,I,C,R,B}
    surrogate::S
    acquisition::A
    initial_design::I
    candidate_sampler::C
    initial_points::Int
    candidate_pool::Int
    batch_size::Int
    refit_interval::Int
    prediction_chunk_size::Int
    adaptive_refit::Bool
    repeat_policy::R
    batch_strategy::B
end
function SMBO(;
    surrogate=GaussianProcessSurrogate(),
    acquisition=LowerConfidenceBound(),
    initial_design=LatinHypercubeDesign(),
    candidate_sampler=GlobalLocalCandidates(),
    initial_points=0,
    candidate_pool=4096,
    batch_size=1,
    refit_interval=1,
    prediction_chunk_size=4096,
    adaptive_refit=true,
    repeat_policy=AvoidRepeatedEvaluations(),
    batch_strategy=LocalPenalization(),
    exploration=nothing,
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
    return SMBO(
        surrogate,
        acquisition,
        initial_design,
        candidate_sampler,
        initial_points,
        candidate_pool,
        batch_size,
        refit_interval,
        prediction_chunk_size,
        adaptive_refit,
        repeat_policy,
        batch_strategy,
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

mutable struct SMBOWorkspace{TX,TY}
    means::Vector{TY}
    variances::Vector{TY}
    scores::Vector{TY}
    selected::Matrix{TX}
end

mutable struct SMBOState{C,A,T,W}
    core::C
    algorithm::A
    model::Any
    pending::Dict{UInt64,PendingBatch{T}}
    next_identifier::UInt64
    observations_at_fit::Int
    failures::Vector{Any}
    initial_design::Matrix{T}
    initial_design_cursor::Int
    workspace::W
end

function init(problem::Problem, algorithm::SMBO; initial_points=nothing, initial_values=nothing, kwargs...)
    core = _makecore(problem; kwargs...)
    _seed_initial!(core, initial_points, initial_values)
    T = eltype(core.trace.latent_points)
    TY = eltype(core.trace.objective_values)
    d = dimension(problem.space)
    initial_count = algorithm.initial_points == 0 ? max(2d + 1, 5) :
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
    workspace = SMBOWorkspace(
        TY[],
        TY[],
        TY[],
        Matrix{T}(undef, d, 0),
    )
    return SMBOState(
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
    )
end

function _fit_smbo!(state::SMBOState; force=false)
    trace = state.core.trace
    trace.count > 0 || return state
    interval = state.algorithm.adaptive_refit ?
        max(state.algorithm.refit_interval, floor(Int, sqrt(trace.count))) :
        state.algorithm.refit_interval
    if force || isnothing(state.model) ||
            trace.count - state.observations_at_fit >= interval
        state.model = fitmodel(
            state.algorithm.surrogate,
            latentpoints(trace),
            _loss.(Ref(state.core.problem.sense), objectivevalues(trace)),
            state.core.rng,
        )
        state.observations_at_fit = trace.count
    end
    return state
end

function _ensure_smbo_workspace!(state::SMBOState, candidates, requested)
    workspace = state.workspace
    count = size(candidates, 2)
    resize!(workspace.means, count)
    resize!(workspace.variances, count)
    resize!(workspace.scores, count)
    if size(workspace.selected) != (size(candidates, 1), requested)
        workspace.selected = Matrix{eltype(candidates)}(
            undef,
            size(candidates, 1),
            requested,
        )
    end
    return workspace
end

function _global_local_candidates(state::SMBOState, count, ::UniformCandidates)
    return _sample_feasible(state.core, count, UniformDesign())
end

function _global_local_candidates(state::SMBOState, count, sampler::GlobalLocalCandidates)
    core = state.core
    T = eltype(core.trace.latent_points)
    d = dimension(core.problem.space)
    candidates = Matrix{T}(undef, d, count)
    global_count = clamp(round(Int, count * sampler.global_fraction), 1, count)
    _sample_feasible!(
        core.rng,
        @view(candidates[:, 1:global_count]),
        UniformDesign(),
        core.problem,
    )
    global_count == count && return candidates

    trace = core.trace
    order = sortperm(_loss.(Ref(core.problem.sense), objectivevalues(trace)))
    elite_count = clamp(round(Int, sqrt(trace.count)), 1, trace.count)
    write = global_count
    attempts = 0
    while write < count && attempts < 200count
        attempts += 1
        rank = Random.rand(core.rng, 1:elite_count)
        center = @view trace.latent_points[:, order[rank]]
        proposal = @view candidates[:, write + 1]
        scale = T(sampler.local_scale) / sqrt(T(1 + core.iteration) / T(2))
        for row in 1:d
            proposal[row] = center[row] + scale * Random.randn(core.rng, T)
        end
        project!(proposal, core.problem.space)
        _canonicalize!(proposal, core.problem.space)
        isfeasible(core.problem, decode(core.problem.space, proposal)) || continue
        write += 1
    end
    if write < count
        _sample_feasible!(
            core.rng,
            @view(candidates[:, write+1:count]),
            UniformDesign(),
            core.problem,
        )
    end
    return candidates
end

@inline function _same_candidate(left, right)
    length(left) == length(right) || return false
    tolerance = sqrt(eps(promote_type(eltype(left), eltype(right))))
    @inbounds for i in eachindex(left, right)
        abs(left[i] - right[i]) <= tolerance || return false
    end
    return true
end

function _is_duplicate(state::SMBOState, candidate, selected, selected_count)
    state.algorithm.repeat_policy isa AllowRepeatedEvaluations && return false
    trace = state.core.trace
    for column in 1:trace.count
        _same_candidate(candidate, @view(trace.latent_points[:, column])) && return true
    end
    for pending in values(state.pending), column in axes(pending.points, 2)
        pending.unresolved[column] || continue
        _same_candidate(candidate, @view(pending.points[:, column])) && return true
    end
    for column in 1:selected_count
        _same_candidate(candidate, @view(selected[:, column])) && return true
    end
    return false
end

function _predict_candidates!(state, means, variances, candidates)
    chunk = state.algorithm.prediction_chunk_size
    for first in 1:chunk:size(candidates, 2)
        last = min(first + chunk - 1, size(candidates, 2))
        predictmeanvariance!(
            @view(means[first:last]),
            @view(variances[first:last]),
            state.model,
            @view(candidates[:, first:last]),
        )
    end
    return means, variances
end

function _penalized_score(
    score,
    candidate,
    selected,
    selected_count,
    ::GreedyBatch,
)
    return score
end
function _penalized_score(
    score,
    candidate,
    selected,
    selected_count,
    strategy::LocalPenalization,
)
    selected_count == 0 && return score
    distance_squared = minimum(1:selected_count) do column
        sum(abs2, candidate .- @view(selected[:, column]))
    end
    penalty = strategy.strength *
        exp(-distance_squared / (2strategy.radius^2))
    return score + penalty
end

function _deduplicated_fallback!(state, selected, count, requested)
    attempts = 0
    while count < requested && attempts < 100requested
        attempts += 1
        fallback = _sample_feasible(state.core, 1, UniformDesign())
        candidate = @view fallback[:, 1]
        _is_duplicate(state, candidate, selected, count) && continue
        count += 1
        selected[:, count] .= candidate
    end
    return count
end

function _occupied_count(state::SMBOState)
    occupied = Set{Tuple}()
    trace = state.core.trace
    for column in 1:trace.count
        push!(occupied, Tuple(@view trace.latent_points[:, column]))
    end
    for pending in values(state.pending), column in axes(pending.points, 2)
        pending.unresolved[column] &&
            push!(occupied, Tuple(@view pending.points[:, column]))
    end
    return length(occupied)
end

function _deduplicate_candidates(state, candidates, requested)
    state.algorithm.repeat_policy isa AllowRepeatedEvaluations &&
        return candidates
    selected = Matrix{eltype(candidates)}(
        undef,
        size(candidates, 1),
        requested,
    )
    count = 0
    for column in axes(candidates, 2)
        candidate = @view candidates[:, column]
        _is_duplicate(state, candidate, selected, count) && continue
        count += 1
        selected[:, count] .= candidate
        count == requested && break
    end
    count < requested &&
        (count = _deduplicated_fallback!(
            state,
            selected,
            count,
            requested,
        ))
    return Matrix(@view selected[:, 1:count])
end

function _select_candidates(state::SMBOState, candidates, requested)
    workspace = _ensure_smbo_workspace!(state, candidates, requested)
    selected = workspace.selected
    if state.core.trace.count == 0
        selected .= @view candidates[:, 1:requested]
        return copy(selected)
    end
    _fit_smbo!(state)
    means = workspace.means
    variances = workspace.variances
    scores = workspace.scores
    _predict_candidates!(state, means, variances, candidates)
    acquisitionvalues!(
        scores,
        state.algorithm.acquisition,
        means,
        variances,
        _loss(state.core.problem, state.core.best_value),
    )
    shortlist_count = min(length(scores), max(32, 16requested))
    order = partialsortperm(scores, 1:shortlist_count)
    count = 0
    available = trues(length(order))
    while count < requested && any(available)
        choice = 0
        choice_score = eltype(scores)(Inf)
        for position in eachindex(order)
            available[position] || continue
            index = order[position]
            candidate = @view candidates[:, index]
            adjusted = _penalized_score(
                scores[index],
                candidate,
                selected,
                count,
                state.algorithm.batch_strategy,
            )
            if adjusted < choice_score
                choice = position
                choice_score = adjusted
            end
        end
        iszero(choice) && break
        available[choice] = false
        index = order[choice]
        candidate = @view candidates[:, index]
        _is_duplicate(state, candidate, selected, count) && continue
        count += 1
        selected[:, count] .= candidate
    end
    if count < requested
        count = _deduplicated_fallback!(
            state,
            selected,
            count,
            requested,
        )
    end
    return copy(@view selected[:, 1:count])
end

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
    if state.algorithm.repeat_policy isa AvoidRepeatedEvaluations &&
            !isnothing(cardinality) &&
            _occupied_count(state) >= cardinality
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
    if available_design > 0
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
                    state.algorithm.initial_design,
                ),
            )
        end
    elseif state.core.trace.count == 0
        candidates = _sample_feasible(
            state.core,
            count,
            state.algorithm.initial_design,
        )
    else
        pool_size = max(state.algorithm.candidate_pool, 8count)
        pool = _global_local_candidates(state, pool_size, state.algorithm.candidate_sampler)
        candidates = _select_candidates(state, pool, count)
    end
    candidates = _deduplicate_candidates(state, candidates, count)
    count = size(candidates, 2)
    count == 0 && begin
        state.core.retcode = :space_exhausted
        return CandidateBatch(
            UInt64(0),
            candidates,
            state.core.problem.space,
        )
    end
    identifier = state.next_identifier
    state.next_identifier += UInt64(1)
    state.pending[identifier] = PendingBatch(copy(candidates), trues(count))
    return CandidateBatch(identifier, candidates, state.core.problem.space)
end

function _pending_batch(state::SMBOState, batch::CandidateBatch)
    pending = get(state.pending, batch.identifier, nothing)
    isnothing(pending) && throw(ArgumentError("unknown or completed batch"))
    batch.latent_points == pending.points ||
        throw(ArgumentError("candidate batch was modified after ask!"))
    return pending
end

function _validated_pending_indices(pending::PendingBatch, indices)
    selected = collect(Int, indices)
    isempty(selected) && return selected
    allunique(selected) || throw(ArgumentError("batch indices must be unique"))
    all(index -> 1 <= index <= size(pending.points, 2), selected) ||
        throw(BoundsError(pending.points, (:, selected)))
    all(index -> pending.unresolved[index], selected) ||
        throw(ArgumentError("batch candidate was already completed"))
    return selected
end

function _resolve_pending!(state::SMBOState, identifier, pending, indices)
    pending.unresolved[indices] .= false
    !any(pending.unresolved) && delete!(state.pending, identifier)
    return state
end

function _tell!(
    state::SMBOState,
    batch::CandidateBatch,
    indices,
    values,
    source::ObservationSource,
)
    pending = _pending_batch(state, batch)
    selected = _validated_pending_indices(pending, indices)
    candidates = pending.points[:, selected]
    converted = _validated_values(state.core, candidates, values)
    length(selected) <= _remaining(state.core) ||
        throw(ArgumentError("completed candidates exceed the evaluation budget"))
    _resolve_pending!(state, batch.identifier, pending, selected)
    state.core.iteration += 1
    _commit_batch!(state.core, candidates, converted; source)
    return state
end
tell!(state::SMBOState, batch::CandidateBatch, values) =
    _tell!(
        state,
        batch,
        findall(_pending_batch(state, batch).unresolved),
        values,
        ExternalEvaluation,
    )
tell!(state::SMBOState, batch::CandidateBatch, indices, values) =
    _tell!(state, batch, indices, values, ExternalEvaluation)

function cancel!(state::SMBOState, batch::CandidateBatch, indices)
    pending = _pending_batch(state, batch)
    selected = _validated_pending_indices(pending, indices)
    _resolve_pending!(state, batch.identifier, pending, selected)
    return state
end
cancel!(state::SMBOState, batch::CandidateBatch) =
    cancel!(state, batch, findall(_pending_batch(state, batch).unresolved))

function fail!(state::SMBOState, batch::CandidateBatch, indices, errors)
    pending = _pending_batch(state, batch)
    selected = _validated_pending_indices(pending, indices)
    collected_errors = collect(errors)
    length(collected_errors) == length(selected) ||
        throw(DimensionMismatch("one error per failed candidate required"))
    for (index, error) in zip(selected, collected_errors)
        push!(
            state.failures,
            (batch=batch.identifier, index=index, error=error),
        )
    end
    _resolve_pending!(state, batch.identifier, pending, selected)
    return state
end
fail!(state::SMBOState, batch::CandidateBatch, errors) =
    fail!(
        state,
        batch,
        findall(_pending_batch(state, batch).unresolved),
        errors,
    )

function _resolve_explicit_pending!(state::SMBOState, candidates)
    for column in axes(candidates, 2)
        candidate = @view candidates[:, column]
        for (identifier, pending) in collect(state.pending)
            matched = findfirst(axes(pending.points, 2)) do index
                pending.unresolved[index] &&
                    _same_candidate(candidate, @view(pending.points[:, index]))
            end
            isnothing(matched) || begin
                _resolve_pending!(state, identifier, pending, [matched])
                break
            end
        end
    end
    return state
end

function tell!(state::SMBOState, points, values)
    latent = _points_to_latent(state.core.problem.space, points)
    for column in axes(latent, 2)
        point = decode(state.core.problem.space, @view(latent[:, column]))
        isfeasible(state.core.problem, point) ||
            throw(ArgumentError("explicit point $column is infeasible"))
    end
    supplied_values = values isa Real ? [values] : values
    converted = _validated_values(state.core, latent, supplied_values)
    size(latent, 2) <= _remaining(state.core) ||
        throw(ArgumentError("completed candidates exceed the evaluation budget"))
    _resolve_explicit_pending!(state, latent)
    state.core.iteration += 1
    _commit_batch!(state.core, latent, converted; source=ExternalEvaluation)
    return state
end

function step!(state::SMBOState)
    batch = ask!(state)
    if isempty(batch)
        state.core.retcode = :evaluation_limit
        return state
    end
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
        while !_finished(state.core)
            step!(state)
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

struct SMBOCheckpoint{A,M,P,R,E,C,TR,SC,N,T,F,I,W}
    algorithm::A
    model::M
    pending::P
    rng::R
    executor::E
    callback::C
    trace::TR
    criteria::SC
    nonfinite::N
    iteration::Int
    retcode::Symbol
    best_value::T
    best_index::Int
    last_significant_improvement_evaluation::Int
    evaluations::Int
    next_identifier::UInt64
    observations_at_fit::Int
    failures::F
    initial_design::I
    initial_design_cursor::Int
    workspace::W
    elapsed_seconds::Float64
end

function checkpoint(state::SMBOState)
    core = state.core
    return SMBOCheckpoint(
        state.algorithm,
        deepcopy(state.model),
        deepcopy(state.pending),
        deepcopy(core.rng),
        core.executor,
        core.callback,
        _snapshot(core.trace),
        core.criteria,
        core.nonfinite,
        core.iteration,
        core.retcode,
        core.best_value,
        core.best_index,
        core.last_significant_improvement_evaluation,
        core.evaluations,
        state.next_identifier,
        state.observations_at_fit,
        deepcopy(state.failures),
        copy(state.initial_design),
        state.initial_design_cursor,
        deepcopy(state.workspace),
        (time_ns() - core.started) / 1e9,
    )
end

function restore(problem::Problem, saved::SMBOCheckpoint)
    elapsed_nanoseconds = UInt64(round(saved.elapsed_seconds * 1e9))
    now = time_ns()
    started = now - min(now, elapsed_nanoseconds)
    core = SolverCore(
        problem,
        deepcopy(saved.rng),
        saved.executor,
        saved.callback,
        _snapshot(saved.trace),
        saved.criteria,
        saved.nonfinite,
        saved.iteration,
        started,
        saved.retcode,
        saved.best_value,
        saved.best_index,
        saved.last_significant_improvement_evaluation,
        saved.evaluations,
    )
    TX = eltype(core.trace.latent_points)
    pending = Dict{UInt64,PendingBatch{TX}}()
    for (identifier, batch) in saved.pending
        pending[identifier] = PendingBatch(
            Matrix{TX}(batch.points),
            copy(batch.unresolved),
        )
    end
    return SMBOState(
        core,
        saved.algorithm,
        deepcopy(saved.model),
        pending,
        saved.next_identifier,
        saved.observations_at_fit,
        deepcopy(saved.failures),
        Matrix{TX}(saved.initial_design),
        saved.initial_design_cursor,
        deepcopy(saved.workspace),
    )
end
