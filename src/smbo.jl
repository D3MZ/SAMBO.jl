struct UniformCandidates end
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
GlobalLocalCandidates(; global_fraction=0.35, local_scale=0.15) =
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

struct SMBO{S,A,I,C,F,R,B,E}
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
    refit_schedule=nothing,
    prediction_chunk_size=4096,
    adaptive_refit=true,
    repeat_policy=AvoidRepeatedEvaluations(),
    batch_strategy=LocalPenalization(),
    candidate_equality=ExactCandidateEquality(),
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

function _unique_design_columns(design, equality)
    selected = similar(design)
    count = 0
    for column in axes(design, 2)
        candidate = @view design[:, column]
        duplicate = any(1:count) do existing
            _same_candidate(
                candidate,
                @view(selected[:, existing]),
                equality,
            )
        end
        duplicate && continue
        count += 1
        selected[:, count] .= candidate
    end
    return Matrix(@view selected[:, 1:count])
end
_prepare_initial_design(design, ::AllowRepeatedEvaluations) = design
_prepare_initial_design(
    design,
    ::AvoidRepeatedEvaluations,
    equality,
) = _unique_design_columns(design, equality)
_prepare_initial_design(design, ::AllowRepeatedEvaluations, equality) = design

function _initialize_occupancy!(
    occupied::BitSet,
    core,
    ::AvoidRepeatedEvaluations,
    ::ExactCandidateEquality,
    cardinality,
)
    isnothing(cardinality) && return occupied
    for column in 1:core.trace.count
        push!(
            occupied,
            canonical_index(
                core.problem.space,
                @view(core.trace.latent_points[:, column]),
            ),
        )
    end
    return occupied
end
_initialize_occupancy!(
    occupied,
    core,
    ::Union{AllowRepeatedEvaluations,AvoidRepeatedEvaluations},
    equality,
    cardinality,
) = occupied

_occupancy(
    ::AvoidRepeatedEvaluations,
    ::ExactCandidateEquality,
    cardinality,
    ::Type{T},
) where {T} = isnothing(cardinality) ? Set{Vector{T}}() : BitSet()
_occupancy(policy, equality, cardinality, ::Type{T}) where {T} =
    Set{Vector{T}}()

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

function generate_candidates!(
    destination,
    state::SMBOState,
    ::UniformCandidates,
)
    _sample_feasible!(
        state.core.rng,
        destination,
        UniformDesign(),
        state.core.problem,
    )
    return size(destination, 2)
end

function generate_candidates!(
    candidates,
    state::SMBOState,
    sampler::EliteGaussianCandidates,
)
    core = state.core
    T = eltype(core.trace.latent_points)
    d = dimension(core.problem.space)
    count = size(candidates, 2)
    trace = core.trace
    trace.count == 0 && return 0
    elite_count = clamp(round(Int, sqrt(trace.count)), 1, trace.count)
    order = state.workspace.order
    resize!(order, trace.count)
    partialsortperm!(
        order,
        objectivevalues(trace),
        1:elite_count;
        by=value -> _loss(core.problem.sense, value),
    )
    write = 0
    attempts = 0
    while write < count && attempts < 200count
        attempts += 1
        rank = Random.rand(core.rng, 1:elite_count)
        center = @view trace.latent_points[:, order[rank]]
        proposal = @view candidates[:, write + 1]
        scale = T(sampler.scale) / sqrt(T(1 + core.iteration) / T(2))
        for row in 1:d
            proposal[row] = center[row] + scale * Random.randn(core.rng, T)
        end
        project!(proposal, core.problem.space)
        _canonicalize!(proposal, core.problem.space)
        isfeasible(core.problem, decode(core.problem.space, proposal)) || continue
        write += 1
    end
    return write
end

function generate_candidates!(
    candidates,
    state::SMBOState,
    sampler::MixtureCandidates,
)
    count = size(candidates, 2)
    global_count = round(Int, count * sampler.global_fraction)
    local_count = count - global_count
    generated_global = global_count == 0 ? 0 : generate_candidates!(
        @view(candidates[:, 1:global_count]),
        state,
        sampler.global_sampler,
    )
    generated_global == global_count || return generated_global
    generated_local = local_count == 0 ? 0 : generate_candidates!(
        @view(candidates[:, global_count+1:count]),
        state,
        sampler.local_sampler,
    )
    return global_count + generated_local
end

function _ensure_candidate_capacity!(workspace, rows, columns)
    if size(workspace.candidates, 1) != rows ||
            size(workspace.candidates, 2) < columns
        workspace.candidates = Matrix{eltype(workspace.candidates)}(
            undef,
            rows,
            columns,
        )
    end
    return workspace
end

@inline function _same_candidate(left, right, ::ExactCandidateEquality)
    length(left) == length(right) || return false
    @inbounds for i in eachindex(left, right)
        isequal(left[i], right[i]) || return false
    end
    return true
end
@inline function _same_candidate(
    left,
    right,
    equality::ApproximateCandidateEquality,
)
    length(left) == length(right) || return false
    @inbounds for index in eachindex(left, right)
        abs(left[index] - right[index]) <= equality.tolerance ||
            return false
    end
    return true
end

_is_duplicate(
    occupied,
    ::AllowRepeatedEvaluations,
    state,
    candidate,
    selected,
    selected_count,
) = false
function _is_duplicate(
    occupied::BitSet,
    ::AvoidRepeatedEvaluations,
    state,
    candidate,
    selected,
    selected_count,
)
    canonical_index(state.core.problem.space, candidate) in occupied &&
        return true
    for column in 1:selected_count
        canonical_index(
            state.core.problem.space,
            @view(selected[:, column]),
        ) == canonical_index(state.core.problem.space, candidate) &&
            return true
    end
    return false
end
function _is_duplicate(
    occupied,
    ::AvoidRepeatedEvaluations,
    state,
    candidate,
    selected,
    selected_count,
)
    trace = state.core.trace
    for column in 1:trace.count
        _same_candidate(
            candidate,
            @view(trace.latent_points[:, column]),
            state.algorithm.candidate_equality,
        ) && return true
    end
    for pending in values(state.pending), column in axes(pending.points, 2)
        pending.unresolved[column] || continue
        _same_candidate(
            candidate,
            @view(pending.points[:, column]),
            state.algorithm.candidate_equality,
        ) && return true
    end
    for column in 1:selected_count
        _same_candidate(
            candidate,
            @view(selected[:, column]),
            state.algorithm.candidate_equality,
        ) && return true
    end
    return false
end
_is_duplicate(state::SMBOState, candidate, selected, selected_count) =
    _is_duplicate(
        state.occupied,
        state.algorithm.repeat_policy,
        state,
        candidate,
        selected,
        selected_count,
    )

function _predict_candidates!(
    state,
    means,
    variances,
    candidates,
    model,
    ::LowerConfidenceBound,
)
    chunk = state.algorithm.prediction_chunk_size
    for first in 1:chunk:size(candidates, 2)
        last = min(first + chunk - 1, size(candidates, 2))
        predictmeanvariance!(
            @view(means[first:last]),
            @view(variances[first:last]),
            model,
            @view(candidates[:, first:last]),
            state.workspace.prediction,
        )
    end
    return means, variances
end
function _predict_candidates!(
    state,
    means,
    variances,
    candidates,
    model,
    ::GreedyMean,
)
    chunk = state.algorithm.prediction_chunk_size
    for first in 1:chunk:size(candidates, 2)
        last = min(first + chunk - 1, size(candidates, 2))
        predictmean!(
            @view(means[first:last]),
            model,
            @view(candidates[:, first:last]),
            state.workspace.prediction,
        )
    end
    fill!(variances, zero(eltype(variances)))
    return means, variances
end
_predict_candidates!(state, means, variances, candidates, acquisition) =
    _predict_candidates!(
        state,
        means,
        variances,
        candidates,
        state.model,
        acquisition,
    )

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
        _distance_squared(candidate, @view(selected[:, column]))
    end
    penalty = strategy.strength *
        exp(-distance_squared / (2strategy.radius^2))
    return score + penalty
end

function _select_from_scores!(
    selected,
    state,
    candidates,
    scores,
    requested,
    ::GreedyBatch,
)
    order = state.workspace.order
    resize!(order, length(scores))
    sortperm!(order, scores)
    count = 0
    for index in order
        candidate = @view candidates[:, index]
        _is_duplicate(state, candidate, selected, count) && continue
        count += 1
        selected[:, count] .= candidate
        count == requested && break
    end
    return count
end

function _select_from_scores!(
    selected,
    state,
    candidates,
    scores,
    requested,
    strategy::LocalPenalization,
)
    available = state.workspace.available
    resize!(available, length(scores))
    fill!(available, true)
    available_count = length(available)
    count = 0
    while count < requested && available_count > 0
        choice = 0
        choice_score = eltype(scores)(Inf)
        for index in eachindex(scores)
            available[index] || continue
            candidate = @view candidates[:, index]
            adjusted = _penalized_score(
                scores[index],
                candidate,
                selected,
                count,
                strategy,
            )
            if adjusted < choice_score
                choice = index
                choice_score = adjusted
            end
        end
        iszero(choice) && break
        available[choice] = false
        available_count -= 1
        candidate = @view candidates[:, choice]
        _is_duplicate(state, candidate, selected, count) && continue
        count += 1
        selected[:, count] .= candidate
    end
    return count
end

function _occupied_count_scan(state::SMBOState)
    occupied = empty(state.occupied)
    trace = state.core.trace
    for column in 1:trace.count
        push!(occupied, collect(@view trace.latent_points[:, column]))
    end
    for pending in values(state.pending), column in axes(pending.points, 2)
        pending.unresolved[column] &&
            push!(occupied, collect(@view pending.points[:, column]))
    end
    return length(occupied)
end
_tracks_occupancy(
    ::AllowRepeatedEvaluations,
    equality,
    cardinality,
) = false
_tracks_occupancy(
    ::AvoidRepeatedEvaluations,
    ::ExactCandidateEquality,
    cardinality,
) = !isnothing(cardinality)
_tracks_occupancy(
    ::AvoidRepeatedEvaluations,
    equality,
    cardinality,
) = false
_tracks_occupancy(state::SMBOState) = _tracks_occupancy(
    state.algorithm.repeat_policy,
    state.algorithm.candidate_equality,
    space_cardinality(state.core.problem.space),
)
_occupied_count(state::SMBOState) =
    _tracks_occupancy(state) ? length(state.occupied) : _occupied_count_scan(state)
function _occupy!(state::SMBOState, points)
    _tracks_occupancy(state) || return state
    for column in axes(points, 2)
        _occupy!(
            state.occupied,
            state.core.problem.space,
            @view(points[:, column]),
        )
    end
    return state
end
_occupy!(occupied::BitSet, space, point) =
    push!(occupied, canonical_index(space, point))
_occupy!(occupied, space, point) = push!(occupied, collect(point))
function _release_occupied!(state::SMBOState, pending, indices)
    _tracks_occupancy(state) || return state
    for index in indices
        _release_occupied!(
            state.occupied,
            state.core.problem.space,
            @view(pending.points[:, index]),
        )
    end
    return state
end
_release_occupied!(occupied::BitSet, space, point) =
    delete!(occupied, canonical_index(space, point))
_release_occupied!(occupied, space, point) =
    delete!(occupied, collect(point))

_space_exhausted(::AllowRepeatedEvaluations, state, cardinality) = false
function _space_exhausted(
    ::AvoidRepeatedEvaluations,
    state,
    cardinality,
)
    capacity = isnothing(state.feasible_indices) ?
        cardinality : length(state.feasible_indices)
    return !isnothing(capacity) && _occupied_count(state) >= capacity
end

function _unused_finite_candidates(state::SMBOState, requested)
    isnothing(state.feasible_indices) && return nothing
    capacity = min(
        requested,
        length(state.feasible_indices) - length(state.occupied),
    )
    selected = Vector{Int}(undef, max(capacity, 0))
    seen = 0
    for index in state.feasible_indices
        index in state.occupied && continue
        seen += 1
        if seen <= capacity
            selected[seen] = index
        else
            replacement = Random.rand(state.core.rng, 1:seen)
            replacement <= capacity &&
                (selected[replacement] = index)
        end
    end
    return _finite_points(state.core.problem, selected)
end

_deduplicate_candidates(
    state,
    candidates,
    requested,
    ::AllowRepeatedEvaluations,
) = candidates
function _deduplicate_candidates(
    state,
    candidates,
    requested,
    ::AvoidRepeatedEvaluations,
)
    selected = _ensure_smbo_workspace!(state, candidates, requested).selected
    count = 0
    for column in axes(candidates, 2)
        candidate = @view candidates[:, column]
        _is_duplicate(state, candidate, selected, count) && continue
        count += 1
        selected[:, count] .= candidate
        count == requested && break
    end
    return Matrix(@view selected[:, 1:count])
end
_deduplicate_candidates(state, candidates, requested) =
    _deduplicate_candidates(
        state,
        candidates,
        requested,
        state.algorithm.repeat_policy,
    )

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
    _predict_candidates!(
        state,
        means,
        variances,
        candidates,
        state.algorithm.acquisition,
    )
    acquisitionvalues!(
        scores,
        state.algorithm.acquisition,
        means,
        variances,
        _loss(state.core.problem, state.core.best_value),
    )
    count = _select_from_scores!(
        selected,
        state,
        candidates,
        scores,
        requested,
        state.algorithm.batch_strategy,
    )
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
                    state.algorithm.initial_design,
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
function tell!(state::SMBOState, batch::CandidateBatch, values)
    pending = _pending_batch(state, batch)
    if all(pending.unresolved)
        converted = _validated_values(state.core, pending.points, values)
        length(converted) <= _remaining(state.core) ||
            throw(ArgumentError(
                "completed candidates exceed the evaluation budget",
            ))
        delete!(state.pending, batch.identifier)
        state.core.iteration += 1
        _commit_batch!(
            state.core,
            pending.points,
            converted;
            source=ExternalEvaluation,
        )
        return state
    end
    return _tell!(
        state,
        batch,
        findall(pending.unresolved),
        values,
        ExternalEvaluation,
    )
end
tell!(state::SMBOState, batch::CandidateBatch, indices, values) =
    _tell!(state, batch, indices, values, ExternalEvaluation)

function cancel!(state::SMBOState, batch::CandidateBatch, indices)
    pending = _pending_batch(state, batch)
    selected = _validated_pending_indices(pending, indices)
    _release_occupied!(state, pending, selected)
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
    _release_occupied!(state, pending, selected)
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
                    _same_candidate(
                        candidate,
                        @view(pending.points[:, index]),
                        state.algorithm.candidate_equality,
                    )
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
    _occupy!(state, latent)
    return state
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

struct SMBOCheckpoint{A,M,P,R,E,C,TR,SC,N,T,F,I,W,S,OS,TX,TY}
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
    space_signature::S
    sense_type::OS
    coordinate_type::TX
    objective_type::TY
end

_descriptor_signature(dimension::Continuous) =
    (:continuous, dimension.lower, dimension.upper)
_descriptor_signature(dimension::AbstractRange) =
    (:range, first(dimension), step(dimension), length(dimension))
_descriptor_signature(dimension::Choices) =
    (:choices, dimension.values)
_space_signature(space::Box) = (
    :box,
    Tuple(space.lower),
    Tuple(space.upper),
    latenttype(space),
)
_space_signature(space::SearchSpace) = (
    :search_space,
    map(_descriptor_signature, Tuple(space.dimensions)),
    dimensionnames(space),
    latenttype(space),
)

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
        _space_signature(core.problem.space),
        typeof(core.problem.sense),
        eltype(core.trace.latent_points),
        eltype(core.trace.objective_values),
    )
end

function restore(problem::Problem, saved::SMBOCheckpoint)
    _space_signature(problem.space) == saved.space_signature ||
        throw(ArgumentError("checkpoint search space does not match the problem"))
    typeof(problem.sense) === saved.sense_type ||
        throw(ArgumentError("checkpoint optimization sense does not match the problem"))
    latenttype(problem.space) === saved.coordinate_type ||
        throw(ArgumentError("checkpoint coordinate type does not match the problem"))
    eltype(saved.trace.objective_values) === saved.objective_type ||
        throw(ArgumentError("checkpoint objective type metadata is inconsistent"))
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
    workspace = deepcopy(saved.workspace)
    cardinality = space_cardinality(problem.space)
    occupied = _occupancy(
        saved.algorithm.repeat_policy,
        saved.algorithm.candidate_equality,
        cardinality,
        TX,
    )
    feasible_indices = _tracks_occupancy(
        saved.algorithm.repeat_policy,
        saved.algorithm.candidate_equality,
        cardinality,
    ) ? _finite_feasible_indices(problem) : nothing
    for column in 1:core.trace.count
        _occupy!(
            occupied,
            problem.space,
            @view(core.trace.latent_points[:, column]),
        )
    end
    for batch in values(pending), column in axes(batch.points, 2)
        batch.unresolved[column] &&
            _occupy!(
                occupied,
                problem.space,
                @view(batch.points[:, column]),
            )
    end
    Model = Union{
        Nothing,
        fittedmodeltype(
            saved.algorithm.surrogate,
            TX,
            eltype(core.trace.objective_values),
        ),
    }
    return SMBOState{
        typeof(core),
        typeof(saved.algorithm),
        TX,
        typeof(workspace),
        Model,
        typeof(occupied),
        typeof(feasible_indices),
    }(
        core,
        saved.algorithm,
        deepcopy(saved.model),
        pending,
        saved.next_identifier,
        saved.observations_at_fit,
        deepcopy(saved.failures),
        Matrix{TX}(saved.initial_design),
        saved.initial_design_cursor,
        workspace,
        occupied,
        feasible_indices,
    )
end
