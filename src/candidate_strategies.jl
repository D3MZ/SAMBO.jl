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
    sampler::AdaptiveDensityCandidates,
)
    d = dimension(state.core.problem.space)
    if state.core.trace.count < 10d^2
        return generate_candidates!(candidates, state, UniformCandidates())
    end
    trace = state.core.trace
    count = trace.count
    points = @view trace.latent_points[:, 1:count]
    losses = _loss.(Ref(state.core.problem.sense), objectivevalues(trace))
    weights = maximum(losses) .- losses
    if !(sum(weights) > 0)
        return generate_candidates!(candidates, state, UniformCandidates())
    end
    weights ./= sum(weights)
    weights .^= 3
    weights ./= sum(weights)
    mean_point = points * weights
    covariance = zeros(eltype(points), d, d)
    for column in 1:count
        delta = @view(points[:, column]) .- mean_point
        covariance .+= weights[column] .* (delta * delta')
    end
    squared_weight_sum = sum(abs2, weights)
    bandwidth = _silverman_bandwidth(weights, d)
    covariance ./= 1 - squared_weight_sum
    factor = cholesky(
        Symmetric(covariance + eps(eltype(covariance)) * I);
        check=false,
    )
    isposdef(factor) ||
        return generate_candidates!(candidates, state, UniformCandidates())
    cumulative = cumsum(weights)
    written = 0
    attempts = 0
    while written < size(candidates, 2) && attempts < 1_000size(candidates, 2)
        attempts += 1
        center = searchsortedfirst(cumulative, Random.rand(state.core.rng))
        proposal = @view candidates[:, written + 1]
        proposal .= @view(points[:, center]) .+
            bandwidth .* (factor.L * Random.randn(state.core.rng, d))
        all(0 .<= proposal .<= 1) || continue
        isfeasible(state.core.problem, decode(state.core.problem.space, proposal)) ||
            continue
        written += 1
    end
    return written
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
    acquisition = state.algorithm.acquisition
    if acquisition isa RandomizedLowerConfidenceBound
        coefficient = acquisition.lower +
            Random.rand(state.core.rng) *
            (acquisition.upper - acquisition.lower)
        acquisition = LowerConfidenceBound(coefficient)
    end
    _predict_candidates!(
        state,
        means,
        variances,
        candidates,
        acquisition,
    )
    acquisitionvalues!(
        scores,
        acquisition,
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
