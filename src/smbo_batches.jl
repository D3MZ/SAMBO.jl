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
    return _commit_pending!(
        state,
        batch.identifier,
        pending,
        selected,
        candidates,
        values,
        source,
    )
end

function _commit_pending!(
    state,
    identifier,
    pending,
    selected,
    candidates,
    values,
    source,
)
    converted = _validated_values(state.core, candidates, values)
    length(selected) <= _remaining(state.core) ||
        throw(ArgumentError("completed candidates exceed the evaluation budget"))
    _resolve_pending!(state, identifier, pending, selected)
    state.core.iteration += 1
    _commit_batch!(state.core, candidates, converted; source)
    return state
end
function tell!(state::SMBOState, batch::CandidateBatch, values)
    pending = _pending_batch(state, batch)
    if all(pending.unresolved)
        return _commit_pending!(
            state,
            batch.identifier,
            pending,
            axes(pending.points, 2),
            pending.points,
            values,
            ExternalEvaluation,
        )
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
