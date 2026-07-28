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
) where {T} = isnothing(cardinality) ? nothing : BitSet()
_occupancy(policy, equality, cardinality, ::Type{T}) where {T} = nothing
_tracks_finite_space(::AllowRepeatedEvaluations, cardinality) = false
_tracks_finite_space(::AvoidRepeatedEvaluations, cardinality) =
    !isnothing(cardinality)

function _occupy!(state::SMBOState, points)
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
_occupy!(::Nothing, space, point) = nothing
function _release_occupied!(state::SMBOState, pending, indices)
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
_release_occupied!(::Nothing, space, point) = nothing

function _unused_finite_candidates(state::SMBOState, requested)
    isnothing(state.feasible_indices) && return nothing
    capacity = min(requested, length(state.feasible_indices))
    selected = Vector{Int}(undef, max(capacity, 0))
    seen = 0
    for index in state.feasible_indices
        _finite_index_occupied(state, index) && continue
        seen += 1
        if seen <= capacity
            selected[seen] = index
        else
            replacement = Random.rand(state.core.rng, 1:seen)
            replacement <= capacity &&
                (selected[replacement] = index)
        end
    end
    return _finite_points(
        state.core.problem,
        @view(selected[1:min(seen, capacity)]),
    )
end

_finite_index_occupied(state, index) =
    _finite_index_occupied(state, index, state.occupied)
_finite_index_occupied(state, index, occupied::BitSet) = index in occupied
function _finite_index_occupied(state, index, occupied)
    space = state.core.problem.space
    candidate = Vector{eltype(state.core.trace.latent_points)}(
        undef,
        dimension(space),
    )
    canonical_latent!(candidate, space, index)
    equality = state.algorithm.candidate_equality
    trace = state.core.trace
    for column in 1:trace.count
        _same_candidate(
            candidate,
            @view(trace.latent_points[:, column]),
            equality,
        ) &&
            return true
    end
    for pending in values(state.pending), column in axes(pending.points, 2)
        pending.unresolved[column] || continue
        _same_candidate(candidate, @view(pending.points[:, column]), equality) &&
            return true
    end
    return false
end
