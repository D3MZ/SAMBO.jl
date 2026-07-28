"""Internal policy used to reproduce SciPy's incremental Delaunay construction."""
struct PythonIncrementalDelaunayTopology end
struct FarthestFromLatestMinimum end
struct PythonSAMBOProfile end

buildcomplex(points, ::PythonIncrementalDelaunayTopology) =
    _delaunay_complex(
        points,
        "qhull d Qt Qc Qx Q11";
        scipy_shgo_adjacency=true,
    )

"""
Configuration matching the effective SHGO policy used by Python SAMBO 1.25.2.
"""
function SHGO(::PythonSAMBOProfile; kwargs...)
    defaults = (
        sampling=ScrambledHaltonDesign(skip=0),
        topology=PythonIncrementalDelaunayTopology(),
        local_solver=QuasiNewtonSearch(
            finite_difference_step=sqrt(eps(Float64)),
            gradient_tolerance=1e-6,
            minimum_step=1e-8,
        ),
        sampling_points=80,
        local_starts=4,
        local_start_policy=FarthestFromLatestMinimum(),
        minimum_homology_growth=0,
        homology_patience=1,
        minimum_local_reserve=0,
        divide_automatic_local_budget=false,
        convergence_tolerance=1e-6,
        convergence_window=30,
    )
    return SHGO(; merge(defaults, (; kwargs...))...)
end

function _filter_topological_candidates!(
    candidates,
    state::SHGOState{C,<:SHGO{S,<:PythonIncrementalDelaunayTopology}},
) where {C,S}
    filter!(
        index -> !isempty(state.workspace.complex.adjacency[index]),
        candidates,
    )
    return candidates
end

_order_local_candidates!(
    candidates,
    state::SHGOState{C,<:SHGO{S,<:PythonIncrementalDelaunayTopology}},
) where {C,S} = candidates

function local_minimize!(
    state::SHGOState{C,<:SHGO{S,<:PythonIncrementalDelaunayTopology}},
    solver::QuasiNewtonSearch,
    start,
    start_value,
    lower,
    upper,
    budget,
) where {C,S}
    budget <= 0 && return copy(start), start_value
    core = state.core
    candidate = reshape(copy(start), :, 1)
    values = Vector{eltype(core.trace.objective_values)}(undef, 1)
    _evaluate_batch!(values, core, candidate)
    core.iteration += 1
    _commit_batch!(core, candidate, values)
    _shgo_mark_value_spread_converged!(state)
    return _quasi_newton_minimize!(
        state,
        solver,
        start,
        values[1],
        lower,
        upper,
        budget - 1,
    )
end

function _next_local_candidate!(
    candidates,
    state,
    ::FarthestFromLatestMinimum,
    latest,
)
    isnothing(latest) && return popfirst!(candidates)
    workspace = state.workspace
    latest_decoded = decode(state.core.problem.space, latest)
    farthest = firstindex(candidates)
    farthest_distance = -Inf
    for position in eachindex(candidates)
        candidate = candidates[position]
        point = @view workspace.sample_points[:, candidate]
        decoded = decode(state.core.problem.space, point)
        distance = sum(
            abs2(left - right)
            for (left, right) in zip(decoded, latest_decoded)
        )
        if distance > farthest_distance
            farthest = position
            farthest_distance = distance
        end
    end
    return popat!(candidates, farthest)
end
