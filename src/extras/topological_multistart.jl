"""Experimental one-refinement topological multistart search."""
struct TopologicalMultistart{S,T,L}
    sampling::S
    topology::T
    local_solver::L
    samples::Int
    local_starts::Int
end
function TopologicalMultistart(;
    sampling=SobolDesign(),
    topology=DelaunayTopology(),
    local_solver=PatternSearch(),
    samples=256,
    local_starts=8,
)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    local_starts > 0 || throw(ArgumentError("local_starts must be positive"))
    return TopologicalMultistart(sampling, topology, local_solver, samples, local_starts)
end

mutable struct TopologicalMultistartState{C,A,S}
    core::C
    algorithm::A
    inner::S
end

function init(
    problem::Problem,
    algorithm::TopologicalMultistart;
    initial_points=nothing,
    initial_values=nothing,
    kwargs...,
)
    _continuous_space(problem.space) || throw(ArgumentError(
        "TopologicalMultistart supports continuous dimensions only",
    ))
    dimensions = length(active_dimensions(problem.space))
    budget = get(kwargs, :maximum_evaluations, 100)
    known = isnothing(initial_values) ? 0 :
        initial_values isa Real ? 1 : length(initial_values)
    reserve = min(max(0, budget - 1), max(2dimensions, 2algorithm.local_starts))
    minimum_complex = min(budget, max(dimensions + 2, 2dimensions + 1))
    new_target = max(minimum_complex, budget - reserve)
    sample_count = min(algorithm.samples, known + new_target)
    internal = SHGO(
        sampling=algorithm.sampling,
        topology=algorithm.topology,
        local_solver=algorithm.local_solver,
        sampling_points=max(0, sample_count - known),
        local_starts=algorithm.local_starts,
        maximum_refinements=1,
        minimum_local_reserve=0,
        divide_automatic_local_budget=false,
    )
    inner = init(problem, internal; initial_points, initial_values, kwargs...)
    required_vertices = dimensions == 1 ? 2 : dimensions + 2
    sample_count < required_vertices &&
        (inner.core.retcode = :evaluation_limit)
    return TopologicalMultistartState(inner.core, algorithm, inner)
end

function _topological_retcode!(state)
    state.core.retcode == :success && (state.core.retcode = :stalled)
    return state
end

step!(state::TopologicalMultistartState) =
    (step!(state.inner); _topological_retcode!(state))

function solve!(state::TopologicalMultistartState)
    solve!(state.inner)
    _topological_retcode!(state)
    return result(state)
end

result(state::TopologicalMultistartState) = _result(
    state.core,
    state.algorithm;
    statistics=(
        iterations=state.core.iteration,
        local_candidates=min(length(
            state.inner.workspace.candidate_indices,
        ), state.algorithm.local_starts),
    ),
)
