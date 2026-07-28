"""
Experimental Delaunay/neighbor-graph multistart search.

This is not an implementation of simplicial homology global optimization.
"""
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

mutable struct TopologicalMultistartWorkspace{TX,TY}
    sample_points::Matrix{TX}
    sample_values::Vector{TY}
    local_indices::Vector{Int}
    current_start::Int
    center::Vector{TX}
    center_value::TY
    step_size::TX
    initialized::Bool
    local_minima_count::Int
    completed_starts::Int
    proposals::Matrix{TX}
    proposal_values::Vector{TY}
end

mutable struct TopologicalMultistartState{C,A,W}
    core::C
    algorithm::A
    workspace::W
end

function init(
    problem::Problem,
    algorithm::TopologicalMultistart;
    initial_points=nothing,
    initial_values=nothing,
    kwargs...,
)
    _continuous_space(problem.space) ||
        throw(ArgumentError("TopologicalMultistart supports continuous dimensions only"))
    core = _makecore(problem; kwargs...)
    _seed_initial!(core, initial_points, initial_values)
    TX = eltype(core.trace.latent_points)
    TY = eltype(core.trace.objective_values)
    remaining = _remaining(core)
    reserve = min(max(0, remaining - 1), max(2dimension(problem.space), 2algorithm.local_starts))
    minimum_complex = min(
        remaining,
        max(dimension(problem.space) + 2, 2dimension(problem.space) + 1),
    )
    new_target = iszero(remaining) ? 0 : max(minimum_complex, remaining - reserve)
    sample_count = min(algorithm.samples, core.trace.count + new_target)
    required_vertices = dimension(problem.space) == 1 ? 2 : dimension(problem.space) + 2
    sample_count < required_vertices && (core.retcode = :evaluation_limit)
    workspace = TopologicalMultistartWorkspace(
        Matrix{TX}(undef, dimension(problem.space), sample_count),
        Vector{TY}(undef, sample_count),
        Int[],
        0,
        zeros(TX, dimension(problem.space)),
        _worst(TY, problem.sense),
        TX(initialstep(algorithm.local_solver)),
        false,
        0,
        0,
        Matrix{TX}(undef, dimension(problem.space), 2dimension(problem.space)),
        Vector{TY}(undef, 2dimension(problem.space)),
    )
    return TopologicalMultistartState(core, algorithm, workspace)
end

function _initialize_topological_multistart!(state::TopologicalMultistartState)
    workspace = state.workspace
    trace = state.core.trace
    sample_count = size(workspace.sample_points, 2)
    initial_count = min(trace.count, sample_count)
    if initial_count > 0
        initial_indices = partialsortperm(
            _loss.(Ref(state.core.problem.sense), objectivevalues(trace)),
            1:initial_count,
        )
        workspace.sample_points[:, 1:initial_count] .=
            @view trace.latent_points[:, initial_indices]
        workspace.sample_values[1:initial_count] .= trace.objective_values[initial_indices]
    end
    new_count = sample_count - initial_count
    if new_count > 0
        new_points = @view workspace.sample_points[:, initial_count+1:sample_count]
        _sample_feasible!(
            state.core.rng,
            new_points,
            state.algorithm.sampling,
            state.core.problem,
        )
        new_values = _evaluate_batch(state.core, new_points)
        workspace.sample_values[initial_count+1:sample_count] .= new_values
        state.core.iteration += 1
        _commit_batch!(state.core, new_points, new_values)
    end
    complex = buildcomplex(workspace.sample_points, state.algorithm.topology)
    workspace.local_indices = localcandidates(
        complex,
        _loss.(Ref(state.core.problem.sense), workspace.sample_values),
    )
    isempty(workspace.local_indices) && push!(
        workspace.local_indices,
        argmin(_loss.(Ref(state.core.problem.sense), workspace.sample_values)),
    )
    sort!(
        workspace.local_indices;
        by=index -> _loss(state.core.problem, workspace.sample_values[index]),
    )
    resize!(
        workspace.local_indices,
        min(length(workspace.local_indices), state.algorithm.local_starts),
    )
    workspace.local_minima_count = length(workspace.local_indices)
    workspace.initialized = true
    _begin_local_start!(state, 1)
    return state
end

function _begin_local_start!(state::TopologicalMultistartState, start)
    workspace = state.workspace
    1 <= start <= length(workspace.local_indices) ||
        throw(BoundsError(workspace.local_indices, start))
    workspace.current_start = start
    index = workspace.local_indices[start]
    workspace.center .= @view workspace.sample_points[:, index]
    workspace.center_value = workspace.sample_values[index]
    workspace.step_size = eltype(workspace.center)(initialstep(state.algorithm.local_solver))
    return state
end

struct LocalStartExhausted end

function _finish_local_start!(state::TopologicalMultistartState)
    workspace = state.workspace
    workspace.completed_starts += 1
    if workspace.completed_starts >= length(workspace.local_indices)
        state.core.retcode = :stalled
    else
        _begin_local_start!(state, workspace.current_start + 1)
    end
    return LocalStartExhausted()
end

function _local_proposals(state::TopologicalMultistartState, ::PatternSearch)
    workspace = state.workspace
    core = state.core
    d = dimension(core.problem.space)
    maximum = min(2d, _remaining(core))
    proposals = @view workspace.proposals[:, 1:maximum]
    count = 0
    for axis in 1:d, direction in (-1, 1)
        count == maximum && break
        proposal = @view proposals[:, count + 1]
        proposal .= workspace.center
        proposal[axis] += direction * workspace.step_size
        project!(proposal, core.problem.space)
        _canonicalize!(proposal, core.problem.space)
        isfeasible(core.problem, decode(core.problem.space, proposal)) || continue
        count += 1
    end
    return @view proposals[:, 1:count]
end

function step!(state::TopologicalMultistartState)
    _finished(state.core) && return state
    !state.workspace.initialized && return _initialize_topological_multistart!(state)
    proposals = _local_proposals(state, state.algorithm.local_solver)
    if isempty(proposals)
        _finish_local_start!(state)
        return state
    end
    values = @view state.workspace.proposal_values[1:size(proposals, 2)]
    _evaluate_batch!(values, state.core, proposals)
    state.core.iteration += 1
    _commit_batch!(state.core, proposals, values)
    best = argmin(_loss.(Ref(state.core.problem.sense), values))
    if _isbetter(state.core.problem, values[best], state.workspace.center_value)
        state.workspace.center .= @view proposals[:, best]
        state.workspace.center_value = values[best]
    else
        state.workspace.step_size *= eltype(state.workspace.center)(0.5)
        state.workspace.step_size < minimumstep(state.algorithm.local_solver) &&
            _finish_local_start!(state)
    end
    return state
end

function solve!(state::TopologicalMultistartState)
    isnothing(state.core.problem.objective) &&
        throw(ArgumentError("solve! requires an objective"))
    try
        while !_finished(state.core)
            step!(state)
        end
    catch error
        error isa InfeasibleSpaceError || rethrow()
        state.core.retcode = :infeasible_space
    end
    return result(state)
end
solve(problem::Problem, algorithm::TopologicalMultistart; kwargs...) =
    solve!(init(problem, algorithm; kwargs...))
trace(state::TopologicalMultistartState) = state.core.trace
result(state::TopologicalMultistartState) = _result(
    state.core,
    state.algorithm;
    statistics=(
        iterations=state.core.iteration,
        local_candidates=length(state.workspace.local_indices),
    ),
)
