struct SimplicialSampling end
struct SimplexTopology
    neighbors::Int
end
SimplexTopology(; neighbors=0) = SimplexTopology(neighbors)
struct PatternSearch{T<:Real}
    initial_step::T
    minimum_step::T
end
PatternSearch(; initial_step=0.15, minimum_step=1e-5) =
    PatternSearch(promote(initial_step, minimum_step)...)

struct SHGO{S,T,L}
    sampling::S
    topology::T
    local_solver::L
    samples::Int
    local_starts::Int
end
function SHGO(;
    sampling=SimplicialSampling(),
    topology=SimplexTopology(),
    local_solver=PatternSearch(),
    samples=256,
    local_starts=8,
)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    local_starts > 0 || throw(ArgumentError("local_starts must be positive"))
    return SHGO(sampling, topology, local_solver, samples, local_starts)
end

struct NeighborComplex
    adjacency::Vector{Vector{Int}}
end
neighbors(complex::NeighborComplex, vertex) = complex.adjacency[vertex]

function buildcomplex(points, topology::SimplexTopology)
    _, count = size(points)
    count > 0 || return NeighborComplex(Vector{Vector{Int}}())
    dimension_count = size(points, 1)
    if dimension_count == 1
        order = sortperm(@view points[1, :])
        adjacency = [Int[] for _ in 1:count]
        for position in eachindex(order)
            position > 1 && push!(adjacency[order[position]], order[position - 1])
            position < count && push!(adjacency[order[position]], order[position + 1])
        end
        return NeighborComplex(adjacency)
    elseif count >= dimension_count + 1
        try
            simplices = MiniQhull.delaunay(points, "qhull d Qt Qbb Qc QJ Pp")
            adjacency = [Int[] for _ in 1:count]
            for simplex in eachcol(simplices), left in eachindex(simplex), right in left+1:length(simplex)
                a = Int(simplex[left])
                b = Int(simplex[right])
                (1 <= a <= count && 1 <= b <= count) || continue
                push!(adjacency[a], b)
                push!(adjacency[b], a)
            end
            foreach(unique!, adjacency)
            all(list -> !isempty(list), adjacency) && return NeighborComplex(adjacency)
        catch error
            error isa InterruptException && rethrow()
        end
    end
    # Degenerate clouds fall back to a geometric neighborhood graph.
    neighbor_count = topology.neighbors == 0 ?
        min(count - 1, max(2, 2dimension_count + 1)) :
        min(count - 1, topology.neighbors)
    adjacency = [Int[] for _ in 1:count]
    distances = Vector{eltype(points)}(undef, count)
    for vertex in 1:count
        for candidate in 1:count
            if candidate == vertex
                distances[candidate] = eltype(points)(Inf)
                continue
            end
            distance = zero(eltype(points))
            @inbounds for row in axes(points, 1)
                delta = points[row, vertex] - points[row, candidate]
                distance += delta * delta
            end
            distances[candidate] = distance
        end
        adjacency[vertex] = partialsortperm(distances, 1:neighbor_count)
    end
    return NeighborComplex(adjacency)
end

function localcandidates(complex::NeighborComplex, values)
    candidates = Int[]
    for vertex in eachindex(values)
        all(neighbor -> values[vertex] <= values[neighbor], neighbors(complex, vertex)) &&
            push!(candidates, vertex)
    end
    return candidates
end

mutable struct SHGOWorkspace{T}
    sample_points::Matrix{T}
    sample_values::Vector{T}
    local_indices::Vector{Int}
    current_start::Int
    center::Vector{T}
    center_value::T
    step_size::T
    initialized::Bool
    homology_rank::Int
end

mutable struct SHGOState{C,A,W}
    core::C
    algorithm::A
    workspace::W
end

function _continuous_space(space::Box)
    return true
end
function _continuous_space(space::SearchSpace)
    return all(dimension -> dimension isa Continuous, space.dimensions)
end

function init(problem::Problem, algorithm::SHGO; initial_points=nothing, initial_values=nothing, kwargs...)
    _continuous_space(problem.space) ||
        throw(ArgumentError("SHGO supports continuous dimensions only"))
    core = _makecore(problem; kwargs...)
    _seed_initial!(core, initial_points, initial_values)
    T = eltype(core.trace.objective_values)
    sample_count = min(algorithm.samples, _remaining(core))
    workspace = SHGOWorkspace(
        Matrix{T}(undef, dimension(problem.space), sample_count),
        Vector{T}(undef, sample_count),
        Int[],
        0,
        zeros(T, dimension(problem.space)),
        T(Inf),
        T(algorithm.local_solver.initial_step),
        false,
        0,
    )
    return SHGOState(core, algorithm, workspace)
end

function _shgo_design(::SimplicialSampling)
    return SobolDesign()
end
_shgo_design(design::AbstractDesign) = design

function _initialize_shgo!(state::SHGOState)
    workspace = state.workspace
    _sample_feasible!(
        state.core.rng,
        workspace.sample_points,
        _shgo_design(state.algorithm.sampling),
        state.core.problem,
    )
    workspace.sample_values .= _evaluate_batch(state.core, workspace.sample_points)
    state.core.iteration += 1
    _commit_batch!(state.core, workspace.sample_points, workspace.sample_values)
    complex = buildcomplex(workspace.sample_points, state.algorithm.topology)
    workspace.local_indices = localcandidates(complex, workspace.sample_values)
    isempty(workspace.local_indices) &&
        push!(workspace.local_indices, argmin(workspace.sample_values))
    sort!(workspace.local_indices; by=index -> workspace.sample_values[index])
    resize!(workspace.local_indices, min(length(workspace.local_indices), state.algorithm.local_starts))
    workspace.homology_rank = length(workspace.local_indices)
    workspace.initialized = true
    _begin_local_start!(state, 1)
    return state
end

function _begin_local_start!(state::SHGOState, start)
    workspace = state.workspace
    start = mod1(start, length(workspace.local_indices))
    workspace.current_start = start
    index = workspace.local_indices[start]
    workspace.center .= @view workspace.sample_points[:, index]
    workspace.center_value = workspace.sample_values[index]
    workspace.step_size = eltype(workspace.center)(state.algorithm.local_solver.initial_step)
    return state
end

function _local_proposals(state::SHGOState)
    workspace = state.workspace
    core = state.core
    d = dimension(core.problem.space)
    maximum = min(2d, _remaining(core))
    proposals = Matrix{eltype(workspace.center)}(undef, d, maximum)
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
    if count == 0 && maximum > 0
        proposals[:, 1] .= @view _sample_feasible(core, 1, UniformDesign())[:, 1]
        count = 1
    end
    return @view proposals[:, 1:count]
end

function step!(state::SHGOState)
    _finished(state.core) && return state
    !state.workspace.initialized && return _initialize_shgo!(state)
    proposals = _local_proposals(state)
    isempty(proposals) && (state.core.retcode = :infeasible_space; return state)
    values = _evaluate_batch(state.core, proposals)
    state.core.iteration += 1
    _commit_batch!(state.core, proposals, values)
    best = argmin(values)
    if values[best] < state.workspace.center_value
        state.workspace.center .= @view proposals[:, best]
        state.workspace.center_value = values[best]
    else
        state.workspace.step_size *= eltype(state.workspace.center)(0.5)
        if state.workspace.step_size < state.algorithm.local_solver.minimum_step
            _begin_local_start!(state, state.workspace.current_start + 1)
        end
    end
    return state
end

homologyrank(state::SHGOState) = state.workspace.homology_rank

function solve!(state::SHGOState)
    isnothing(state.core.problem.objective) &&
        throw(ArgumentError("solve! requires an objective"))
    while !_finished(state.core)
        step!(state)
    end
    return result(state)
end
solve(problem::Problem, algorithm::SHGO; kwargs...) = solve!(init(problem, algorithm; kwargs...))
result(state::SHGOState) = _result(
    state.core,
    state.algorithm;
    statistics=(
        iterations=state.core.iteration,
        homology_rank=state.workspace.homology_rank,
        local_candidates=length(state.workspace.local_indices),
    ),
)
