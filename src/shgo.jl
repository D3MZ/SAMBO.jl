struct DelaunayTopology end
struct KNearestTopology
    neighbors::Int
end
function KNearestTopology(; neighbors)
    neighbors > 0 || throw(ArgumentError("topology neighbors must be positive"))
    return KNearestTopology(neighbors)
end
const SimplexTopology = DelaunayTopology
struct ComplexConstructionError <: Exception
    message::String
end
Base.showerror(io::IO, error::ComplexConstructionError) =
    print(io, error.message)
struct PatternSearch{T<:Real}
    initial_step::T
    minimum_step::T
end
function PatternSearch(; initial_step=0.15, minimum_step=1e-5)
    isfinite(initial_step) && initial_step > 0 ||
        throw(ArgumentError("initial_step must be finite and positive"))
    isfinite(minimum_step) && minimum_step > 0 ||
        throw(ArgumentError("minimum_step must be finite and positive"))
    minimum_step <= initial_step ||
        throw(ArgumentError("minimum_step must not exceed initial_step"))
    return PatternSearch(promote(initial_step, minimum_step)...)
end
initialstep(solver::PatternSearch) = solver.initial_step
minimumstep(solver::PatternSearch) = solver.minimum_step

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

struct NeighborComplex
    adjacency::Vector{Vector{Int}}
end
neighbors(complex::NeighborComplex, vertex) = complex.adjacency[vertex]

function buildcomplex(points, ::DelaunayTopology)
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
    end
    count >= dimension_count + 2 || throw(ComplexConstructionError(
        "Delaunay construction requires at least dimension + 2 vertices",
    ))
    offsets = Float64.(
        points[:, 2:end] .- @view(points[:, 1]),
    )
    rank(offsets) == dimension_count || throw(ComplexConstructionError(
        "Delaunay construction requires a full-dimensional point cloud",
    ))
    simplices = try
        MiniQhull.delaunay(points, "qhull d Qt Qbb Qc Qz QJ Pp")
    catch error
        error isa ErrorException || rethrow()
        throw(ComplexConstructionError(
            "Delaunay construction failed: $(sprint(showerror, error))",
        ))
    end
    adjacency = [Int[] for _ in 1:count]
    for simplex in eachcol(simplices), left in eachindex(simplex), right in left+1:length(simplex)
        a = Int(simplex[left])
        b = Int(simplex[right])
        (1 <= a <= count && 1 <= b <= count) || continue
        push!(adjacency[a], b)
        push!(adjacency[b], a)
    end
    foreach(unique!, adjacency)
    all(list -> !isempty(list), adjacency) || throw(ComplexConstructionError(
        "Delaunay construction produced isolated vertices",
    ))
    return NeighborComplex(adjacency)
end

function buildcomplex(points, topology::KNearestTopology)
    _, count = size(points)
    count > 0 || return NeighborComplex(Vector{Vector{Int}}())
    count > 1 || return NeighborComplex([Int[]])
    neighbor_count = min(count - 1, topology.neighbors)
    adjacency = [Int[] for _ in 1:count]
    distances = Vector{eltype(points)}(undef, count)
    for vertex in 1:count
        for candidate in 1:count
            if candidate == vertex
                distances[candidate] = eltype(points)(Inf)
            else
                distance = zero(eltype(points))
                @inbounds for row in axes(points, 1)
                    delta = points[row, vertex] - points[row, candidate]
                    distance += delta * delta
                end
                distances[candidate] = distance
            end
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
end

mutable struct TopologicalMultistartState{C,A,W}
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
    new_target = iszero(remaining) ? 0 :
        max(minimum_complex, remaining - reserve)
    sample_count = min(
        algorithm.samples,
        core.trace.count + new_target,
    )
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
        workspace.sample_values[1:initial_count] .=
            trace.objective_values[initial_indices]
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
    isempty(workspace.local_indices) &&
        push!(
            workspace.local_indices,
            argmin(_loss.(Ref(state.core.problem.sense), workspace.sample_values)),
        )
    sort!(
        workspace.local_indices;
        by=index -> _loss(state.core.problem, workspace.sample_values[index]),
    )
    resize!(workspace.local_indices, min(length(workspace.local_indices), state.algorithm.local_starts))
    workspace.local_minima_count = length(workspace.local_indices)
    workspace.initialized = true
    _begin_local_start!(state, 1)
    return state
end

function _begin_local_start!(state::TopologicalMultistartState, start)
    workspace = state.workspace
    start = mod1(start, length(workspace.local_indices))
    workspace.current_start = start
    index = workspace.local_indices[start]
    workspace.center .= @view workspace.sample_points[:, index]
    workspace.center_value = workspace.sample_values[index]
    workspace.step_size = eltype(workspace.center)(initialstep(state.algorithm.local_solver))
    return state
end

function _local_proposals(state::TopologicalMultistartState, ::PatternSearch)
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
    return @view proposals[:, 1:count]
end

function step!(state::TopologicalMultistartState)
    _finished(state.core) && return state
    !state.workspace.initialized && return _initialize_topological_multistart!(state)
    proposals = _local_proposals(state, state.algorithm.local_solver)
    isempty(proposals) && (state.core.retcode = :infeasible_space; return state)
    values = _evaluate_batch(state.core, proposals)
    state.core.iteration += 1
    _commit_batch!(state.core, proposals, values)
    best = argmin(_loss.(Ref(state.core.problem.sense), values))
    if _isbetter(
        state.core.problem,
        values[best],
        state.workspace.center_value,
    )
        state.workspace.center .= @view proposals[:, best]
        state.workspace.center_value = values[best]
    else
        state.workspace.step_size *= eltype(state.workspace.center)(0.5)
        if state.workspace.step_size < minimumstep(state.algorithm.local_solver)
            _begin_local_start!(state, state.workspace.current_start + 1)
        end
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

"""
Simplicial homology global optimization using iterative sampling-complex
refinement and local minimization of topographical minima.

`sampling_points` new vertices are added on each refinement. `local_solver` is
dispatched through `local_minimize!`, allowing external local solvers to extend
the algorithm without changing SHGO.
"""
struct MinimizeEveryRefinement end
struct MinimizeAtTermination end

struct SHGO{S,T,L,M}
    sampling::S
    topology::T
    local_solver::L
    sampling_points::Int
    local_starts::Int
    local_budget::Int
    minimum_homology_growth::Int
    homology_patience::Int
    minimization_schedule::M
end
function SHGO(;
    sampling=SobolDesign(),
    topology=DelaunayTopology(),
    local_solver=PatternSearch(),
    sampling_points=0,
    local_starts=8,
    local_budget=0,
    minimum_homology_growth=0,
    homology_patience=2,
    minimize_every_iteration=true,
    minimization_schedule=nothing,
)
    sampling_points >= 0 ||
        throw(ArgumentError("sampling_points must be nonnegative"))
    local_starts > 0 || throw(ArgumentError("local_starts must be positive"))
    local_budget >= 0 || throw(ArgumentError("local_budget must be nonnegative"))
    minimum_homology_growth >= 0 ||
        throw(ArgumentError("minimum_homology_growth must be nonnegative"))
    homology_patience > 0 ||
        throw(ArgumentError("homology_patience must be positive"))
    schedule = isnothing(minimization_schedule) ?
        (minimize_every_iteration ? MinimizeEveryRefinement() :
            MinimizeAtTermination()) : minimization_schedule
    return SHGO(
        sampling,
        topology,
        local_solver,
        sampling_points,
        local_starts,
        local_budget,
        minimum_homology_growth,
        homology_patience,
        schedule,
    )
end

minimize_candidates!(state, ::MinimizeEveryRefinement) =
    update_minimizer_pool!(state)
minimize_candidates!(state, ::MinimizeAtTermination) = state
finalize_local_search!(state, ::MinimizeEveryRefinement) = state
function finalize_local_search!(state, ::MinimizeAtTermination)
    state.workspace.sample_count == 0 && return state
    _remaining(state.core) == 0 && return state
    update_complex!(state)
    return update_minimizer_pool!(state)
end

mutable struct SHGOWorkspace{TX,TY}
    sample_points::Matrix{TX}
    sample_values::Vector{TY}
    sample_count::Int
    complex::NeighborComplex
    candidate_indices::Vector{Int}
    mapped_vertices::BitVector
    minimizer_points::Vector{Vector{TX}}
    minimizer_values::Vector{TY}
    refinements::Int
    homology_rank::Int
    homology_rank_differential::Int
    stagnant_homology_iterations::Int
    initialized_observations::Int
    local_center::Vector{TX}
    local_lower::Vector{TX}
    local_upper::Vector{TX}
    local_proposals::Matrix{TX}
    local_candidates::Matrix{TX}
    local_values::Vector{TY}
end

mutable struct SHGOState{C,A,W}
    core::C
    algorithm::A
    workspace::W
end

function init(
    problem::Problem,
    algorithm::SHGO;
    initial_points=nothing,
    initial_values=nothing,
    kwargs...,
)
    _continuous_space(problem.space) ||
        throw(ArgumentError("SHGO supports continuous dimensions only"))
    core = _makecore(problem; kwargs...)
    _seed_initial!(core, initial_points, initial_values)
    TX = eltype(core.trace.latent_points)
    TY = eltype(core.trace.objective_values)
    workspace = SHGOWorkspace(
        Matrix{TX}(
            undef,
            dimension(problem.space),
            core.criteria.maximum_evaluations + core.trace.count,
        ),
        Vector{TY}(
            undef,
            core.criteria.maximum_evaluations + core.trace.count,
        ),
        0,
        NeighborComplex(Vector{Vector{Int}}()),
        Int[],
        falses(core.criteria.maximum_evaluations + core.trace.count),
        Vector{TX}[],
        TY[],
        0,
        0,
        0,
        0,
        0,
        Vector{TX}(undef, dimension(problem.space)),
        Vector{TX}(undef, dimension(problem.space)),
        Vector{TX}(undef, dimension(problem.space)),
        Matrix{TX}(undef, dimension(problem.space), 2dimension(problem.space)),
        Matrix{TX}(undef, dimension(problem.space), 2dimension(problem.space)),
        Vector{TY}(undef, 2dimension(problem.space)),
    )
    return SHGOState(core, algorithm, workspace)
end

function _append_samples!(workspace::SHGOWorkspace, points, values)
    size(points, 2) == length(values) ||
        throw(DimensionMismatch("one value per sample point required"))
    count = length(values)
    first = workspace.sample_count + 1
    last = workspace.sample_count + count
    last <= size(workspace.sample_points, 2) ||
        throw(AssertionError("SHGO sample capacity exceeded"))
    copyto!(@view(workspace.sample_points[:, first:last]), points)
    copyto!(@view(workspace.sample_values[first:last]), values)
    workspace.sample_count = last
    return workspace
end

function _initialize_shgo_observations!(state::SHGOState)
    workspace = state.workspace
    workspace.initialized_observations > 0 && return state
    trace = state.core.trace
    if trace.count > 0
        _append_samples!(
            workspace,
            Matrix(latentpoints(trace)),
            collect(objectivevalues(trace)),
        )
    end
    workspace.initialized_observations = trace.count + 1
    return state
end

function _sampling_points_per_refinement(state::SHGOState)
    configured = state.algorithm.sampling_points
    return configured == 0 ? 2^dimension(state.core.problem.space) + 1 : configured
end

function _shgo_sampling_design(state::SHGOState)
    skip = state.workspace.sample_count
    return advance(state.algorithm.sampling, skip)
end

function refine_sampling!(state::SHGOState)
    _initialize_shgo_observations!(state)
    core = state.core
    remaining = _remaining(core)
    remaining == 0 && return state
    d = dimension(core.problem.space)
    minimum_local_reserve = min(remaining - 1, max(32, 16d))
    requested = min(
        _sampling_points_per_refinement(state),
        remaining - minimum_local_reserve,
    )
    requested <= 0 && return state
    points = _sample_feasible(core, requested, _shgo_sampling_design(state))
    values = _evaluate_batch(core, points)
    core.iteration += 1
    _commit_batch!(core, points, values)
    _append_samples!(state.workspace, points, values)
    state.workspace.refinements += 1
    return state
end

function update_complex!(state::SHGOState)
    workspace = state.workspace
    workspace.sample_count == 0 && return state
    workspace.complex = buildcomplex(
        @view(workspace.sample_points[:, 1:workspace.sample_count]),
        state.algorithm.topology,
    )
    return state
end

function local_minimum_candidates(state::SHGOState)
    workspace = state.workspace
    candidates = localcandidates(
        workspace.complex,
        _loss.(
            Ref(state.core.problem.sense),
            @view(workspace.sample_values[1:workspace.sample_count]),
        ),
    )
    filter!(candidates) do index
        !workspace.mapped_vertices[index]
    end
    sort!(
        candidates;
        by=index -> _loss(state.core.problem, workspace.sample_values[index]),
    )
    workspace.candidate_indices = candidates
    return candidates
end

homology_rank(state::SHGOState) = state.workspace.homology_rank
homology_rank_differential(state::SHGOState) =
    state.workspace.homology_rank_differential
topographical_candidate_count(state::SHGOState) =
    length(state.workspace.candidate_indices)
minimizer_count(state::SHGOState) =
    length(state.workspace.minimizer_values)

function _local_bounds!(lower, upper, state::SHGOState, vertex)
    workspace = state.workspace
    fill!(lower, zero(eltype(lower)))
    fill!(upper, one(eltype(upper)))
    center = @view workspace.sample_points[:, vertex]
    for neighbor in neighbors(workspace.complex, vertex),
            axis in eachindex(center, lower, upper)
        value = workspace.sample_points[axis, neighbor]
        value < center[axis] && (lower[axis] = max(lower[axis], value))
        value > center[axis] && (upper[axis] = min(upper[axis], value))
    end
    return lower, upper
end

function _bounded_local_proposals!(proposals, center, step, lower, upper)
    d = length(center)
    count = 0
    for axis in 1:d, direction in (-1, 1)
        proposal = @view proposals[:, count + 1]
        proposal .= center
        proposal[axis] = clamp(
            proposal[axis] + direction * step,
            lower[axis],
            upper[axis],
        )
        proposal == center && continue
        count += 1
    end
    return @view proposals[:, 1:count]
end

"""
Locally minimize a SHGO topographical candidate.

Packages can extend this method for their local-solver type. Implementations
must consume objective evaluations through `state.core` and return
`(point, value)`.
"""
function local_minimize!(
    state::SHGOState,
    solver::PatternSearch,
    start,
    start_value,
    lower,
    upper,
    budget,
)
    core = state.core
    TX = eltype(core.trace.latent_points)
    workspace = state.workspace
    center = workspace.local_center
    copyto!(center, start)
    value = start_value
    step = TX(initialstep(solver))
    used = 0
    while used < budget && !_finished(core) && step >= TX(minimumstep(solver))
        proposals = _bounded_local_proposals!(
            workspace.local_proposals,
            center,
            step,
            lower,
            upper,
        )
        count = min(size(proposals, 2), budget - used, _remaining(core))
        count == 0 && break
        feasible_count = 0
        for column in 1:count
            proposal = @view proposals[:, column]
            _canonicalize!(proposal, core.problem.space)
            isfeasible(core.problem, decode(
                core.problem.space,
                proposal,
            )) || continue
            feasible_count += 1
            copyto!(
                @view(workspace.local_candidates[:, feasible_count]),
                proposal,
            )
        end
        if feasible_count == 0
            step /= TX(2)
            continue
        end
        candidates = @view workspace.local_candidates[:, 1:feasible_count]
        values = @view workspace.local_values[1:feasible_count]
        _evaluate_batch!(values, core, candidates)
        core.iteration += 1
        _commit_batch!(core, candidates, values)
        used += feasible_count
        best = 1
        for index in 2:feasible_count
            _isbetter(core.problem, values[index], values[best]) &&
                (best = index)
        end
        if _isbetter(core.problem, values[best], value)
            center .= @view candidates[:, best]
            value = values[best]
        else
            step /= TX(2)
        end
    end
    return copy(center), value
end

function _same_minimum(left, right)
    tolerance = 32sqrt(eps(promote_type(eltype(left), eltype(right))))
    return all(abs(left[i] - right[i]) <= tolerance for i in eachindex(left, right))
end

function update_minimizer_pool!(state::SHGOState)
    workspace = state.workspace
    candidates = local_minimum_candidates(state)
    count = min(length(candidates), state.algorithm.local_starts)
    previous_rank = workspace.homology_rank
    for candidate in @view candidates[1:count]
        _finished(state.core) && break
        start = @view workspace.sample_points[:, candidate]
        workspace.mapped_vertices[candidate] = true
        lower = workspace.local_lower
        upper = workspace.local_upper
        _local_bounds!(lower, upper, state, candidate)
        automatic_budget = max(32, 16dimension(state.core.problem.space))
        budget = state.algorithm.local_budget == 0 ?
            automatic_budget : state.algorithm.local_budget
        budget = min(budget, _remaining(state.core))
        point, value = local_minimize!(
            state,
            state.algorithm.local_solver,
            start,
            workspace.sample_values[candidate],
            lower,
            upper,
            budget,
        )
        duplicate = findfirst(
            existing -> _same_minimum(existing, point),
            workspace.minimizer_points,
        )
        if isnothing(duplicate)
            push!(workspace.minimizer_points, point)
            push!(workspace.minimizer_values, value)
        elseif _isbetter(
            state.core.problem,
            value,
            workspace.minimizer_values[duplicate],
        )
            workspace.minimizer_points[duplicate] = point
            workspace.minimizer_values[duplicate] = value
        end
    end
    workspace.homology_rank = length(workspace.minimizer_values)
    workspace.homology_rank_differential =
        workspace.homology_rank - previous_rank
    if workspace.homology_rank_differential <=
            state.algorithm.minimum_homology_growth
        workspace.stagnant_homology_iterations += 1
    else
        workspace.stagnant_homology_iterations = 0
    end
    return state
end

abstract type SHGOStepOutcome end
struct Refined <: SHGOStepOutcome end
struct EvaluationBudgetExhausted <: SHGOStepOutcome end

function _step!(state::SHGOState)
    _finished(state.core) && return state
    required_vertices = dimension(state.core.problem.space) == 1 ?
        2 : dimension(state.core.problem.space) + 2
    state.workspace.sample_count + _remaining(state.core) < required_vertices &&
        return EvaluationBudgetExhausted()
    previous_samples = state.workspace.sample_count
    refine_sampling!(state)
    state.workspace.sample_count == previous_samples &&
        return EvaluationBudgetExhausted()
    update_complex!(state)
    local_minimum_candidates(state)
    minimize_candidates!(state, state.algorithm.minimization_schedule)
    if !_finished(state.core) &&
            state.workspace.refinements > 1 &&
            state.workspace.stagnant_homology_iterations >=
                state.algorithm.homology_patience
        state.core.retcode = :success
    end
    return Refined()
end

function step!(state::SHGOState)
    _step!(state)
    return state
end

function solve!(state::SHGOState)
    isnothing(state.core.problem.objective) &&
        throw(ArgumentError("solve! requires an objective"))
    try
        while !_finished(state.core)
            outcome = _step!(state)
            if outcome isa EvaluationBudgetExhausted && !_finished(state.core)
                state.core.retcode = :evaluation_limit
            end
        end
    catch error
        error isa InfeasibleSpaceError || rethrow()
        state.core.retcode = :infeasible_space
    end
    finalize_local_search!(state, state.algorithm.minimization_schedule)
    return result(state)
end
solve(problem::Problem, algorithm::SHGO; kwargs...) =
    solve!(init(problem, algorithm; kwargs...))
trace(state::SHGOState) = state.core.trace
result(state::SHGOState) = _result(
    state.core,
    state.algorithm;
    statistics=(
        iterations=state.core.iteration,
        refinements=state.workspace.refinements,
        homology_rank=state.workspace.homology_rank,
        homology_rank_differential=state.workspace.homology_rank_differential,
        local_minima=length(state.workspace.minimizer_values),
    ),
)
