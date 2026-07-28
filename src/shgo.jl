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
    function PatternSearch(initial_step::T, minimum_step::T) where {T<:Real}
        isfinite(initial_step) && initial_step > 0 ||
            throw(ArgumentError("initial_step must be finite and positive"))
        isfinite(minimum_step) && minimum_step > 0 ||
            throw(ArgumentError("minimum_step must be finite and positive"))
        minimum_step <= initial_step ||
            throw(ArgumentError("minimum_step must not exceed initial_step"))
        new{T}(initial_step, minimum_step)
    end
end
function PatternSearch(; initial_step=0.15, minimum_step=1e-5)
    return PatternSearch(promote(initial_step, minimum_step)...)
end
initialstep(solver::PatternSearch) = solver.initial_step
minimumstep(solver::PatternSearch) = solver.minimum_step

struct QuasiNewtonSearch{T<:Real}
    finite_difference_step::T
    gradient_tolerance::T
    minimum_step::T
    function QuasiNewtonSearch(
        finite_difference_step::T,
        gradient_tolerance::T,
        minimum_step::T,
    ) where {T<:Real}
        isfinite(finite_difference_step) && finite_difference_step > 0 ||
            throw(ArgumentError("finite_difference_step must be finite and positive"))
        isfinite(gradient_tolerance) && gradient_tolerance > 0 ||
            throw(ArgumentError("gradient_tolerance must be finite and positive"))
        isfinite(minimum_step) && minimum_step > 0 ||
            throw(ArgumentError("minimum_step must be finite and positive"))
        new{T}(finite_difference_step, gradient_tolerance, minimum_step)
    end
end
function QuasiNewtonSearch(;
    finite_difference_step=1e-7,
    gradient_tolerance=1e-9,
    minimum_step=1e-8,
)
    return QuasiNewtonSearch(promote(
        finite_difference_step,
        gradient_tolerance,
        minimum_step,
    )...)
end

struct NeighborComplex
    adjacency::Vector{Vector{Int}}
    vertex_order::Vector{Int}
end
NeighborComplex(adjacency::Vector{Vector{Int}}) =
    NeighborComplex(adjacency, collect(eachindex(adjacency)))
neighbors(complex::NeighborComplex, vertex) = complex.adjacency[vertex]

function _connect_delaunay_simplex!(
    adjacency,
    simplex,
    count;
    scipy_shgo_adjacency=false,
    vertex_order=nothing,
    inserted=nothing,
)
    if scipy_shgo_adjacency
        isnothing(vertex_order) && (vertex_order = Int[])
        isnothing(inserted) && (inserted = falses(count))
    end
    vertices = scipy_shgo_adjacency ?
        @view(simplex[1:min(3, length(simplex))]) :
        simplex
    for left in eachindex(vertices), right in left+1:length(vertices)
        a = Int(vertices[left])
        b = Int(vertices[right])
        (1 <= a <= count && 1 <= b <= count) || continue
        if scipy_shgo_adjacency
            if !inserted[a]
                push!(vertex_order, a)
                inserted[a] = true
            end
            if !inserted[b]
                push!(vertex_order, b)
                inserted[b] = true
            end
        end
        push!(adjacency[a], b)
        push!(adjacency[b], a)
    end
    return adjacency
end

function _delaunay_complex(points, options; scipy_shgo_adjacency=false)
    eltype(points) <: Union{Float32,Float64} || throw(ArgumentError(
        "Delaunay topology supports Float32 and Float64 coordinates",
    ))
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
        MiniQhull.delaunay(points, options)
    catch error
        error isa ErrorException || rethrow()
        throw(ComplexConstructionError(
            "Delaunay construction failed: $(sprint(showerror, error))",
        ))
    end
    adjacency = [Int[] for _ in 1:count]
    vertex_order = Int[]
    inserted = falses(count)
    for simplex in eachcol(simplices)
        _connect_delaunay_simplex!(
            adjacency,
            simplex,
            count;
            scipy_shgo_adjacency,
            vertex_order,
            inserted,
        )
    end
    foreach(unique!, adjacency)
    if !scipy_shgo_adjacency
        all(list -> !isempty(list), adjacency) ||
            throw(ComplexConstructionError(
                "Delaunay construction produced isolated vertices",
            ))
    end
    return NeighborComplex(
        adjacency,
        scipy_shgo_adjacency ? vertex_order : collect(1:count),
    )
end
buildcomplex(points, ::DelaunayTopology) =
    _delaunay_complex(points, "qhull d Qt Qbb Qc Qz QJ Pp")
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
    for vertex in complex.vertex_order
        all(neighbor -> values[vertex] <= values[neighbor], neighbors(complex, vertex)) &&
            push!(candidates, vertex)
    end
    return candidates
end

"""
Simplicial homology global optimization using iterative sampling-complex
refinement and local minimization of topographical minima.

`sampling_points` new vertices are added on each refinement. `local_solver` is
dispatched through `local_minimize!`, allowing external local solvers to extend
the algorithm without changing SHGO.
"""
struct MinimizeEveryRefinement end
struct MinimizeAtTermination end
struct RandomShiftedSampling{S}
    design::S
end
struct FixedShiftDesign{S,V}
    design::S
    shift::V
end
advance(design::FixedShiftDesign, count) =
    FixedShiftDesign(advance(design.design, count), design.shift)
struct FixedScrambledHaltonDesign
    skip::Int
    seed::UInt64
end
struct GlobalBoxLocalBounds end
struct TopographicalLocalBounds end
struct BestLocalStarts end

struct SHGO{S,T,L,M,B,P}
    sampling::S
    topology::T
    local_solver::L
    sampling_points::Int
    local_starts::Int
    local_budget::Int
    minimum_homology_growth::Int
    homology_patience::Int
    minimization_schedule::M
    local_bounds::B
    local_start_policy::P
    minimum_local_reserve::Int
    divide_automatic_local_budget::Bool
    convergence_tolerance::Float64
    convergence_window::Int
end
function SHGO(;
    sampling=RandomShiftedSampling(SobolDesign()),
    topology=DelaunayTopology(),
    local_solver=QuasiNewtonSearch(),
    sampling_points=0,
    local_starts=2,
    local_budget=0,
    minimum_homology_growth=0,
    homology_patience=2,
    minimize_every_iteration=true,
    minimization_schedule=nothing,
    local_bounds=GlobalBoxLocalBounds(),
    local_start_policy=BestLocalStarts(),
    minimum_local_reserve=-1,
    divide_automatic_local_budget=true,
    convergence_tolerance=0.0,
    convergence_window=0,
)
    sampling_points >= 0 ||
        throw(ArgumentError("sampling_points must be nonnegative"))
    local_starts > 0 || throw(ArgumentError("local_starts must be positive"))
    local_budget >= 0 || throw(ArgumentError("local_budget must be nonnegative"))
    minimum_homology_growth >= 0 ||
        throw(ArgumentError("minimum_homology_growth must be nonnegative"))
    homology_patience > 0 ||
        throw(ArgumentError("homology_patience must be positive"))
    minimum_local_reserve >= -1 ||
        throw(ArgumentError("minimum_local_reserve must be at least -1"))
    isfinite(convergence_tolerance) && convergence_tolerance >= 0 ||
        throw(ArgumentError("convergence_tolerance must be finite and nonnegative"))
    convergence_window >= 0 ||
        throw(ArgumentError("convergence_window must be nonnegative"))
    iszero(convergence_window) == iszero(convergence_tolerance) ||
        throw(ArgumentError(
            "convergence_tolerance and convergence_window must both be zero or positive",
        ))
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
        local_bounds,
        local_start_policy,
        minimum_local_reserve,
        divide_automatic_local_budget,
        Float64(convergence_tolerance),
        convergence_window,
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
    sampling_shift::Any
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
    sampling_shift = _sampling_shift(
        core.rng,
        algorithm.sampling,
        TX,
        dimension(problem.space),
    )
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
        sampling_shift,
        Vector{TX}(undef, dimension(problem.space)),
        Vector{TX}(undef, dimension(problem.space)),
        Vector{TX}(undef, dimension(problem.space)),
        Matrix{TX}(undef, dimension(problem.space), 2dimension(problem.space)),
        Matrix{TX}(undef, dimension(problem.space), 2dimension(problem.space)),
        Vector{TY}(undef, 2dimension(problem.space)),
    )
    return SHGOState(core, algorithm, workspace)
end

function _sampling_shift(rng, ::RandomShiftedSampling, ::Type{T}, dimensions) where {T}
    shift = Vector{T}(undef, dimensions)
    Random.rand!(rng, shift)
    return shift
end
_sampling_shift(rng, sampling, ::Type{T}, dimensions) where {T} =
    zeros(T, dimensions)
_sampling_shift(rng, design::ScrambledHaltonDesign, ::Type{T}, dimensions) where {T} =
    rand(rng, UInt64) ⊻ design.seed

function sample!(
    rng,
    destination::AbstractMatrix,
    design::FixedShiftDesign,
    space,
)
    sample!(rng, destination, design.design, space)
    @inbounds for column in axes(destination, 2), axis in axes(destination, 1)
        destination[axis, column] = mod(
            destination[axis, column] + design.shift[axis],
            one(eltype(destination)),
        )
    end
    return _canonicalize_samples!(destination, space)
end

function sample!(
    rng,
    destination::AbstractMatrix,
    design::FixedScrambledHaltonDesign,
    space,
)
    _scrambled_halton!(destination, design.skip, design.seed)
    return _canonicalize_samples!(destination, space)
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
    return configured == 0 ?
        automatic_sampling_count(
            dimension(state.core.problem.space),
            _remaining(state.core),
        ) :
        configured
end

function automatic_sampling_count(dimension_count::Integer, remaining::Integer)
    dimension_count >= 0 ||
        throw(ArgumentError("dimension count must be nonnegative"))
    remaining >= 0 ||
        throw(ArgumentError("remaining budget must be nonnegative"))
    remaining <= 1 && return Int(remaining)
    count = 1
    for _ in 1:dimension_count
        count > (remaining - 1) ÷ 2 && return Int(remaining)
        count *= 2
    end
    return min(count + 1, Int(remaining))
end

function _shgo_sampling_design(state::SHGOState)
    skip = state.workspace.sample_count
    return advance(state.algorithm.sampling, skip)
end
advance(sampling::RandomShiftedSampling, count) =
    RandomShiftedSampling(advance(sampling.design, count))
function _shgo_sampling_design(
    state::SHGOState{C,<:SHGO{<:RandomShiftedSampling}},
) where {C}
    sampling = state.algorithm.sampling
    return FixedShiftDesign(
        advance(sampling.design, state.workspace.sample_count),
        state.workspace.sampling_shift,
    )
end
function _shgo_sampling_design(
    state::SHGOState{C,<:SHGO{<:ScrambledHaltonDesign}},
) where {C}
    return FixedScrambledHaltonDesign(
        state.algorithm.sampling.skip + state.workspace.sample_count,
        state.workspace.sampling_shift,
    )
end

function refine_sampling!(state::SHGOState)
    _initialize_shgo_observations!(state)
    core = state.core
    remaining = _remaining(core)
    remaining == 0 && return state
    d = dimension(core.problem.space)
    configured_reserve = state.algorithm.minimum_local_reserve
    minimum_local_reserve = configured_reserve < 0 ?
        min(remaining - 1, max(32, 16d)) :
        min(remaining - 1, configured_reserve)
    requested = min(
        _sampling_points_per_refinement(state),
        remaining - minimum_local_reserve,
    )
    requested <= 0 && return state
    points = _sample_feasible(core, requested, _shgo_sampling_design(state))
    values = Vector{eltype(core.trace.objective_values)}(undef, requested)
    core.iteration += 1
    evaluated = 0
    if state.algorithm.convergence_window > 0
        for column in axes(points, 2)
            point = @view points[:, column:column]
            value = @view values[column:column]
            _evaluate_batch!(value, core, point)
            _commit_batch!(core, point, value)
            evaluated += 1
            if _shgo_mark_value_spread_converged!(state)
                break
            end
        end
    else
        _evaluate_batch!(values, core, points)
        _commit_batch!(core, points, values)
        evaluated = requested
    end
    evaluated > 0 && _append_samples!(
        state.workspace,
        @view(points[:, 1:evaluated]),
        @view(values[1:evaluated]),
    )
    state.workspace.refinements += 1
    return state
end

function _shgo_value_spread_converged(state::SHGOState)
    window = state.algorithm.convergence_window
    window > 0 || return false
    trace = state.core.trace
    losses = [
        _loss(state.core.problem.sense, trace.objective_values[index])
        for index in 1:trace.count
        if trace.source[index] != KnownObservation
    ]
    length(losses) >= window || return false
    best = partialsort!(losses, 1:window)
    return best[end] - best[1] < state.algorithm.convergence_tolerance
end

function _shgo_mark_value_spread_converged!(state::SHGOState)
    state.core.retcode == :running &&
        _shgo_value_spread_converged(state) &&
        (state.core.retcode = :success)
    return state.core.retcode == :success
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
    _filter_topological_candidates!(candidates, state)
    _order_local_candidates!(candidates, state)
    workspace.candidate_indices = candidates
    return candidates
end

_filter_topological_candidates!(candidates, state) = candidates

function _order_local_candidates!(candidates, state)
    sort!(
        candidates;
        by=index -> _loss(
            state.core.problem,
            state.workspace.sample_values[index],
        ),
    )
    return candidates
end

homology_rank(state::SHGOState) = state.workspace.homology_rank
homology_rank_differential(state::SHGOState) =
    state.workspace.homology_rank_differential
topographical_candidate_count(state::SHGOState) =
    length(state.workspace.candidate_indices)
minimizer_count(state::SHGOState) =
    length(state.workspace.minimizer_values)

function _local_bounds!(
    lower,
    upper,
    state::SHGOState,
    vertex,
    ::GlobalBoxLocalBounds,
)
    fill!(lower, zero(eltype(lower)))
    fill!(upper, one(eltype(upper)))
    return lower, upper
end

function _local_bounds!(
    lower,
    upper,
    state::SHGOState,
    vertex,
    ::TopographicalLocalBounds,
)
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
use the public solver-context accessors and consume objective evaluations through
`evaluate!(state, values, candidates)`, then return `(point, value)`.
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

function _local_gradient!(
    gradient,
    state::SHGOState,
    solver::QuasiNewtonSearch,
    center,
    value,
    lower,
    upper,
)
    core = state.core
    workspace = state.workspace
    TX = eltype(center)
    d = length(center)
    proposals = workspace.local_candidates
    deltas = workspace.local_center
    count = 0
    for axis in 1:d
        physical_step = max(
            TX(solver.finite_difference_step),
            sqrt(eps(TX)),
        )
        h = _latent_finite_difference_step(
            core.problem.space,
            axis,
            physical_step,
        )
        delta = center[axis] + h <= upper[axis] ? h : -h
        center[axis] + delta >= lower[axis] || return 0
        count += 1
        proposal = @view proposals[:, count]
        copyto!(proposal, center)
        proposal[axis] += delta
        _canonicalize!(proposal, core.problem.space)
        deltas[axis] = proposal[axis] - center[axis]
        deltas[axis] != zero(TX) || return 0
    end
    values = @view workspace.local_values[1:count]
    _evaluate_batch!(values, core, @view(proposals[:, 1:count]))
    core.iteration += 1
    _commit_batch!(core, @view(proposals[:, 1:count]), values)
    _shgo_mark_value_spread_converged!(state)
    center_loss = _loss(core.problem, value)
    for axis in 1:d
        gradient[axis] =
            (_loss(core.problem, values[axis]) - center_loss) /
            (deltas[axis] * _local_coordinate_scale(core.problem.space, axis))
    end
    return count
end

_latent_finite_difference_step(space, axis, step) = step
_local_coordinate_scale(space, axis) = one(latenttype(space))
_local_coordinate_scale(space::Box, axis) =
    space.upper[axis] - space.lower[axis]
function _latent_finite_difference_step(space::Box, axis, step)
    width = _local_coordinate_scale(space, axis)
    return iszero(width) ? step : step / width
end

function _quasi_newton_minimize!(
    state::SHGOState,
    solver::QuasiNewtonSearch,
    start,
    start_value,
    lower,
    upper,
    budget,
)
    if !(state.core.problem.constraint isa Unconstrained)
        return local_minimize!(
            state,
            PatternSearch(),
            start,
            start_value,
            lower,
            upper,
            budget,
        )
    end
    core = state.core
    TX = eltype(core.trace.latent_points)
    d = length(start)
    scales = TX[
        _local_coordinate_scale(core.problem.space, axis)
        for axis in 1:d
    ]
    physical_start = start .* scales
    physical_lower = lower .* scales
    physical_upper = upper .* scales
    last_value = Ref(start_value)
    gradient = Vector{TX}(undef, d)

    evaluate = function(point, gradient_required)
        latent = point ./ scales
        _canonicalize!(latent, core.problem.space)
        if gradient_required
            count = _local_gradient!(
                gradient,
                state,
                solver,
                latent,
                last_value[],
                lower,
                upper,
            )
            return _loss(core.problem, last_value[]),
                count == d ? copy(gradient) : nothing,
                count
        end
        proposal = @view state.workspace.local_candidates[:, 1]
        copyto!(proposal, latent)
        values = @view state.workspace.local_values[1:1]
        _evaluate_batch!(
            values,
            core,
            @view(state.workspace.local_candidates[:, 1:1]),
        )
        core.iteration += 1
        _commit_batch!(
            core,
            @view(state.workspace.local_candidates[:, 1:1]),
            values,
        )
        last_value[] = values[1]
        _shgo_mark_value_spread_converged!(state)
        return _loss(core.problem, values[1]), nothing, 1
    end

    point, loss, _ = _bounded_bfgs(
        evaluate,
        physical_start,
        physical_lower,
        physical_upper;
        initial_value=_loss(core.problem, start_value),
        max_iterations=typemax(Int),
        max_cost=min(budget, _remaining(core)),
        gradient_cost=d,
        gradient_tolerance=solver.gradient_tolerance,
        minimum_step=solver.minimum_step,
        value_tolerance=solver.gradient_tolerance,
        retry_identity=true,
        reset_on_bad_curvature=true,
        stop=() -> _finished(core),
    )
    return point ./ scales, _loss(core.problem.sense, loss)
end

local_minimize!(
    state::SHGOState,
    solver::QuasiNewtonSearch,
    start,
    start_value,
    lower,
    upper,
    budget,
) = _quasi_newton_minimize!(
    state,
    solver,
    start,
    start_value,
    lower,
    upper,
    budget,
)

function _same_minimum(left, right)
    tolerance = 32sqrt(eps(promote_type(eltype(left), eltype(right))))
    return all(abs(left[i] - right[i]) <= tolerance for i in eachindex(left, right))
end

_next_local_candidate!(candidates, state, ::BestLocalStarts, latest) =
    popfirst!(candidates)

function update_minimizer_pool!(state::SHGOState)
    workspace = state.workspace
    candidates = copy(local_minimum_candidates(state))
    count = min(length(candidates), state.algorithm.local_starts)
    automatic_budget = count == 0 ? 0 :
        state.algorithm.divide_automatic_local_budget ?
            max(1, _remaining(state.core) ÷ count) :
            _remaining(state.core)
    previous_rank = workspace.homology_rank
    latest_minimizer = nothing
    for _ in 1:count
        _finished(state.core) && break
        candidate = _next_local_candidate!(
            candidates,
            state,
            state.algorithm.local_start_policy,
            latest_minimizer,
        )
        start = @view workspace.sample_points[:, candidate]
        workspace.mapped_vertices[candidate] = true
        lower = workspace.local_lower
        upper = workspace.local_upper
        _local_bounds!(
            lower,
            upper,
            state,
            candidate,
            state.algorithm.local_bounds,
        )
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
        latest_minimizer = point
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
