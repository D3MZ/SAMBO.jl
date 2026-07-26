struct MoveTowardCentroid end
struct RejectAndResample end
struct ProjectToBounds end

struct SCEUA{R<:Real,C<:Real,P}
    complexes::Int
    complex_size::Int
    reflection::R
    contraction::C
    repair::P
    population_tolerance::Float64
end
function SCEUA(;
    complexes=0,
    complex_size=0,
    reflection=1.0,
    contraction=0.5,
    repair=MoveTowardCentroid(),
    population_tolerance=0.0,
)
    complexes >= 0 || throw(ArgumentError("complexes must be nonnegative"))
    complex_size >= 0 || throw(ArgumentError("complex_size must be nonnegative"))
    isfinite(reflection) && reflection > 0 ||
        throw(ArgumentError("reflection must be finite and positive"))
    isfinite(contraction) && 0 < contraction <= 1 ||
        throw(ArgumentError("contraction must lie in (0, 1]"))
    population_tolerance >= 0 ||
        throw(ArgumentError("population_tolerance must be nonnegative"))
    return SCEUA(
        complexes,
        complex_size,
        reflection,
        contraction,
        repair,
        Float64(population_tolerance),
    )
end

mutable struct SCEUAWorkspace{T}
    population::Matrix{T}
    values::Vector{T}
    initialized::Bool
end

mutable struct SCEUAState{C,A,W}
    core::C
    algorithm::A
    workspace::W
end

function init(problem::Problem, algorithm::SCEUA; initial_points=nothing, initial_values=nothing, kwargs...)
    core = _makecore(problem; kwargs...)
    _seed_initial!(core, initial_points, initial_values)
    d = dimension(problem.space)
    complexes = algorithm.complexes == 0 ? clamp(d, 2, 5) : algorithm.complexes
    complex_size = algorithm.complex_size == 0 ? max(2d + 1, 5) : algorithm.complex_size
    population_size = min(_remaining(core), complexes * complex_size)
    T = eltype(core.trace.objective_values)
    workspace = SCEUAWorkspace(
        Matrix{T}(undef, d, population_size),
        Vector{T}(undef, population_size),
        false,
    )
    return SCEUAState(core, algorithm, workspace)
end

function _initialize_sceua!(state::SCEUAState)
    workspace = state.workspace
    sample_count = size(workspace.population, 2)
    _sample_feasible!(
        state.core.rng,
        workspace.population,
        LatinHypercubeDesign(),
        state.core.problem,
    )
    values = _evaluate_batch(state.core, workspace.population)
    state.core.iteration += 1
    _commit_batch!(state.core, workspace.population, values)
    workspace.values .= values
    workspace.initialized = true
    return state
end

function _repair_proposal!(state::SCEUAState, proposal, centroid)
    project!(proposal, state.core.problem.space)
    _canonicalize!(proposal, state.core.problem.space)
    isfeasible(state.core.problem, decode(state.core.problem.space, proposal)) && return proposal
    policy = state.algorithm.repair
    if policy isa MoveTowardCentroid
        for fraction in (0.25, 0.5, 0.75, 0.9, 1.0)
            @. proposal = (1 - fraction) * proposal + fraction * centroid
            project!(proposal, state.core.problem.space)
            _canonicalize!(proposal, state.core.problem.space)
            isfeasible(state.core.problem, decode(state.core.problem.space, proposal)) &&
                return proposal
        end
    elseif policy isa ProjectToBounds
        return proposal
    end
    replacement = _sample_feasible(state.core, 1, UniformDesign())
    proposal .= @view replacement[:, 1]
    return proposal
end

function _complex_indices(population_size, complexes, complex_index)
    indices = collect(complex_index:complexes:population_size)
    if complex_index != 1 && !isempty(indices)
        indices[1] = 1 # Retain SAMBO's global-best participation in every complex.
        unique!(indices)
    end
    return indices
end

function _sceua_span(workspace)
    isempty(workspace.values) && return Inf
    maximum_span = zero(eltype(workspace.population))
    for row in axes(workspace.population, 1)
        values = @view workspace.population[row, :]
        maximum_span = max(maximum_span, maximum(values) - minimum(values))
    end
    return maximum_span
end

function step!(state::SCEUAState)
    _finished(state.core) && return state
    !state.workspace.initialized && return _initialize_sceua!(state)

    core = state.core
    workspace = state.workspace
    permutation = sortperm(workspace.values)
    workspace.population .= workspace.population[:, permutation]
    workspace.values .= workspace.values[permutation]

    d = size(workspace.population, 1)
    population_size = size(workspace.population, 2)
    requested_complexes = state.algorithm.complexes == 0 ? clamp(d, 2, 5) :
        state.algorithm.complexes
    complexes = min(requested_complexes, population_size)
    proposals = Matrix{eltype(workspace.population)}(undef, d, complexes)
    worst_indices = Vector{Int}(undef, complexes)
    active = 0
    for complex_index in 1:complexes
        indices = _complex_indices(population_size, complexes, complex_index)
        length(indices) >= 2 || continue
        simplex_count = min(d + 1, length(indices))
        simplex_indices = indices[1:simplex_count]
        worst_index = simplex_indices[end]
        centroid = vec(mean(@view(workspace.population[:, simplex_indices[1:end-1]]); dims=2))
        active += 1
        proposal = @view proposals[:, active]
        worst = @view workspace.population[:, worst_index]
        @. proposal = centroid + state.algorithm.reflection * (centroid - worst)
        _repair_proposal!(state, proposal, centroid)
        worst_indices[active] = worst_index
    end
    active = min(active, _remaining(core))
    active == 0 && (core.retcode = :evaluation_limit; return state)
    proposals = @view proposals[:, 1:active]
    worst_indices = @view worst_indices[1:active]
    reflection_values = _evaluate_batch(core, proposals)
    core.iteration += 1
    _commit_batch!(core, proposals, reflection_values)

    failed = Int[]
    for index in 1:active
        population_index = worst_indices[index]
        if reflection_values[index] < workspace.values[population_index]
            workspace.population[:, population_index] .= @view proposals[:, index]
            workspace.values[population_index] = reflection_values[index]
        else
            push!(failed, index)
        end
    end
    _finished(core) && return state

    contraction_count = min(length(failed), _remaining(core))
    if contraction_count > 0
        contractions = Matrix{eltype(workspace.population)}(undef, d, contraction_count)
        for (output_index, failed_index) in enumerate(@view failed[1:contraction_count])
            population_index = worst_indices[failed_index]
            worst = @view workspace.population[:, population_index]
            reflected = @view proposals[:, failed_index]
            proposal = @view contractions[:, output_index]
            @. proposal = worst + state.algorithm.contraction * (reflected - worst)
            _repair_proposal!(state, proposal, worst)
        end
        contraction_values = _evaluate_batch(core, contractions)
        _commit_batch!(core, contractions, contraction_values)
        for output_index in 1:contraction_count
            failed_index = failed[output_index]
            population_index = worst_indices[failed_index]
            if contraction_values[output_index] < workspace.values[population_index]
                workspace.population[:, population_index] .= @view contractions[:, output_index]
                workspace.values[population_index] = contraction_values[output_index]
                failed[output_index] = 0
            end
        end
    end
    _finished(core) && return state

    remaining_failed = filter(value -> !iszero(value), failed)
    replacement_count = min(length(remaining_failed), _remaining(core))
    if replacement_count > 0
        replacements = _sample_feasible(core, replacement_count, UniformDesign())
        replacement_values = _evaluate_batch(core, replacements)
        _commit_batch!(core, replacements, replacement_values)
        for output_index in 1:replacement_count
            failed_index = remaining_failed[output_index]
            population_index = worst_indices[failed_index]
            workspace.population[:, population_index] .= @view replacements[:, output_index]
            workspace.values[population_index] = replacement_values[output_index]
        end
    end
    if state.algorithm.population_tolerance > 0 &&
            _sceua_span(workspace) <= state.algorithm.population_tolerance &&
            !_finished(core)
        core.retcode = :success
    end
    return state
end

function solve!(state::SCEUAState)
    isnothing(state.core.problem.objective) &&
        throw(ArgumentError("solve! requires an objective"))
    while !_finished(state.core)
        step!(state)
    end
    return result(state)
end
solve(problem::Problem, algorithm::SCEUA; kwargs...) = solve!(init(problem, algorithm; kwargs...))
result(state::SCEUAState) = _result(
    state.core,
    state.algorithm;
    statistics=(
        iterations=state.core.iteration,
        population_span=state.workspace.initialized ? _sceua_span(state.workspace) : Inf,
    ),
)
