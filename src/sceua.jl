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
    complex_size == 0 || complex_size >= 2 ||
        throw(ArgumentError("complex_size must be zero (automatic) or at least two"))
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

mutable struct SCEUAWorkspace{TX,TY}
    population::Matrix{TX}
    values::Vector{TY}
    sorted_population::Matrix{TX}
    sorted_values::Vector{TY}
    losses::Vector{TY}
    permutation::Vector{Int}
    centroids::Matrix{TX}
    proposals::Matrix{TX}
    proposal_values::Vector{TY}
    worst_indices::Vector{Int}
    failed_indices::Vector{Int}
    valid_indices::Vector{Int}
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
    population_size = min(core.trace.count + _remaining(core), complexes * complex_size)
    TX = eltype(core.trace.latent_points)
    TY = eltype(core.trace.objective_values)
    workspace = SCEUAWorkspace(
        Matrix{TX}(undef, d, population_size),
        Vector{TY}(undef, population_size),
        Matrix{TX}(undef, d, population_size),
        Vector{TY}(undef, population_size),
        Vector{TY}(undef, population_size),
        collect(1:population_size),
        Matrix{TX}(undef, d, population_size),
        Matrix{TX}(undef, d, population_size),
        Vector{TY}(undef, population_size),
        Vector{Int}(undef, population_size),
        Vector{Int}(undef, population_size),
        Vector{Int}(undef, population_size),
        false,
    )
    return SCEUAState(core, algorithm, workspace)
end

function _initialize_sceua!(state::SCEUAState)
    workspace = state.workspace
    population_count = size(workspace.population, 2)
    trace = state.core.trace
    initial_count = min(trace.count, population_count)
    if initial_count > 0
        initial_indices = partialsortperm(
            _loss.(Ref(state.core.problem.sense), objectivevalues(trace)),
            1:initial_count,
        )
        workspace.population[:, 1:initial_count] .=
            @view trace.latent_points[:, initial_indices]
        workspace.values[1:initial_count] .= trace.objective_values[initial_indices]
    end
    sample_count = population_count - initial_count
    if sample_count > 0
        sampled = @view workspace.population[:, initial_count+1:population_count]
        _sample_feasible!(
            state.core.rng,
            sampled,
            LatinHypercubeDesign(),
            state.core.problem,
        )
        values = _evaluate_batch(state.core, sampled)
        state.core.iteration += 1
        _commit_batch!(state.core, sampled, values)
        workspace.values[initial_count+1:population_count] .= values
    end
    workspace.initialized = true
    return state
end

function _project_feasible!(proposal, problem)
    project!(proposal, problem.space)
    return isfeasible(problem, decode(problem.space, proposal))
end

function repair!(rng, proposal, ::MoveTowardCentroid, problem, centroid)
    original = copy(proposal)
    _project_feasible!(proposal, problem) && return true
    for fraction in (0.25, 0.5, 0.75, 0.9, 1.0)
        @. proposal = (1 - fraction) * original + fraction * centroid
        _project_feasible!(proposal, problem) && return true
    end
    return false
end

function repair!(rng, proposal, ::RejectAndResample, problem, centroid)
    buffer = reshape(proposal, :, 1)
    try
        _sample_feasible!(rng, buffer, UniformDesign(), problem)
        return true
    catch error
        error isa InfeasibleSpaceError || rethrow()
        return false
    end
end

repair!(rng, proposal, ::ProjectToBounds, problem, centroid) =
    _project_feasible!(proposal, problem)

function _complex_members(population_size, complexes, complex_index)
    return collect(complex_index:complexes:population_size)
end

function _complex_centroid!(centroid, population, members, global_best)
    worst = members[end]
    fill!(centroid, zero(eltype(centroid)))
    count = 0
    if !(global_best in members[1:end-1])
        centroid .+= @view population[:, global_best]
        count += 1
    end
    for member in @view members[1:end-1]
        centroid .+= @view population[:, member]
        count += 1
    end
    centroid ./= count
    return centroid
end
function _complex_centroid!(
    centroid,
    population,
    population_size::Int,
    complexes::Int,
    complex_index::Int,
    global_best::Int,
)
    worst = complex_index +
        fld(population_size - complex_index, complexes) * complexes
    worst == complex_index && return 0
    fill!(centroid, zero(eltype(centroid)))
    count = 0
    contains_global_best = false
    for member in complex_index:complexes:(worst - complexes)
        centroid .+= @view population[:, member]
        contains_global_best |= member == global_best
        count += 1
    end
    if !contains_global_best
        centroid .+= @view population[:, global_best]
        count += 1
    end
    centroid ./= count
    return worst
end

function _reflection!(proposal, centroid, worst, coefficient)
    @. proposal = centroid + coefficient * (centroid - worst)
    return proposal
end

function _contraction!(proposal, centroid, worst, coefficient)
    @. proposal = worst + coefficient * (centroid - worst)
    return proposal
end

_complex_indices(population_size, complexes, complex_index) =
    _complex_members(population_size, complexes, complex_index)

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
    for index in eachindex(workspace.values)
        workspace.losses[index] =
            _loss(core.problem.sense, workspace.values[index])
    end
    sortperm!(workspace.permutation, workspace.losses)
    for destination in eachindex(workspace.permutation)
        source = workspace.permutation[destination]
        copyto!(
            @view(workspace.sorted_population[:, destination]),
            @view(workspace.population[:, source]),
        )
        workspace.sorted_values[destination] = workspace.values[source]
    end
    copyto!(workspace.population, workspace.sorted_population)
    copyto!(workspace.values, workspace.sorted_values)

    d = size(workspace.population, 1)
    population_size = size(workspace.population, 2)
    requested_complexes = state.algorithm.complexes == 0 ? clamp(d, 2, 5) :
        state.algorithm.complexes
    complexes = min(requested_complexes, population_size)
    proposals = workspace.proposals
    worst_indices = workspace.worst_indices
    centroids = workspace.centroids
    active = 0
    for complex_index in 1:complexes
        active += 1
        proposal = @view proposals[:, active]
        centroid = @view centroids[:, active]
        worst_index = _complex_centroid!(
            centroid,
            workspace.population,
            population_size,
            complexes,
            complex_index,
            1,
        )
        iszero(worst_index) && (active -= 1; continue)
        worst = @view workspace.population[:, worst_index]
        _reflection!(proposal, centroid, worst, state.algorithm.reflection)
        repair!(
            core.rng,
            proposal,
            state.algorithm.repair,
            core.problem,
            centroid,
        ) || (active -= 1; continue)
        worst_indices[active] = worst_index
    end
    active = min(active, _remaining(core))
    active == 0 && (core.retcode = :infeasible_space; return state)
    proposals = @view proposals[:, 1:active]
    worst_indices = @view worst_indices[1:active]
    reflection_values = @view workspace.proposal_values[1:active]
    _evaluate_batch!(reflection_values, core, proposals)
    core.iteration += 1
    _commit_batch!(core, proposals, reflection_values)

    failed = workspace.failed_indices
    failed_count = 0
    for index in 1:active
        population_index = worst_indices[index]
        if _isbetter(
            core.problem,
            reflection_values[index],
            workspace.values[population_index],
        )
            workspace.population[:, population_index] .= @view proposals[:, index]
            workspace.values[population_index] = reflection_values[index]
        else
            failed_count += 1
            failed[failed_count] = index
        end
    end
    _finished(core) && return state

    contraction_count = min(failed_count, _remaining(core))
    if contraction_count > 0
        contractions = workspace.proposals
        contraction_to_failure = workspace.valid_indices
        valid_count = 0
        for failure_position in 1:contraction_count
            failed_index = failed[failure_position]
            population_index = worst_indices[failed_index]
            worst = @view workspace.population[:, population_index]
            proposal = @view contractions[:, valid_count + 1]
            centroid = @view centroids[:, failed_index]
            _contraction!(
                proposal,
                centroid,
                worst,
                state.algorithm.contraction,
            )
            repair!(
                core.rng,
                proposal,
                state.algorithm.repair,
                core.problem,
                centroid,
            ) || continue
            valid_count += 1
            contraction_to_failure[valid_count] = failure_position
        end
        if valid_count > 0
            valid_contractions = @view contractions[:, 1:valid_count]
            contraction_values = @view workspace.proposal_values[1:valid_count]
            _evaluate_batch!(contraction_values, core, valid_contractions)
            _commit_batch!(core, valid_contractions, contraction_values)
            for output_index in 1:valid_count
                failure_position = contraction_to_failure[output_index]
                failed_index = failed[failure_position]
                population_index = worst_indices[failed_index]
                if _isbetter(
                    core.problem,
                    contraction_values[output_index],
                    workspace.values[population_index],
                )
                    workspace.population[:, population_index] .=
                        @view contractions[:, output_index]
                    workspace.values[population_index] =
                        contraction_values[output_index]
                    failed[failure_position] = 0
                end
            end
        end
    end
    _finished(core) && return state

    remaining_count = 0
    for index in 1:failed_count
        iszero(failed[index]) && continue
        remaining_count += 1
        failed[remaining_count] = failed[index]
    end
    replacement_count = min(remaining_count, _remaining(core))
    if replacement_count > 0
        replacements = @view workspace.proposals[:, 1:replacement_count]
        _sample_feasible!(
            core.rng,
            replacements,
            UniformDesign(),
            core.problem,
        )
        replacement_values = @view workspace.proposal_values[1:replacement_count]
        _evaluate_batch!(replacement_values, core, replacements)
        _commit_batch!(core, replacements, replacement_values)
        for output_index in 1:replacement_count
            failed_index = failed[output_index]
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
solve(problem::Problem, algorithm::SCEUA; kwargs...) = solve!(init(problem, algorithm; kwargs...))
trace(state::SCEUAState) = state.core.trace
result(state::SCEUAState) = _result(
    state.core,
    state.algorithm;
    statistics=(
        iterations=state.core.iteration,
        population_span=state.workspace.initialized ? _sceua_span(state.workspace) : Inf,
    ),
)
