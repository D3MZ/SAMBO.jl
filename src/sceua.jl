struct MoveTowardCentroid end
struct RejectAndResample end
struct ProjectToBounds end

struct SCEUA{R<:Real,C<:Real,P,T<:Real,U<:Real}
    complexes::Int
    complex_size::Int
    reflection::R
    contraction::C
    repair::P
    population_tolerance::T
    objective_tolerance::U
    stall_iterations::Int
end
function SCEUA(;
    complexes=0,
    complex_size=0,
    reflection=1.0,
    contraction=0.5,
    repair=MoveTowardCentroid(),
    population_tolerance=1e-7,
    objective_tolerance=1e-7,
    stall_iterations=30,
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
    objective_tolerance >= 0 ||
        throw(ArgumentError("objective_tolerance must be nonnegative"))
    stall_iterations > 0 ||
        throw(ArgumentError("stall_iterations must be positive"))
    return SCEUA(
        complexes,
        complex_size,
        reflection,
        contraction,
        repair,
        population_tolerance,
        objective_tolerance,
        stall_iterations,
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
    initialized::Bool
    finite_feasible::Union{Nothing,Vector{Int}}
    occupied::BitSet
    complexes::Int
    complex_population::Matrix{TX}
    complex_values::Vector{TY}
    complex_losses::Vector{TY}
    complex_permutation::Vector{Int}
    complex_members::Vector{Int}
    previous_best_loss::TY
    stalled_iterations::Int
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
    complex_size = algorithm.complex_size == 0 ? 2 : algorithm.complex_size
    automatic_complexes = min(
        max(2, core.criteria.maximum_evaluations ÷ complex_size - 1),
        max(5, floor(Int, 3log2(d))),
    )
    complexes = algorithm.complexes == 0 ? automatic_complexes : algorithm.complexes
    finite_feasible = _finite_feasible_indices(problem)
    population_limit = isnothing(finite_feasible) ?
        complexes * complex_size : length(finite_feasible)
    population_size = min(
        core.trace.count + _remaining(core),
        population_limit,
    )
    !isnothing(finite_feasible) && isempty(finite_feasible) &&
        (core.retcode = :infeasible_space)
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
        false,
        finite_feasible,
        BitSet(),
        complexes,
        Matrix{TX}(undef, d, population_size + 1),
        Vector{TY}(undef, population_size + 1),
        Vector{TY}(undef, population_size + 1),
        Vector{Int}(undef, population_size + 1),
        Vector{Int}(undef, population_size + 1),
        convert(TY, Inf),
        0,
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
        if !isnothing(workspace.finite_feasible)
            for column in 1:initial_count
                push!(
                    workspace.occupied,
                    canonical_index(
                        state.core.problem.space,
                        @view(workspace.population[:, column]),
                    ),
                )
            end
        end
    end
    sample_count = population_count - initial_count
    if sample_count > 0
        sampled = @view workspace.population[:, initial_count+1:population_count]
        if isnothing(workspace.finite_feasible)
            _sample_feasible!(
                state.core.rng,
                sampled,
                LatinHypercubeDesign(),
                state.core.problem,
            )
        else
            unused = filter(
                index -> !(index in workspace.occupied),
                workspace.finite_feasible,
            )
            Random.shuffle!(state.core.rng, unused)
            length(unused) >= sample_count ||
                throw(AssertionError("finite SCE-UA population exceeds unused points"))
            for column in 1:sample_count
                canonical_latent!(
                    @view(sampled[:, column]),
                    state.core.problem.space,
                    unused[column],
                )
                push!(workspace.occupied, unused[column])
            end
        end
        values = _evaluate_batch(state.core, sampled)
        state.core.iteration += 1
        _commit_batch!(state.core, sampled, values)
        workspace.values[initial_count+1:population_count] .= values
    end
    workspace.initialized = true
    !isnothing(workspace.finite_feasible) &&
        length(workspace.occupied) >= length(workspace.finite_feasible) &&
        !_finished(state.core) &&
        (state.core.retcode = :space_exhausted)
    return state
end

function _project_feasible!(proposal, problem)
    project!(proposal, problem.space)
    return isfeasible(problem, decode(problem.space, proposal))
end

function repair!(rng, proposal, ::MoveTowardCentroid, problem, centroid)
    original = copy(proposal)
    _project_feasible!(proposal, problem) && return true
    for fraction in 0.1:0.1:1.0
        @. proposal = (1 - fraction) * original + fraction * centroid
        _project_feasible!(proposal, problem) && return true
    end
    return repair!(
        rng,
        proposal,
        RejectAndResample(),
        problem,
        centroid,
    )
end
function repair!(
    rng,
    proposal,
    ::MoveTowardCentroid,
    problem::Problem{F,S,Unconstrained},
    centroid,
) where {F,S}
    _project_feasible!(proposal, problem)
    return true
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

function _python_complex_members!(members, population_size, complexes, complex_index)
    count = 1
    members[count] = 1
    for member in complex_index:complexes:population_size
        count += 1
        members[count] = member
    end
    return count
end

function _sceua_theta(space::Box)
    width = zero(eltype(space.lower))
    for index in eachindex(space.lower, space.upper)
        width = max(width, space.upper[index] - space.lower[index])
    end
    width <= 0 && return 0.2
    log_width = log10(width)
    return clamp(0.2 + 0.1 * (log_width - 2), 0.2, 0.5)
end
_sceua_theta(::SearchSpace) = 0.2

function _reflection!(proposal, centroid, worst, coefficient)
    @. proposal = centroid + coefficient * (centroid - worst)
    return proposal
end

function _contraction!(proposal, centroid, worst, coefficient)
    @. proposal = worst + coefficient * (centroid - worst)
    return proposal
end

function _sceua_span(workspace, space)
    isempty(workspace.values) && return Inf
    total_span = zero(eltype(workspace.population))
    for row in axes(workspace.population, 1)
        values = @view workspace.population[row, :]
        total_span += maximum(values) - minimum(values)
    end
    return total_span
end
function _sceua_span(workspace, space::Box)
    isempty(workspace.values) && return Inf
    total_span = zero(eltype(workspace.population))
    for row in axes(workspace.population, 1)
        values = @view workspace.population[row, :]
        total_span +=
            (maximum(values) - minimum(values)) *
            (space.upper[row] - space.lower[row])
    end
    return total_span
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

    population_size = size(workspace.population, 2)
    best_loss = workspace.losses[workspace.permutation[1]]
    improvement = workspace.previous_best_loss - best_loss
    if (
        state.algorithm.objective_tolerance > 0 &&
        improvement < state.algorithm.objective_tolerance
    ) || workspace.previous_best_loss == best_loss
        workspace.stalled_iterations += 1
        if workspace.stalled_iterations == state.algorithm.stall_iterations
            core.retcode = :success
            return state
        end
    else
        workspace.previous_best_loss = best_loss
        workspace.stalled_iterations = 0
    end
    if state.algorithm.population_tolerance > 0 &&
            _sceua_span(workspace, core.problem.space) <
            state.algorithm.population_tolerance
        core.retcode = :success
        return state
    end

    complexes = min(workspace.complexes, population_size)
    theta = _sceua_theta(core.problem.space)
    members = workspace.complex_members
    core.iteration += 1
    evolved = false
    for complex_index in 1:complexes
        member_count = _python_complex_members!(
            members,
            population_size,
            complexes,
            complex_index,
        )
        member_count <= 1 && continue
        worst_index = members[member_count]
        centroid = @view workspace.centroids[:, complex_index]
        fill!(centroid, zero(eltype(centroid)))
        for member_position in 1:member_count-1
            centroid .+= @view workspace.population[:, members[member_position]]
        end
        centroid ./= member_count - 1

        proposal = @view workspace.proposals[:, complex_index]
        worst = @view workspace.population[:, worst_index]
        _reflection!(proposal, centroid, worst, state.algorithm.reflection)
        @. proposal = (1 - theta) * proposal +
            theta * workspace.population[:, 1]
        repair!(
            core.rng,
            proposal,
            state.algorithm.repair,
            core.problem,
            centroid,
        ) || continue
        _remaining(core) == 0 && return state
        evolved = true
        proposal_matrix = @view workspace.proposals[:, complex_index:complex_index]
        proposal_value = @view workspace.proposal_values[complex_index:complex_index]
        _evaluate_batch!(proposal_value, core, proposal_matrix)
        _commit_batch!(core, proposal_matrix, proposal_value)
        accepted = false
        if _isbetter(
            core.problem,
            proposal_value[1],
            workspace.values[worst_index],
        )
            workspace.population[:, worst_index] .= proposal
            workspace.values[worst_index] = proposal_value[1]
            accepted = true
        else
            _contraction!(
                proposal,
                centroid,
                worst,
                state.algorithm.contraction,
            )
            @. proposal = (1 - theta) * proposal +
                theta * workspace.population[:, 1]
            repair!(
                core.rng,
                proposal,
                state.algorithm.repair,
                core.problem,
                centroid,
            ) || continue
            _finished(core) && return state
            _evaluate_batch!(proposal_value, core, proposal_matrix)
            _commit_batch!(core, proposal_matrix, proposal_value)
            best_index = members[1]
            if _isbetter(
                core.problem,
                proposal_value[1],
                workspace.values[best_index],
            )
                workspace.population[:, worst_index] .= proposal
                workspace.values[worst_index] = proposal_value[1]
                accepted = true
            end
            if !accepted
                _finished(core) && return state
                replacement = @view workspace.proposals[:, complex_index:complex_index]
                _sample_feasible!(
                    core.rng,
                    replacement,
                    UniformDesign(),
                    core.problem,
                )
                _evaluate_batch!(proposal_value, core, replacement)
                _commit_batch!(core, replacement, proposal_value)
                workspace.population[:, worst_index] .= proposal
                workspace.values[worst_index] = proposal_value[1]
            end
        end

        for member_position in 1:member_count
            member = members[member_position]
            workspace.complex_population[:, member_position] .=
                @view workspace.population[:, member]
            workspace.complex_values[member_position] = workspace.values[member]
            workspace.complex_losses[member_position] =
                _loss(core.problem.sense, workspace.values[member])
        end
        local_permutation =
            @view workspace.complex_permutation[1:member_count]
        sortperm!(
            local_permutation,
            @view(workspace.complex_losses[1:member_count]),
        )
        for member_position in 1:member_count
            member = members[member_position]
            source = local_permutation[member_position]
            workspace.population[:, member] .=
                @view workspace.complex_population[:, source]
            workspace.values[member] = workspace.complex_values[source]
        end
        _finished(core) && return state
    end
    !evolved && !_finished(core) && (core.retcode = :stalled)
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
        population_span=state.workspace.initialized ?
                        _sceua_span(state.workspace, state.core.problem.space) :
                        Inf,
    ),
)
