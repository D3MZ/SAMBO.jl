struct UniformCandidates end
struct GlobalLocalCandidates{T<:Real}
    global_fraction::T
    local_scale::T
end
GlobalLocalCandidates(; global_fraction=0.35, local_scale=0.15) =
    GlobalLocalCandidates(promote(global_fraction, local_scale)...)

struct SMBO{S,A,I,C}
    surrogate::S
    acquisition::A
    initial_design::I
    candidate_sampler::C
    initial_points::Int
    candidate_pool::Int
    batch_size::Int
    refit_interval::Int
end
function SMBO(;
    surrogate=GaussianProcessSurrogate(),
    acquisition=LowerConfidenceBound(),
    initial_design=LatinHypercubeDesign(),
    candidate_sampler=GlobalLocalCandidates(),
    initial_points=0,
    candidate_pool=4096,
    batch_size=1,
    refit_interval=1,
    exploration=nothing,
)
    if !isnothing(exploration)
        acquisition = LowerConfidenceBound(exploration)
    end
    initial_points >= 0 || throw(ArgumentError("initial_points must be nonnegative"))
    candidate_pool > 0 || throw(ArgumentError("candidate_pool must be positive"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    refit_interval > 0 || throw(ArgumentError("refit_interval must be positive"))
    return SMBO(
        surrogate,
        acquisition,
        initial_design,
        candidate_sampler,
        initial_points,
        candidate_pool,
        batch_size,
        refit_interval,
    )
end

struct CandidateBatch{T,S}
    identifier::UInt64
    latent_points::Matrix{T}
    space::S
end
Base.length(batch::CandidateBatch) = size(batch.latent_points, 2)
Base.isempty(batch::CandidateBatch) = length(batch) == 0
Base.getindex(batch::CandidateBatch, index) =
    decode(batch.space, @view batch.latent_points[:, index])
Base.iterate(batch::CandidateBatch, index=1) =
    index > length(batch) ? nothing : (batch[index], index + 1)
latentpoints(batch::CandidateBatch) = batch.latent_points

mutable struct SMBOState{C,A,T}
    core::C
    algorithm::A
    model::Any
    pending::Dict{UInt64,Matrix{T}}
    next_identifier::UInt64
    observations_at_fit::Int
end

function init(problem::Problem, algorithm::SMBO; initial_points=nothing, initial_values=nothing, kwargs...)
    core = _makecore(problem; kwargs...)
    _seed_initial!(core, initial_points, initial_values)
    T = eltype(core.trace.objective_values)
    return SMBOState(
        core,
        algorithm,
        nothing,
        Dict{UInt64,Matrix{T}}(),
        UInt64(1),
        0,
    )
end

function _fit_smbo!(state::SMBOState)
    trace = state.core.trace
    trace.count > 0 || return state
    if isnothing(state.model) ||
            trace.count - state.observations_at_fit >= state.algorithm.refit_interval
        state.model = fitmodel(
            state.algorithm.surrogate,
            latentpoints(trace),
            objectivevalues(trace),
            state.core.rng,
        )
        state.observations_at_fit = trace.count
    end
    return state
end

function _global_local_candidates(state::SMBOState, count, ::UniformCandidates)
    return _sample_feasible(state.core, count, UniformDesign())
end

function _global_local_candidates(state::SMBOState, count, sampler::GlobalLocalCandidates)
    core = state.core
    T = eltype(core.trace.objective_values)
    d = dimension(core.problem.space)
    candidates = Matrix{T}(undef, d, count)
    global_count = clamp(round(Int, count * sampler.global_fraction), 1, count)
    _sample_feasible!(
        core.rng,
        @view(candidates[:, 1:global_count]),
        UniformDesign(),
        core.problem,
    )
    global_count == count && return candidates

    trace = core.trace
    order = sortperm(objectivevalues(trace))
    elite_count = clamp(round(Int, sqrt(trace.count)), 1, trace.count)
    write = global_count
    attempts = 0
    while write < count && attempts < 200count
        attempts += 1
        rank = Random.rand(core.rng, 1:elite_count)
        center = @view trace.latent_points[:, order[rank]]
        proposal = @view candidates[:, write + 1]
        scale = T(sampler.local_scale) / sqrt(T(1 + core.iteration) / T(2))
        for row in 1:d
            proposal[row] = center[row] + scale * Random.randn(core.rng, T)
        end
        project!(proposal, core.problem.space)
        _canonicalize!(proposal, core.problem.space)
        isfeasible(core.problem, decode(core.problem.space, proposal)) || continue
        write += 1
    end
    if write < count
        _sample_feasible!(
            core.rng,
            @view(candidates[:, write+1:count]),
            UniformDesign(),
            core.problem,
        )
    end
    return candidates
end

@inline function _same_candidate(left, right)
    length(left) == length(right) || return false
    tolerance = sqrt(eps(promote_type(eltype(left), eltype(right))))
    @inbounds for i in eachindex(left, right)
        abs(left[i] - right[i]) <= tolerance || return false
    end
    return true
end

function _is_duplicate(state::SMBOState, candidate, selected, selected_count)
    trace = state.core.trace
    for column in 1:trace.count
        _same_candidate(candidate, @view(trace.latent_points[:, column])) && return true
    end
    for pending in values(state.pending), column in axes(pending, 2)
        _same_candidate(candidate, @view(pending[:, column])) && return true
    end
    for column in 1:selected_count
        _same_candidate(candidate, @view(selected[:, column])) && return true
    end
    return false
end

function _select_candidates(state::SMBOState, candidates, requested)
    T = eltype(candidates)
    selected = Matrix{T}(undef, size(candidates, 1), requested)
    if state.core.trace.count == 0
        selected .= @view candidates[:, 1:requested]
        return selected
    end
    _fit_smbo!(state)
    means = Vector{T}(undef, size(candidates, 2))
    variances = similar(means)
    scores = similar(means)
    predictmeanvariance!(means, variances, state.model, candidates)
    acquisitionvalues!(
        scores,
        state.algorithm.acquisition,
        means,
        variances,
        state.core.best_value,
    )
    order = sortperm(scores)
    count = 0
    for index in order
        candidate = @view candidates[:, index]
        _is_duplicate(state, candidate, selected, count) && continue
        count += 1
        selected[:, count] .= candidate
        count == requested && return selected
    end
    if count < requested
        fallback = _sample_feasible(state.core, requested - count, UniformDesign())
        selected[:, count+1:requested] .= fallback
    end
    return selected
end

function ask!(state::SMBOState, requested::Integer=state.algorithm.batch_size)
    requested >= 0 || throw(ArgumentError("batch size must be nonnegative"))
    reserved = sum(matrix -> size(matrix, 2), values(state.pending); init=0)
    count = min(Int(requested), max(0, _remaining(state.core) - reserved))
    T = eltype(state.core.trace.objective_values)
    if count == 0
        return CandidateBatch(
            UInt64(0),
            Matrix{T}(undef, dimension(state.core.problem.space), 0),
            state.core.problem.space,
        )
    end
    d = dimension(state.core.problem.space)
    initial_count = state.algorithm.initial_points == 0 ? max(2d + 1, 5) :
        state.algorithm.initial_points
    if state.core.trace.count < initial_count
        candidates = _sample_feasible(state.core, count, state.algorithm.initial_design)
    else
        pool_size = max(state.algorithm.candidate_pool, 8count)
        pool = _global_local_candidates(state, pool_size, state.algorithm.candidate_sampler)
        candidates = _select_candidates(state, pool, count)
    end
    identifier = state.next_identifier
    state.next_identifier += UInt64(1)
    state.pending[identifier] = copy(candidates)
    return CandidateBatch(identifier, candidates, state.core.problem.space)
end

function tell!(state::SMBOState, batch::CandidateBatch, values)
    candidates = get(state.pending, batch.identifier, nothing)
    isnothing(candidates) && throw(ArgumentError("unknown or completed batch"))
    batch.latent_points == candidates ||
        throw(ArgumentError("candidate batch was modified after ask!"))
    converted = _validated_values(state.core, candidates, values)
    delete!(state.pending, batch.identifier)
    state.core.iteration += 1
    _commit_batch!(state.core, candidates, converted)
    return state
end

function tell!(state::SMBOState, points, values)
    latent = reduce(hcat, (encode(state.core.problem.space, point) for point in points))
    converted = _validated_values(state.core, latent, values)
    state.core.iteration += 1
    _commit_batch!(state.core, latent, converted)
    return state
end

function step!(state::SMBOState)
    batch = ask!(state)
    if isempty(batch)
        state.core.retcode = :evaluation_limit
        return state
    end
    values = _evaluate_batch(state.core, batch.latent_points)
    return tell!(state, batch, values)
end

function solve!(state::SMBOState)
    isnothing(state.core.problem.objective) &&
        throw(ArgumentError("solve! requires an objective; use ask!/tell!"))
    while !_finished(state.core)
        step!(state)
    end
    _fit_smbo!(state)
    return result(state)
end
solve(problem::Problem, algorithm::SMBO; kwargs...) = solve!(init(problem, algorithm; kwargs...))
result(state::SMBOState) = _result(
    state.core,
    state.algorithm,
    state.model;
    statistics=(iterations=state.core.iteration, pending_batches=length(state.pending)),
)
