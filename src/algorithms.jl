Base.@kwdef struct SMBO{T<:Real}
    initial_points::Int = 0
    candidate_pool::Int = 4096
    batch_size::Int = 1
    exploration::T = 2.0
end

Base.@kwdef struct SCEUA{R<:Real,C<:Real}
    complexes::Int = 0
    complex_size::Int = 0
    reflection::R = 1.0
    contraction::C = 0.5
end

Base.@kwdef struct SHGO
    samples::Int = 256
    local_starts::Int = 8
end

struct CandidateBatch{T,S}
    identifier::UInt64
    latent_points::Matrix{T}
    space::S
end
Base.length(batch::CandidateBatch) = size(batch.latent_points, 2)
Base.getindex(batch::CandidateBatch, i) = decode(batch.space, @view batch.latent_points[:, i])
Base.iterate(batch::CandidateBatch, i=1) = i > length(batch) ? nothing : (batch[i], i + 1)
latentpoints(batch::CandidateBatch) = batch.latent_points

mutable struct SolverState{P,A,R,E,C,T}
    problem::P
    algorithm::A
    rng::R
    executor::E
    callback::C
    trace::Trace{T}
    maximum_evaluations::Int
    iteration::Int
    started::Float64
    retcode::Symbol
    pending::Dict{UInt64,Matrix{T}}
    nextid::UInt64
end

_validate_algorithm(algorithm) = algorithm
function _validate_algorithm(algorithm::SMBO)
    algorithm.initial_points >= 0 || throw(ArgumentError("initial_points must be nonnegative"))
    algorithm.candidate_pool > 0 || throw(ArgumentError("candidate_pool must be positive"))
    algorithm.batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    isfinite(algorithm.exploration) && algorithm.exploration >= 0 ||
        throw(ArgumentError("exploration must be finite and nonnegative"))
    return algorithm
end
function _validate_algorithm(algorithm::SCEUA)
    algorithm.complexes >= 0 || throw(ArgumentError("complexes must be nonnegative"))
    algorithm.complex_size >= 0 || throw(ArgumentError("complex_size must be nonnegative"))
    isfinite(algorithm.reflection) && algorithm.reflection > 0 ||
        throw(ArgumentError("reflection must be finite and positive"))
    isfinite(algorithm.contraction) && 0 < algorithm.contraction <= 1 ||
        throw(ArgumentError("contraction must lie in (0, 1]"))
    return algorithm
end
function _validate_algorithm(algorithm::SHGO)
    algorithm.samples > 0 || throw(ArgumentError("samples must be positive"))
    algorithm.local_starts > 0 || throw(ArgumentError("local_starts must be positive"))
    return algorithm
end

function init(
    problem::Problem,
    algorithm;
    maximum_evaluations=100,
    rng=Random.default_rng(),
    executor=Serial(),
    callback=Returns(false),
)
    maximum_evaluations > 0 || throw(ArgumentError("maximum_evaluations must be positive"))
    _validate_algorithm(algorithm)
    T = Float64
    trace = Trace{T}(dimension(problem.space), maximum_evaluations)
    pending = Dict{UInt64,Matrix{T}}()
    return SolverState(
        problem, algorithm, rng, executor, callback, trace, maximum_evaluations,
        0, time(), :running, pending, UInt64(1),
    )
end
solve(problem::Problem, algorithm; kwargs...) = solve!(init(problem, algorithm; kwargs...))

function _random_feasible(state, n::Integer)
    n >= 0 || throw(ArgumentError("number of candidates must be nonnegative"))
    d = dimension(state.problem.space)
    out = Matrix{Float64}(undef, d, n)
    n == 0 && return out
    got = 0
    for _ in 1:max(1000, 100n)
        z = rand(state.rng, d)
        state.problem.constraint(decode(state.problem.space, z)) || continue
        got += 1
        out[:, got] .= z
        got == n && return out
    end
    throw(ArgumentError("unable to sample feasible points"))
end

function _evaluate_batch(state, Z)
    T = eltype(state.trace.objective_values)
    values = Vector{T}(undef, size(Z, 2))
    if state.executor isa Threaded
        Threads.@threads for j in axes(Z, 2)
            values[j] = _evaluate(state.problem, @view(Z[:, j]), T)
        end
    else
        for j in axes(Z, 2)
            values[j] = _evaluate(state.problem, @view(Z[:, j]), T)
        end
    end
    return values
end

function _validated_values(state, Z, values)
    length(values) == size(Z, 2) || throw(DimensionMismatch("one value per candidate required"))
    T = eltype(state.trace.objective_values)
    converted = Vector{T}(undef, length(values))
    for i in eachindex(converted)
        converted[i] = _checkedvalue(T, values[i])
    end
    return converted
end

function _commit_batch!(state, Z, values)
    size(Z, 2) <= state.maximum_evaluations - state.trace.count ||
        throw(ArgumentError("batch exceeds the remaining evaluation budget"))
    for j in axes(Z, 2)
        _commit!(state.trace, @view(Z[:, j]), values[j], state.iteration, state.started)
    end
    state.trace.count == state.maximum_evaluations && (state.retcode = :evaluation_limit)
    state.callback(result(state)) && (state.retcode = :callback_stop)
    return state
end

function _record_batch!(state, Z, values)
    converted = _validated_values(state, Z, values)
    return _commit_batch!(state, Z, converted)
end

function _eval_batch!(state, Z)
    values = _evaluate_batch(state, Z)
    return _commit_batch!(state, Z, values)
end

function _rbf_predict(trace, Z)
    X = latentpoints(trace)
    y = objectivevalues(trace)
    n = length(y)
    m = size(Z, 2)
    n == 0 && return zeros(m), ones(m)

    ymean = mean(y)
    μ = fill(ymean, m)
    σ = ones(m)
    scale = max(0.05, n^(-1 / max(size(X, 1), 1)))
    denominator = 2scale^2
    for j in axes(Z, 2)
        weight_sum = 0.0
        weighted_y = 0.0
        nearest = Inf
        @inbounds for i in eachindex(y)
            distance_squared = 0.0
            for k in axes(X, 1)
                difference = X[k, i] - Z[k, j]
                distance_squared += difference * difference
            end
            weight = exp(-distance_squared / denominator)
            weight_sum += weight
            weighted_y += weight * y[i]
            nearest = min(nearest, distance_squared)
        end
        μ[j] = weight_sum > eps() ? weighted_y / weight_sum : ymean
        σ[j] = sqrt(nearest + eps())
    end
    return μ, σ
end

function ask!(state::SolverState{<:Any,<:SMBO}, n::Integer=state.algorithm.batch_size)
    n >= 0 || throw(ArgumentError("batch size must be nonnegative"))
    reserved = sum(pending -> size(pending, 2), values(state.pending); init=0)
    remaining = state.maximum_evaluations - state.trace.count - reserved
    n = min(Int(n), remaining)
    if n == 0
        Z = zeros(dimension(state.problem.space), 0)
        return CandidateBatch(UInt64(0), Z, state.problem.space)
    end

    d = dimension(state.problem.space)
    initial = state.algorithm.initial_points == 0 ? max(2d + 1, 5) : state.algorithm.initial_points
    if state.trace.count < initial
        Z = _random_feasible(state, n)
    else
        candidate_count = max(state.algorithm.candidate_pool, n)
        candidates = _random_feasible(state, candidate_count)
        μ, σ = _rbf_predict(state.trace, candidates)
        scores = μ .- state.algorithm.exploration .* σ
        order = partialsortperm(scores, 1:n)
        Z = candidates[:, order]
    end

    identifier = state.nextid
    state.nextid += 1
    state.pending[identifier] = copy(Z)
    return CandidateBatch(identifier, Z, state.problem.space)
end

function tell!(state::SolverState{<:Any,<:SMBO}, batch::CandidateBatch, values)
    Z = get(state.pending, batch.identifier, nothing)
    isnothing(Z) && throw(ArgumentError("unknown or completed batch"))
    batch.latent_points == Z || throw(ArgumentError("candidate batch was modified after ask!"))
    converted = _validated_values(state, Z, values)
    delete!(state.pending, batch.identifier)
    state.iteration += 1
    return _commit_batch!(state, Z, converted)
end

function step!(state::SolverState{<:Any,<:SMBO})
    batch = ask!(state)
    if isempty(batch)
        state.retcode = :evaluation_limit
        return state
    end
    values = _evaluate_batch(state, batch.latent_points)
    return tell!(state, batch, values)
end

function step!(state::SolverState{<:Any,<:SCEUA})
    state.iteration += 1
    d = dimension(state.problem.space)
    initial = max(2d + 1, 10)
    if state.trace.count < initial
        n = min(initial - state.trace.count, state.maximum_evaluations - state.trace.count)
        _eval_batch!(state, _random_feasible(state, n))
        return state
    end

    trace = state.trace
    order = sortperm(objectivevalues(trace))
    elite = max(2, min(length(order), d + 1))
    n = min(elite, state.maximum_evaluations - trace.count)
    Z = Matrix{Float64}(undef, d, n)
    elite_points = @view trace.latent_points[:, order[1:elite]]
    centroid = vec(mean(elite_points; dims=2))
    worst = @view trace.latent_points[:, order[end]]
    for j in axes(Z, 2)
        if j == 1
            Z[:, j] .= centroid .+ state.algorithm.reflection .* (centroid .- worst)
        else
            Z[:, j] .= centroid .+ 0.2 .* randn(state.rng, d)
        end
        project!(@view Z[:, j])
        point = decode(state.problem.space, @view Z[:, j])
        if !state.problem.constraint(point)
            Z[:, j] .= @view _random_feasible(state, 1)[:, 1]
        end
    end
    _eval_batch!(state, Z)
    return state
end

function step!(state::SolverState{<:Any,<:SHGO})
    state.iteration += 1
    remaining = state.maximum_evaluations - state.trace.count
    n = min(remaining, state.trace.count == 0 ? state.algorithm.samples : max(1, state.algorithm.local_starts))
    if state.trace.count == 0
        Z = _random_feasible(state, n)
    else
        order = sortperm(objectivevalues(state.trace))
        d = dimension(state.problem.space)
        Z = Matrix{Float64}(undef, d, n)
        starts = min(length(order), state.algorithm.local_starts)
        for j in axes(Z, 2)
            base = @view state.trace.latent_points[:, order[mod1(j, starts)]]
            Z[:, j] .= base .+ (0.25 / sqrt(state.iteration)) .* randn(state.rng, d)
            project!(@view Z[:, j])
            point = decode(state.problem.space, @view Z[:, j])
            if !state.problem.constraint(point)
                Z[:, j] .= @view _random_feasible(state, 1)[:, 1]
            end
        end
    end
    _eval_batch!(state, Z)
    return state
end

function solve!(state::SolverState)
    isnothing(state.problem.objective) &&
        throw(ArgumentError("solve! requires an objective; use ask!/tell!"))
    while state.trace.count < state.maximum_evaluations && state.retcode == :running
        step!(state)
    end
    state.retcode == :running && (state.retcode = :evaluation_limit)
    return result(state)
end

function result(state::SolverState)
    if state.trace.count == 0
        return Result(
            state.problem.space, nothing, Inf, state.trace, nothing,
            state.algorithm, state.retcode, (iterations=state.iteration,),
        )
    end
    i = argmin(objectivevalues(state.trace))
    z = @view state.trace.latent_points[:, i]
    return Result(
        state.problem.space, decode(state.problem.space, z), state.trace.objective_values[i],
        state.trace, nothing, state.algorithm, state.retcode, (iterations=state.iteration,),
    )
end
