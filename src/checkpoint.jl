struct SMBOCheckpoint{A,M,P,R,E,C,TR,SC,N,T,F,I,W,S,OS,TX,TY}
    algorithm::A
    model::M
    pending::P
    rng::R
    executor::E
    callback::C
    trace::TR
    criteria::SC
    nonfinite::N
    iteration::Int
    retcode::Symbol
    best_value::T
    best_index::Int
    last_significant_improvement_evaluation::Int
    evaluations::Int
    next_identifier::UInt64
    observations_at_fit::Int
    failures::F
    initial_design::I
    initial_design_cursor::Int
    workspace::W
    elapsed_seconds::Float64
    space_signature::S
    sense_type::OS
    coordinate_type::TX
    objective_type::TY
end

_descriptor_signature(dimension::Continuous) =
    (:continuous, dimension.lower, dimension.upper)
_descriptor_signature(dimension::AbstractRange) =
    (:range, first(dimension), step(dimension), length(dimension))
_descriptor_signature(dimension::Choices) =
    (:choices, dimension.values)
_space_signature(space::Box) = (
    :box,
    Tuple(space.lower),
    Tuple(space.upper),
    latenttype(space),
)
_space_signature(space::SearchSpace) = (
    :search_space,
    map(_descriptor_signature, Tuple(space.dimensions)),
    dimensionnames(space),
    latenttype(space),
)

function checkpoint(state::SMBOState)
    core = state.core
    return SMBOCheckpoint(
        state.algorithm,
        deepcopy(state.model),
        deepcopy(state.pending),
        deepcopy(core.rng),
        core.executor,
        core.callback,
        _snapshot(core.trace),
        core.criteria,
        core.nonfinite,
        core.iteration,
        core.retcode,
        core.best_value,
        core.best_index,
        core.last_significant_improvement_evaluation,
        core.evaluations,
        state.next_identifier,
        state.observations_at_fit,
        deepcopy(state.failures),
        copy(state.initial_design),
        state.initial_design_cursor,
        deepcopy(state.workspace),
        (time_ns() - core.started) / 1e9,
        _space_signature(core.problem.space),
        typeof(core.problem.sense),
        eltype(core.trace.latent_points),
        eltype(core.trace.objective_values),
    )
end

function restore(problem::Problem, saved::SMBOCheckpoint)
    _space_signature(problem.space) == saved.space_signature ||
        throw(ArgumentError("checkpoint search space does not match the problem"))
    typeof(problem.sense) === saved.sense_type ||
        throw(ArgumentError("checkpoint optimization sense does not match the problem"))
    latenttype(problem.space) === saved.coordinate_type ||
        throw(ArgumentError("checkpoint coordinate type does not match the problem"))
    eltype(saved.trace.objective_values) === saved.objective_type ||
        throw(ArgumentError("checkpoint objective type metadata is inconsistent"))
    elapsed_nanoseconds = UInt64(round(saved.elapsed_seconds * 1e9))
    now = time_ns()
    started = now - min(now, elapsed_nanoseconds)
    core = SolverCore(
        problem,
        deepcopy(saved.rng),
        saved.executor,
        saved.callback,
        _snapshot(saved.trace),
        saved.criteria,
        saved.nonfinite,
        saved.iteration,
        started,
        saved.retcode,
        saved.best_value,
        saved.best_index,
        saved.last_significant_improvement_evaluation,
        saved.evaluations,
    )
    TX = eltype(core.trace.latent_points)
    pending = Dict{UInt64,PendingBatch{TX}}()
    for (identifier, batch) in saved.pending
        pending[identifier] = PendingBatch(
            Matrix{TX}(batch.points),
            copy(batch.unresolved),
        )
    end
    workspace = deepcopy(saved.workspace)
    cardinality = space_cardinality(problem.space)
    occupied = _occupancy(
        saved.algorithm.repeat_policy,
        saved.algorithm.candidate_equality,
        cardinality,
        TX,
    )
    feasible_indices = _tracks_finite_space(
        saved.algorithm.repeat_policy,
        cardinality,
    ) ? _finite_feasible_indices(problem) : nothing
    for column in 1:core.trace.count
        _occupy!(
            occupied,
            problem.space,
            @view(core.trace.latent_points[:, column]),
        )
    end
    for batch in values(pending), column in axes(batch.points, 2)
        batch.unresolved[column] &&
            _occupy!(
                occupied,
                problem.space,
                @view(batch.points[:, column]),
            )
    end
    Model = Union{
        Nothing,
        fittedmodeltype(
            saved.algorithm.surrogate,
            TX,
            eltype(core.trace.objective_values),
        ),
    }
    return SMBOState{
        typeof(core),
        typeof(saved.algorithm),
        TX,
        typeof(workspace),
        Model,
        typeof(occupied),
        typeof(feasible_indices),
    }(
        core,
        saved.algorithm,
        deepcopy(saved.model),
        pending,
        saved.next_identifier,
        saved.observations_at_fit,
        deepcopy(saved.failures),
        Matrix{TX}(saved.initial_design),
        saved.initial_design_cursor,
        workspace,
        occupied,
        feasible_indices,
    )
end
