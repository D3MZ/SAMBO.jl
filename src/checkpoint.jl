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
    return (
        algorithm=state.algorithm,
        model=deepcopy(state.model),
        pending=deepcopy(state.pending),
        rng=deepcopy(core.rng),
        executor=core.executor,
        callback=core.callback,
        trace=_snapshot(core.trace),
        criteria=core.criteria,
        nonfinite=core.nonfinite,
        iteration=core.iteration,
        retcode=core.retcode,
        best_value=core.best_value,
        best_index=core.best_index,
        last_significant_improvement_evaluation=
            core.last_significant_improvement_evaluation,
        evaluations=core.evaluations,
        next_identifier=state.next_identifier,
        observations_at_fit=state.observations_at_fit,
        failures=deepcopy(state.failures),
        initial_design=copy(state.initial_design),
        initial_design_cursor=state.initial_design_cursor,
        elapsed_seconds=(time_ns() - core.started) / 1e9,
        space_signature=_space_signature(core.problem.space),
        sense_type=typeof(core.problem.sense),
    )
end

function restore(problem::Problem, saved)
    _space_signature(problem.space) == saved.space_signature ||
        throw(ArgumentError("checkpoint search space does not match the problem"))
    typeof(problem.sense) === saved.sense_type ||
        throw(ArgumentError("checkpoint optimization sense does not match the problem"))
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
    TY = eltype(core.trace.objective_values)
    workspace = _make_smbo_workspace(
        saved.algorithm,
        TX,
        TY,
        dimension(problem.space),
    )
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
