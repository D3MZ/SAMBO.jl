# Termination

Normal return codes are `:success`, `:evaluation_limit`, `:iteration_limit`,
`:stalled`, `:callback_stop`, `:time_limit`, `:infeasible_space`, and
`:numerical_failure`.

Known `initial_values` are observations but not objective evaluations. Consequently,
they appear in the trace and observation table without consuming
`maximum_evaluations`; `evaluation_count(result)` reports actual or externally supplied
evaluations consumed by the solve.
Trace rows carry `KnownObservation`, `InternalEvaluation`, or `ExternalEvaluation`
provenance. Known observations use evaluation number zero. `result(state)` freezes the
current trace prefix; `trace(state)` exposes the live trace.

Callbacks receive an immutable `ProgressEvent` and return `true` to stop.
