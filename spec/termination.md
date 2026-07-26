# Termination

Normal return codes are `:success`, `:evaluation_limit`, `:iteration_limit`,
`:stalled`, `:callback_stop`, `:time_limit`, `:infeasible_space`, and
`:numerical_failure`. Finite discrete spaces additionally return
`:space_exhausted` after every canonical point has been occupied.

Known `initial_values` are observations but not objective evaluations. Consequently,
they appear in the trace and observation table without consuming
`maximum_evaluations`; `evaluation_count(result)` reports actual or externally supplied
evaluations consumed by the solve.
Trace rows carry `KnownObservation`, `InternalEvaluation`, or `ExternalEvaluation`
provenance. Known observations use evaluation number zero. `result(state)` freezes the
current trace prefix; `trace(state)` exposes the live trace.

Callbacks receive one immutable `BatchProgressEvent` after a complete batch is
committed and return `true` to stop. The event contains the final evaluation counter
and `batch_size`; callbacks never observe a partially committed batch.

`:infeasible_space` is produced only by an actual feasible-sampling failure. A SHGO
phase that cannot begin within the remaining evaluation budget returns
`:evaluation_limit`.

An internally driven SMBO `ask!` sets a terminal return code before returning an empty
batch. `step!` preserves that reason, including `:space_exhausted`.
