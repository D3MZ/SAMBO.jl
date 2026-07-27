# Termination

Normal return codes are `:success`, `:evaluation_limit`, `:iteration_limit`,
`:stalled`, `:callback_stop`, `:time_limit`, `:infeasible_space`, and
`:numerical_failure`. Finite discrete spaces return `:space_exhausted` after
every feasible canonical point has been occupied.

Known initial values are observations but not objective evaluations. Trace rows
carry `KnownObservation`, `InternalEvaluation`, or `ExternalEvaluation`
provenance. Known observations use evaluation number zero.

Callbacks receive one immutable `BatchProgressEvent` after a complete batch is
committed. `:infeasible_space` is reserved for an actual feasible-sampling
failure. A phase that cannot start within the remaining budget returns
`:evaluation_limit`.

An internally driven SMBO `ask!` sets a terminal return code before returning an
empty batch. `step!` preserves that reason, including `:space_exhausted`.
