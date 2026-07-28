module SAMBO

using LinearAlgebra
using Statistics

import CommonSolve: init, solve, solve!, step!
import MiniQhull
import QuasiMonteCarlo
import Random
import Tables
import Base: minimum

include("spaces.jl")
include("core.jl")
include("sampling.jl")
include("evaluation.jl")
include("bounded_optimization.jl")
include("surrogates.jl")
include("smbo.jl")
include("candidate_strategies.jl")
include("smbo_finite_spaces.jl")
include("smbo_batches.jl")
include("checkpoint.jl")
include("sceua.jl")
include("shgo.jl")
include("extras/topological_multistart.jl")
include("diagnostics.jl")

const PublicSolverState = Union{
    SMBOState,
    SCEUAState,
    SHGOState,
    TopologicalMultistartState,
}

"""Return the problem owned by a live solver state."""
problem(state::PublicSolverState) = state.core.problem
problem(result::Result) = result.problem

"""Return the search space owned by a solver state or result."""
space(state::PublicSolverState) = problem(state).space
space(result::Result) = result.space

"""Return the explicit random-number generator owned by a solver state."""
rng(state::PublicSolverState) = state.core.rng

"""Return the number of objective evaluations remaining in a solver state."""
remaining_evaluations(state::PublicSolverState) = _remaining(state.core)

"""Return the current solver iteration."""
iteration(state::PublicSolverState) = state.core.iteration

"""
Evaluate and atomically commit a latent candidate batch through a solver state.

This is the public evaluation boundary for external local solvers. It enforces
the remaining budget, updates the trace and best observation, and invokes the
configured batch callback.
"""
function evaluate!(state::PublicSolverState, values, candidates)
    size(candidates, 2) <= remaining_evaluations(state) ||
        throw(ArgumentError("batch exceeds the remaining evaluation budget"))
    _evaluate_batch!(values, state.core, candidates)
    _commit_batch!(state.core, candidates, values)
    return values
end

"""Return whether a latent candidate is feasible for a live solver state."""
isfeasible(state::PublicSolverState, latent::AbstractVector) =
    _isfeasible_latent(problem(state), latent)

function convergenceplot end
function convergenceplot! end
function regretplot end
function regretplot! end
function objectiveplot end
function objectiveplot! end
function evaluationsplot end
function evaluationsplot! end
function SAMBOTuning end

export Problem, Result, Trace, StopCriteria, BatchProgressEvent
export ObservationSource
export KnownObservation, InternalEvaluation, ExternalEvaluation
export OptimizationSense, Minimize, Maximize
export NonfinitePolicy, ErrorOnNonfinite, PenalizeNonfinite, AllowInfinite
export NumericalFailureError
export Box, SearchSpace, Continuous, Choices
export SMBO, SCEUA, SHGO, TopologicalMultistart
export GaussianProcessSurrogate, LowerConfidenceBound, RandomizedLowerConfidenceBound
export GreedyMean, DistanceUncertainty, clone_surrogate
export fitmodel, predictmean!, predictmeanvariance!, predictionworkspace
export AutomaticLengthScale, IsotropicLengthScale, ARDLengthScale, GeometricJitter
export GPPredictionWorkspace
export EnsembleSurrogate
export generate_candidates!
export CandidateGenerationError
export minimize, init, solve, solve!, step!, ask!, tell!, cancel!, fail!, result
export checkpoint, restore
export local_minimize!
export problem, space, rng, remaining_evaluations, iteration, evaluate!
export minimizer, minimum, bestpoint, bestvalue, optimizationsense
export trace, retcode, evaluation_count, iteration_count
export fittedmodel
export observations, latentpoints, objectivevalues, encode, decode
export constraint_violation, isfeasible
export convergencedata, regretdata, partialdependence, evaluationsdata
export EvaluationsOnly, IncludeKnownObservations
export FeasibleConditionalDependence, UnconstrainedModelDependence
export PartialDependenceWorkspace
export convergenceplot, convergenceplot!, regretplot, regretplot!
export objectiveplot, objectiveplot!, evaluationsplot, evaluationsplot!
export SAMBOTuning

end
