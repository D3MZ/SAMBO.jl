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
include("surrogates.jl")
include("smbo.jl")
include("sceua.jl")
include("shgo.jl")
include("diagnostics.jl")

const PublicSolverState = Union{SMBOState,SCEUAState,SHGOState}

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
export SMBO, SCEUA, SHGO, TopologicalMultistart, Serial, Threaded, CandidateBatch
export DelaunayTopology, KNearestTopology, ComplexConstructionError
export PatternSearch, QuasiNewtonSearch
export MinimizeEveryRefinement, MinimizeAtTermination
export RandomShiftedSampling, GlobalBoxLocalBounds, TopographicalLocalBounds
export BestLocalStarts, FarthestFromLatestMinimum
export PythonSAMBOProfile
export UniformDesign, LatinHypercubeDesign, HaltonDesign, ScrambledHaltonDesign, SobolDesign
export GaussianProcessSurrogate, LowerConfidenceBound, RandomizedLowerConfidenceBound
export GreedyMean, DistanceUncertainty, clone_surrogate
export fitmodel, predictmean!, predictmeanvariance!, predictionworkspace
export AutomaticLengthScale, IsotropicLengthScale, ARDLengthScale, GeometricJitter
export GPPredictionWorkspace
export EnsembleSurrogate
export UniformCandidates, GlobalLocalCandidates
export EliteGaussianCandidates, MixtureCandidates
export generate_candidates!
export AvoidRepeatedEvaluations, AllowRepeatedEvaluations
export CandidateGenerationError
export ExactCandidateEquality, ApproximateCandidateEquality
export FixedRefit, SquareRootRefit
export GreedyBatch, LocalPenalization, space_cardinality
export minimize, init, solve, solve!, step!, ask!, tell!, cancel!, fail!, result
export checkpoint, restore, SMBOCheckpoint
export refine_sampling!, update_complex!, local_minimum_candidates
export homology_rank, homology_rank_differential, update_minimizer_pool!
export topographical_candidate_count, minimizer_count
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
