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

function convergenceplot end
function convergenceplot! end
function regretplot end
function regretplot! end
function objectiveplot end
function objectiveplot! end
function evaluationsplot end
function evaluationsplot! end
function SAMBOTuning end

export Problem, Result, Trace, StopCriteria, ProgressEvent, BatchProgressEvent
export ObservationSource
export KnownObservation, InternalEvaluation, ExternalEvaluation
export OptimizationSense, Minimize, Maximize
export NonfinitePolicy, ErrorOnNonfinite, PenalizeNonfinite, AllowInfinite
export NumericalFailureError
export Box, SearchSpace, Continuous, Choices
export SMBO, SCEUA, SHGO, TopologicalMultistart, Serial, Threaded, CandidateBatch
export DelaunayTopology, KNearestTopology, ComplexConstructionError
export MinimizeEveryRefinement, MinimizeAtTermination
export UniformDesign, LatinHypercubeDesign, HaltonDesign, SobolDesign
export GaussianProcessSurrogate, LowerConfidenceBound
export GreedyMean, DistanceUncertainty, clone_surrogate
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
export minimizer, minimum, trace, retcode, evaluation_count, iteration_count
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
