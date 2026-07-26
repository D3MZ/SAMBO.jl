module Sambo

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
function SamboTuning end

export Problem, Result, Trace, StopCriteria, ProgressEvent
export Box, SearchSpace, Continuous, Choices
export SMBO, SCEUA, SHGO, Serial, Threaded, CandidateBatch
export UniformDesign, LatinHypercubeDesign, HaltonDesign, SobolDesign
export GaussianProcessSurrogate, LowerConfidenceBound
export UniformCandidates, GlobalLocalCandidates
export minimize, init, solve, solve!, step!, ask!, tell!, result
export minimizer, minimum, trace, retcode, evaluation_count, iteration_count
export fittedmodel
export observations, latentpoints, objectivevalues, encode, decode
export convergencedata, regretdata, partialdependence, evaluationsdata
export convergenceplot, convergenceplot!, regretplot, regretplot!
export objectiveplot, objectiveplot!, evaluationsplot, evaluationsplot!
export SamboTuning

end
