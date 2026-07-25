module Sambo

using Statistics

import CommonSolve: init, solve, solve!, step!
import Random
import Tables
import Base: minimum

include("spaces.jl")
include("core.jl")
include("algorithms.jl")
include("diagnostics.jl")

function convergenceplot end
function convergenceplot! end
function regretplot end
function regretplot! end
function objectiveplot end
function objectiveplot! end
function evaluationsplot end
function evaluationsplot! end

export Problem, Result, Trace, Box, SearchSpace, Continuous, Choices
export SMBO, SCEUA, SHGO, Serial, Threaded, CandidateBatch
export minimize, init, solve, solve!, step!, ask!, tell!, result
export minimizer, minimum, trace, retcode, evaluation_count, iteration_count
export observations, latentpoints, objectivevalues, encode, decode
export convergencedata, regretdata, partialdependence, evaluationsdata
export convergenceplot, convergenceplot!, regretplot, regretplot!
export objectiveplot, objectiveplot!, evaluationsplot, evaluationsplot!

end
