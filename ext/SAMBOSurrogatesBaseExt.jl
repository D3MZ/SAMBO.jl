module SAMBOSurrogatesBaseExt

using SAMBO
using Statistics
import SurrogatesBase

struct DeterministicSurrogateModel{M,X}
    surrogate::M
    observed_points::X
end

struct StochasticSurrogateModel{M}
    surrogate::M
end

function _pointvector(points)
    return [collect(column) for column in eachcol(points)]
end

function SAMBO.fitmodel(
    specification::SurrogatesBase.AbstractDeterministicSurrogate,
    points,
    values,
    rng,
)
    surrogate = deepcopy(specification)
    SurrogatesBase.update!(surrogate, _pointvector(points), collect(values))
    return DeterministicSurrogateModel(surrogate, Matrix(points))
end

function SAMBO.fitmodel(
    specification::SurrogatesBase.AbstractStochasticSurrogate,
    points,
    values,
    rng,
)
    surrogate = deepcopy(specification)
    SurrogatesBase.update!(surrogate, _pointvector(points), collect(values))
    return StochasticSurrogateModel(surrogate)
end

function SAMBO.predictmeanvariance!(means, variances, model::DeterministicSurrogateModel, points)
    predictions = model.surrogate(_pointvector(points))
    means .= predictions
    for column in axes(points, 2)
        nearest = eltype(variances)(Inf)
        for observation in axes(model.observed_points, 2)
            distance = zero(eltype(variances))
            for row in axes(points, 1)
                delta = points[row, column] - model.observed_points[row, observation]
                distance += delta * delta
            end
            nearest = min(nearest, distance)
        end
        variances[column] = nearest
    end
    return means, variances
end

function SAMBO.predictmean!(means, model::DeterministicSurrogateModel, points)
    means .= model.surrogate(_pointvector(points))
    return means
end

function SAMBO.predictmeanvariance!(means, variances, model::StochasticSurrogateModel, points)
    posterior = SurrogatesBase.finite_posterior(model.surrogate, _pointvector(points))
    means .= mean(posterior)
    variances .= var(posterior)
    return means, variances
end

function SAMBO.predictmean!(means, model::StochasticSurrogateModel, points)
    posterior = SurrogatesBase.finite_posterior(model.surrogate, _pointvector(points))
    means .= mean(posterior)
    return means
end

end
