"""
    NormalInverseChisq(μ, σ2, κ, ν)

A Normal-χ^-2 distribution is a conjugate prior for a Normal distribution with
unknown mean and variance.  It has parameters:

* μ: expected mean
* σ2 > 0: expected variance
* κ ≥ 0: mean confidence
* ν ≥ 0: variance confidence

The parameters have a natural interpretation when used as a prior for a Normal
distribution with unknown mean and variance: μ and σ2 are the expected mean and
variance, while κ and ν are the respective degrees of confidence (expressed in
"pseudocounts").  When interpretable parameters are important, this makes it a
slightly more convient parametrization of the conjugate prior.

Equivalent to a Normal-Inverse Gamma distribution with parameters:

* m0 = μ
* v0 = 1/κ
* shape = ν/2
* scale = νσ2/2

Based on Murphy "Conjugate Bayesian analysis of the Gaussian distribution".
"""
struct NormalInverseChisq{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ2::T
    κ::T
    ν::T

    function NormalInverseChisq{T}(μ::T, σ2::T, κ::T, ν::T) where T<:Real
        if ν < 0 || κ < 0 || σ2 ≤ 0
            throw(ArgumentError("Variance and confidence (κ and ν) must all be positive"))
        end
        new{T}(μ, σ2, κ, ν)
    end
end

NormalInverseChisq() = NormalInverseChisq{Float64}(0.0, 1.0, 0.0, 0.0)

function NormalInverseChisq(μ::Real, σ2::Real, κ::Real, ν::Real)
    T = promote_type(typeof(μ), typeof(σ2), typeof(κ), typeof(ν))
    NormalInverseChisq{T}(T(μ), T(σ2), T(κ), T(ν))
end

Base.convert(::Type{NormalInverseGamma}, d::NormalInverseChisq) =
    NormalInverseGamma(d.μ, 1/d.κ, d.ν/2, d.ν*d.σ2/2)

Base.convert(::Type{NormalInverseChisq}, d::NormalInverseGamma) =
    NormalInverseChisq(d.mu, d.scale/d.shape, 1/d.v0, d.shape*2)

insupport(::Type{NormalInverseChisq}, μ::T, σ2::T) where T<:Real =
    isfinite(μ) && zero(σ2) <= σ2 < Inf

params(d::NormalInverseChisq) = d.μ, d.σ2, d.κ, d.ν

# function pdf(d::NormalInverseChisq, μ::T, σ2::T) where T<:Real
#     Zinv = sqrt(d.κ / 2pi) / gamma(d.ν*0.5) * (d.ν * d.σ2 / 2)^(d.ν*0.5)
#     Zinv * σ2^(-(d.ν+3)*0.5) * exp( (d.ν*d.σ2 + d.κ*(d.μ - μ)^2) / (-2 * σ2))
# end

# function logpdf(d::NormalInverseChisq, μ::T, σ2::T) where T<:Real
#     logZinv = (log(d.κ) - log(2pi))*0.5 - lgamma(d.ν*0.5) + (log(d.ν) + log(d.σ2) - log(2)) * (d.ν/2)
#     logZinv + log(σ2)*(-(d.ν+3)*0.5) + (d.ν*d.σ2 + d.κ*(d.μ - μ)^2) / (-2 * σ2)
# end

pdf(d::NormalInverseChisq, μ::T, σ2::T) where T<:Real = pdf(NormalInverseGamma(d), μ, σ2)
logpdf(d::NormalInverseChisq, μ::T, σ2::T) where T<:Real = logpdf(NormalInverseGamma(d), μ, σ2)
mean(d::NormalInverseChisq) = mean(NormalInverseGamma(d))
mode(d::NormalInverseChisq) = mode(NormalInverseGamma(d))
rand(d::NormalInverseChisq) = rand(NormalInverseGamma(d))
