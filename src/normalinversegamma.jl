
# Used "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy as
# a reference.  Note that there were some typos in that document so the code
# here may not correspond exactly.

struct NormalInverseGamma{T<:Real} <: ContinuousUnivariateDistribution
    mu::T
    v0::T     # scales variance of Normal
    shape::T  
    scale::T

    function NormalInverseGamma{T}(mu::T, v0::T, sh::T, r::T) where T<:Real
    	v0 > zero(v0) && sh > zero(sh) && r > zero(r) || error("Both shape and scale must be positive")
    	new{T}(T(mu), T(v0), T(sh), T(r))
    end
end

function NormalInverseGamma(mu::Real, v0::Real, sh::Real, r::Real)
    T = promote_type(typeof(mu), typeof(v0), typeof(sh), typeof(r))
    return NormalInverseGamma{T}(T(mu),T(v0),T(sh),T(r))
end

mu(d::NormalInverseGamma) = d.mu
v0(d::NormalInverseGamma) = d.v0
shape(d::NormalInverseGamma) = d.shape
scale(d::NormalInverseGamma) = d.scale
rate(d::NormalInverseGamma) = 1. / d.scale

insupport(::Type{NormalInverseGamma}, x::T, sig2::T) where T<:Real = 
    isfinite(x) && zero(sig2) <= sig2 < Inf 

# Probably should guard agains dividing by and taking the log of 0.

function pdf(d::NormalInverseGamma, x::T, sig2::T) where T<:Real
    Zinv = d.scale.^d.shape / gamma(d.shape) / sqrt(d.v0 * 2.0*pi)
    return Zinv * 1.0/(sqrt(sig2)*sig2.^(d.shape+1.0)) * exp(-d.scale/sig2 - 0.5/(sig2*d.v0)*(x-d.mu).^2)
end

function logpdf(d::NormalInverseGamma, x::T, sig2::T) where T<:Real
    lZinv = d.shape*log(d.scale) - lgamma(d.shape) - 0.5*(log(d.v0) + log(2pi))
    return lZinv - 0.5*log(sig2) - (d.shape+1.)*log(sig2) - d.scale/sig2 - 0.5/(sig2*d.v0)*(x-d.mu).^2
end

function mode(d::NormalInverseGamma)
    mu = d.mu
    sig2 = d.scale / (d.shape + 1.5)
    return mu, sig2
end

function mean(d::NormalInverseGamma)
    mu = d.mu
    sig2 = d.shape > 1.0 ? d.scale / (d.shape - 1.0) : Inf
    return mu, sig2
end

function rand(d::NormalInverseGamma)
    # Guard against invalid precisions
    sig2 = rand(InverseGamma(d.shape, d.scale))
    if sig2 <= zero(Float64)
        sig2 = eps(Float64)
    end
    mu = rand(Normal(d.mu, sqrt(sig2*d.v0)))
    return mu, sig2
end
