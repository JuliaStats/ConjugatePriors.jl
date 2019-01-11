
# Used "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy as
# a reference.  Note that there were some typos in that document so the code
# here may not correspond exactly.

struct NormalInverseWishart{T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: ContinuousUnivariateDistribution
    dim::Int
    zeromean::Bool
    mu::V
    kappa::T              # This scales precision (inverse covariance)
    Lamchol::Cholesky{T,M}  # Covariance matrix (well, sqrt of one)
    nu::T

    function NormalInverseWishart{T}(mu::AbstractVector{T}, kappa::T,
                                     Lamchol::Cholesky{T,M}, nu::T) where {T<:Real, M<:AbstractMatrix{T}}
        # Probably should put some error checking in here
        d = length(mu)
        zmean::Bool = true
        for i = 1:d
            if !iszero(mu[i])
                zmean = false
                break
            end
        end
        new{T,typeof(mu),M}(d, zmean, mu, T(kappa), Lamchol, T(nu))
    end
end

function NormalInverseWishart(mu::AbstractVector{U}, kappa::Real,
                                Lamchol::Cholesky{S}, nu::Real) where {S<:Real, U<:Real}
    T = promote_type(eltype(mu), typeof(kappa), typeof(nu), S)
    return NormalInverseWishart{T}(AbstractVector{T}(mu), T(kappa), Cholesky{T}(Lamchol), T(nu))
end

function NormalInverseWishart(mu::AbstractVector{U}, kappa::Real,
                              Lambda::AbstractMatrix{S}, nu::Real) where {S<:Real, U<:Real}
    T = promote_type(eltype(mu), typeof(kappa), typeof(nu), S)
    return NormalInverseWishart{T}(AbstractVector{T}(mu), T(kappa), Cholesky{T}(cholesky(Lambda)), T(nu))
end

function insupport(::Type{NormalInverseWishart}, x::AbstractVector{T}, Sig::AbstractMatrix{T}) where T<:Real
    return (all(isfinite, x) &&
           size(Sig, 1) == size(Sig, 2) &&
           isApproxSymmmetric(Sig) &&
           size(Sig, 1) == length(x) &&
           hasCholesky(Sig))
end

"""
    params(niw::NormalInverseWishart)

The parameters are
* μ::AbstractVector{T<:Real} the expected mean vector
* Λchol::Cholesky{T<:Real} the Cholesky decomposition of the scale matrix
* κ::T<:Real prior pseudocount for the mean
* ν::T<:Real prior pseudocount for the covariance
"""
params(niw::NormalInverseWishart) = (niw.mu, niw.Lamchol, niw.kappa, niw.nu)

pdf(niw::NormalInverseWishart, x::AbstractVector{T}, Sig::AbstractMatrix{T}) where T<:Real =
        exp(logpdf(niw, x, Sig))

function logpdf(niw::NormalInverseWishart, x::AbstractVector{T}, Sig::AbstractMatrix{T}) where T<:Real
    if !insupport(NormalInverseWishart, x, Sig)
        return -Inf
    else
        p = size(x, 1)

        nu = niw.nu
        kappa = niw.kappa
        mu = niw.mu
        Lamchol = niw.Lamchol
        hnu = 0.5 * nu
        hp = 0.5 * p

        # Normalization
        logp::T = hnu * logdet(Lamchol)
        logp -= hnu * p * log(2.)
        logp -= logmvgamma(p, hnu)
        logp -= hp * (log(2.0*pi) - log(kappa))

        # Inverse-Wishart
        logp -= (hnu + hp + 1.) * logdet(Sig)
        logp -= 0.5 * tr(Sig \ AbstractMatrix(Lamchol))

        # Normal
        z = niw.zeromean ? x : x - mu
        logp -= 0.5 * kappa * invquad(PDMat(Sig), z)

        return logp

    end
end

function rand(niw::NormalInverseWishart)
    Sig = rand(InverseWishart(niw.nu, niw.Lamchol))
    mu = rand(MvNormal(niw.mu, Sig ./ niw.kappa))
    return (mu, Sig)
end
