
# Used "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy as
# a reference.  Note that there were some typos in that document so the code
# here may not correspond exactly.

struct NormalWishart{T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: ContinuousMultivariateDistribution
    dim::Int
    zeromean::Bool
    mu::V
    kappa::T
    Tchol::Cholesky{T,M}  # Precision matrix (well, sqrt of one)
    nu::T

    function NormalWishart{T}(mu::AbstractVector{T}, kappa::T,
                              Tchol::Cholesky{T,M}, nu::T) where {T<:Real, M<:AbstractMatrix{T}}
        # Probably should put some error checking in here
        d = length(mu)
        zmean::Bool = true
        for i = 1:d
            if !iszero(mu[i])
                zmean = false
                break
            end
        end
        new{T,typeof(mu),M}(d, zmean, mu, T(kappa), Tchol, T(nu))
    end
end

function NormalWishart(mu::AbstractVector{U}, kappa::Real,
                       Tchol::Cholesky{S}, nu::Real) where {S<:Real, U<:Real}
    T = promote_type(U,S,typeof(kappa), typeof(nu))
    return NormalWishart{T}(convert(AbstractVector{T}, mu),T(kappa),Cholesky{T}(Tchol), T(nu))
end

function NormalWishart(mu::AbstractVector{T}, kappa::T,
                       Tmat::AbstractMatrix{T}, nu::T) where T<:Real
    NormalWishart{T}(mu, kappa, cholesky(Tmat), nu)
end

function insupport(::Type{NormalWishart}, x::Vector{T}, Lam::Matrix{T}) where T<:Real
    return (all(isfinite, x) &&
           size(Lam, 1) == size(Lam, 2) &&
           isApproxSymmmetric(Lam) &&
           size(Lam, 1) == length(x) &&
           isposdef(Lam))
end

pdf(nw::NormalWishart, x::Vector{T}, Lam::Matrix{S}) where T<:Real where S<:Real =
        exp(logpdf(nw, x, Lam))

function logpdf(nw::NormalWishart, x::Vector{T}, Lam::Matrix{T}) where T<:Real
    if !insupport(NormalWishart, x, Lam)
        return -Inf
    else
        p = length(x)

        nu = nw.nu
        kappa = nw.kappa
        mu = nw.mu
        Tchol = nw.Tchol
        hnu = 0.5 * nu
        hp = 0.5 * p

        # Normalization
        logp = hp*(log(kappa) - Float64(log2π))
        logp -= hnu * logdet(Tchol)
        logp -= hnu * p * log(2.)
        logp -= logmvgamma(p, hnu)

        # Wishart (MvNormal contributes 0.5 as well)
        logp += (hnu - hp) * logdet(Lam)
        logp -= 0.5 * tr(Tchol \ Lam)

        # Normal
        z = nw.zeromean ? x : x - mu
        logp -= 0.5 * kappa * dot(z, Lam * z)

        return logp

    end
end

function rand(nw::NormalWishart)
    Lam = rand(Wishart(nw.nu, nw.Tchol))
    Lsym = PDMat(Symmetric(inv(Lam) ./ nw.kappa))
    mu = rand(MvNormal(nw.mu, Lsym))
    return (mu, Lam)
end
