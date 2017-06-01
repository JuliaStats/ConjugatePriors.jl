
# Used "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy as
# a reference.  Note that there were some typos in that document so the code
# here may not correspond exactly.

immutable NormalWishart{T} <: ContinuousMultivariateDistribution where T<:Real
    dim::Int
    zeromean::Bool
    mu::Vector{T}
    kappa::T
    Tchol::Cholesky{T,Matrix{T}}  # Precision matrix (well, sqrt of one)
    nu::T
    function NormalWishart{T}(mu::Vector{T}, kappa::T,
                                  Tchol::Cholesky{T,Matrix{T}}, nu::T) where T<:Real
        # Probably should put some error checking in here
        d = length(mu)
        zmean::Bool = true
        for i = 1:d
            if mu[i] != 0.
                zmean = false
                break
            end
        end
        return new(d, zmean, mu, kappa, Tchol, nu)
    end
end

function NormalWishart(mu::Vector{T}, kappa::T,
                       M::Cholesky{T,Matrix{T}}, nu::T) where T<:Real
    NormalWishart{T}(mu, kappa, M, nu)
end

function NormalWishart(mu::Vector{T}, kappa::T,
                       M::Matrix{T}, nu::T) where T<:Real
    NormalWishart{T}(mu, kappa, cholfact(M), nu)
end

function insupport(::Type{NormalWishart}, x::Vector{T}, Lam::Matrix{S}) where T<:Real where S<:Real
    return (all(isfinite(x)) &&
           size(Lam, 1) == size(Lam, 2) &&
           isApproxSymmmetric(Lam) &&
           size(Lam, 1) == length(x) &&
           hasCholesky(Lam))
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
        logp -= lpgamma(p, hnu)

        # Wishart (MvNormal contributes 0.5 as well)
        logp += (hnu - hp) * logdet(Lam)
        logp -= 0.5 * trace(Tchol \ Lam)

        # Normal
        z = nw.zeromean ? x : x - mu
        logp -= 0.5 * kappa * dot(z, Lam * z)

        return logp

    end
end

function rand(nw::NormalWishart)
    Lam = rand(Wishart(nw.nu, nw.Tchol))
    mu = rand(MvNormal(nw.mu, inv(Lam) ./ nw.kappa))
    return (mu, Lam)
end

