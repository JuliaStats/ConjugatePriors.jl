# Conjugates for normal distribution

using Base.Test
using Distributions
using ConjugatePriors

import ConjugatePriors: NormalGamma, NormalInverseGamma
import ConjugatePriors: posterior, posterior_rand, posterior_mode, posterior_randmodel, fit_map

n = 100
w = rand(100)

# Νormal - Νormal (known sigma)

pri = Normal(1.0, 5.0)

x = rand(Normal(2.0, 3.0), n)
p = posterior((pri, 3.0), Normal, x)
@test isa(p, Normal)
@test mean(p) ≈ (mean(pri) / var(pri) + sum(x) / 9.0) / (1.0 / var(pri) + n / 9.0)
@test var(p) ≈ inv(1.0 / var(pri) + n / 9.0)

r = posterior_mode((pri, 3.0), Normal, x)
@test r ≈ mode(p)

f = fit_map((pri, 3.0), Normal, x)
@test isa(f, Normal)
@test f.μ == r
@test f.σ == 3.0

p = posterior((pri, 3.0), Normal, x, w)
@test isa(p, Normal)
@test mean(p) ≈ (mean(pri) / var(pri) + dot(x, w) / 9.0) / (1.0 / var(pri) + sum(w) / 9.0)
@test var(p) ≈ inv(1.0 / var(pri) + sum(w) / 9.0)

r = posterior_mode((pri, 3.0), Normal, x, w)
@test r ≈ mode(p)

f = fit_map((pri, 3.0), Normal, x, w)
@test isa(f, Normal)
@test f.μ == r
@test f.σ == 3.0


# ΙnverseGamma - Νormal (known mu)

pri = InverseGamma(1.5, 0.5)

x = rand(Normal(2.0, 3.0), n)
p = posterior((2.0, pri), Normal, x)
@test isa(p, InverseGamma)
@test shape(p) ≈ shape(pri) + n / 2
@test scale(p) ≈ scale(pri) + sum(abs2.(x .- 2.0)) / 2

r = posterior_mode((2.0, pri), Normal, x)
@test r ≈ mode(p)

f = fit_map((2.0, pri), Normal, x)
@test isa(f, Normal)
@test f.μ == 2.0
@test abs2(f.σ) ≈ r

p = posterior((2.0, pri), Normal, x, w)
@test isa(p, InverseGamma)
@test shape(p) ≈ shape(pri) + sum(w) / 2
@test scale(p) ≈ scale(pri) + dot(w, abs2.(x .- 2.0)) / 2

r = posterior_mode((2.0, pri), Normal, x, w)
@test r ≈ mode(p)

f = fit_map((2.0, pri), Normal, x, w)
@test isa(f, Normal)
@test f.μ == 2.0
@test abs2(f.σ) ≈ r


# Gamma - Νormal (known mu)

pri = Gamma(1.5, 2.0)

x = rand(Normal(2.0, 3.0), n)
p = posterior((2.0, pri), Normal, x)
@test isa(p, Gamma)
@test shape(p) ≈ shape(pri) + n / 2
@test scale(p) ≈ scale(pri) + sum(abs2.(x .- 2.0)) / 2

r = posterior_mode((2.0, pri), Normal, x)
@test r ≈ mode(p)

f = fit_map((2.0, pri), Normal, x)
@test isa(f, Normal)
@test f.μ == 2.0
@test abs2(f.σ) ≈ inv(r)

p = posterior((2.0, pri), Normal, x, w)
@test isa(p, Gamma)
@test shape(p) ≈ shape(pri) + sum(w) / 2
@test scale(p) ≈ scale(pri) + dot(w, abs2.(x .- 2.0)) / 2

r = posterior_mode((2.0, pri), Normal, x, w)
@test r ≈ mode(p)

f = fit_map((2.0, pri), Normal, x, w)
@test isa(f, Normal)
@test f.μ == 2.0
@test abs2(f.σ) ≈ inv(r)


# NormalInverseGamma - Normal

mu_true = 2.
sig2_true = 3.
x = rand(Normal(mu_true, sig2_true), n)

mu0 = 2.
v0 = 3.
shape0 = 5.
scale0 = 2.
pri = NormalInverseGamma(mu0, v0, shape0, scale0)

post = posterior(pri, Normal, x)
@test isa(post, NormalInverseGamma)

@test post.mu ≈ (mu0/v0 + n*mean(x))/(1./v0 + n)
@test post.v0 ≈ 1./(1./v0 + n)
@test post.shape ≈ shape0 + 0.5*n
@test post.scale ≈ scale0 + 0.5*(n-1)*var(x) + n./v0./(n + 1./v0)*0.5*(mean(x)-mu0).^2

ps = posterior_randmodel(pri, Normal, x)

@test isa(ps, Normal)
@test insupport(ps,ps.μ) && ps.σ > zero(ps.σ)


# NormalGamma - Normal

mu_true = 2.
tau2_true = 3.
x = rand(Normal(mu_true, 1./tau2_true), n)

mu0 = 2.
nu0 = 3.
shape0 = 5.
rate0 = 2.
pri = NormalGamma(mu0, nu0, shape0, rate0)

post = posterior(pri, Normal, x)
@test isa(post, NormalGamma)

@test post.mu ≈ (nu0*mu0 + n*mean(x))./(nu0 + n)
@test post.nu ≈ nu0 + n
@test post.shape ≈ shape0 + 0.5*n
@test post.rate ≈ rate0 + 0.5*(n-1)*var(x) + n*nu0/(n + nu0)*0.5*(mean(x)-mu0).^2

ps = posterior_randmodel(pri, Normal, x)

@test isa(ps, Normal)
@test insupport(ps, ps.μ) && ps.σ > zero(ps.σ)
