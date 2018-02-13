using Distributions
using ConjugatePriors

using ConjugatePriors: NormalGamma, NormalInverseGamma, NormalInverseChisq
using ConjugatePriors: posterior, posterior_rand, posterior_mode, posterior_randmodel, fit_map

n = 100
w = rand(100)

@testset "Conjugates for normal distribution" begin

    @testset "Νormal - Νormal (known sigma)" begin

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
        @test mean(p) ≈  (mean(pri) / var(pri) + dot(x, w) / 9.0) / (1.0 / var(pri) + sum(w) / 9.0)
        @test var(p) ≈ inv(1.0 / var(pri) + sum(w) / 9.0)

        r = posterior_mode((pri, 3.0), Normal, x, w)
        @test r ≈ mode(p)

        f = fit_map((pri, 3.0), Normal, x, w)
        @test isa(f, Normal)
        @test f.μ == r
        @test f.σ == 3.0

    end

    @testset "ΙnverseGamma - Νormal (known mu)" begin

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

    end

    @testset "Gamma - Νormal (known mu)" begin

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

    end

    @testset "NormalInverseGamma - Normal" begin

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

        for shape in [0.1, 1., 10.]
            d = NormalInverseGamma(mu0, v0, scale0, shape)
            μ, σ2 = mode(d)
            @test pdf(d, μ, σ2) > pdf(d, μ, σ2 + 0.001)
            @test pdf(d, μ, σ2) > pdf(d, μ, σ2 - 0.001)
        end

    end

    @testset "NormalInverseChisq - Normal" begin

        mu_true = 2.
        sig2_true = 3.
        x = rand(Normal(mu_true, sig2_true), n)

        μ0 = 2.0
        σ20 = 2.0/5.0
        κ0 = 1.0/3.0
        ν0 = 10.0

        pri = NormalInverseChisq(μ0, σ20, κ0, ν0)
        pri2 = NormalInverseGamma(pri)

        @test NormalInverseChisq(pri2) == pri

        @test mode(pri2) == mode(pri)
        @test mean(pri2) == mean(pri)
        @test pdf(pri, mu_true, sig2_true) == pdf(pri2, mu_true, sig2_true)
        
        @test (srand(1); rand(pri)) == (srand(1); rand(pri2))

        # check that updating is consistent between NIχ2 and NIG
        post = posterior(pri, Normal, x)
        post2 = posterior(pri2, Normal, x)
        @test isa(post, NormalInverseChisq)
        @test NormalInverseChisq(post2) == post

        for _ in 1:10
            x = rand(post)
            @test pdf(post, x...) ≈ pdf(post2, x...)
            @test logpdf(post, x...) ≈ logpdf(post2, x...)
        end

        for ν in [0.1, 1., 10.]
            nix = NormalInverseChisq(0., 2., 3., ν)
            μ, σ2 = mode(nix)
            @test pdf(nix, μ, σ2) > pdf(nix, μ, σ2 + 0.001)
            @test pdf(nix, μ, σ2) > pdf(nix, μ, σ2 - 0.001)
        end

    end

    @testset "NormalGamma - Normal" begin

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
    end
    
end
