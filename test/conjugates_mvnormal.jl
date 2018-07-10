# Conjugate models for multivariate normal

using Distributions
using ConjugatePriors

using ConjugatePriors:
    posterior,
    posterior_randmodel,
    NormalWishart,
    NormalInverseWishart

ConjugatePriors.NormalInverseWishart(nix2::NormalInverseChisq) =
    NormalInverseWishart([nix2.μ], nix2.κ, nix2.ν*reshape([nix2.σ2], 1, 1), nix2.ν)

@testset "Conjugate models for multivariate normal" begin

    @testset "MvNormal -- Normal (known covariance)" begin

        n = 3
        p = 4
        X = reshape(Float64[1:12;], p, n)
        w = rand(n)
        Xw = X * Diagonal(w)

        Sigma = 0.75I + fill(0.25, p, p)
        ss = suffstats(MvNormalKnownCov(Sigma), X)
        ssw = suffstats(MvNormalKnownCov(Sigma), X, w)

        s_t = sum(X, dims=2)
        ws_t = sum(Xw, dims=2)
        tw_t = length(w)
        wtw_t = sum(w)

        @test ss.sx ≈ s_t
        @test ss.tw ≈ tw_t

        @test ssw.sx ≈ ws_t
        @test ssw.tw ≈ wtw_t

        # Posterior
        n = 10
        # n = 100
        mu_true = [2., 3.]
        Sig_true = Matrix(1.0I, 2, 2)
        Sig_true[1,2] = Sig_true[2,1] = 0.25
        mu0 = [2.5, 2.5]
        Sig0 = Matrix(1.0I, 2, 2)
        Sig0[1,2] = Sig0[2,1] = 0.5
        X = rand(MultivariateNormal(mu_true, Sig_true), n)
        pri = MultivariateNormal(mu0, Sig0)

        post = posterior((pri, Sig_true), MvNormal, X)
        @test isa(post, FullNormal)

        @test post.μ ≈ inv(inv(Sig0) + n*inv(Sig_true))*(n*inv(Sig_true)*mean(X,dims=2) + inv(Sig0)*mu0)
        @test post.Σ.mat ≈ inv(inv(Sig0) + n*inv(Sig_true))

        # posterior_sample

        ps = posterior_randmodel((pri, Sig_true), MvNormal, X)
        @test isa(ps, FullNormal)
        @test insupport(ps, ps.μ)
        @test insupport(InverseWishart, ps.Σ.mat)

    end
    
    @testset "NormalInverseWishart - MvNormal" begin

        mu_true = [2., 2.]
        Sig_true = Matrix(1.0I, 2, 2)
        Sig_true[1,2] = Sig_true[2,1] = 0.25

        X = rand(MultivariateNormal(mu_true, Sig_true), n)
        Xbar = mean(X,dims=2)
        Xm = X .- mean(X,dims=2)

        mu0 = [2., 3.]
        kappa0 = 3.
        nu0 = 4.
        T0 = Matrix(1.0I, 2, 2)
        T0[1,2] = T0[2,1] = .5
        pri = NormalInverseWishart(mu0, kappa0, T0, nu0)

        post = posterior(pri, MvNormal, X)

        @test post.mu ≈ (kappa0*mu0 + n*Xbar)./(kappa0 + n)
        @test post.kappa ≈ kappa0 + n
        @test post.nu ≈ nu0 + n
        @test Matrix(post.Lamchol) ≈ T0 + Xm*transpose(Xm) + kappa0*n/(kappa0+n)*(Xbar-mu0)*(Xbar-mu0)'

        ps = posterior_randmodel(pri, MultivariateNormal, X)

        @test isa(ps, MultivariateNormal)
        @test insupport(ps, ps.μ) && insupport(InverseWishart, ps.Σ.mat)

        @testset "Equivalence with Normal-Inverse Chi-squared" begin
            nix2 = NormalInverseChisq(1., 2., 3., 4.)
            niw = NormalInverseWishart(nix2)

            x = rand(Normal(3, 2), 100)

            ss = suffstats(Normal, x)
            ss_mv = suffstats(MvNormal, reshape(x, 1, :))

            post_nix2 = posterior(nix2, ss)
            post_niw = posterior(niw, ss_mv)

            @test all(post_nix2.μ .≈ post_niw.mu)
            @test post_nix2.κ ≈ post_niw.kappa
            @test post_nix2.ν ≈ post_niw.nu
            @test all(post_nix2.σ2 .≈ Matrix(post_niw.Lamchol)[1] ./ post_niw.nu)

            μ, σ2 = rand(post_nix2)
            @test logpdf(post_nix2, μ, σ2) ≈ logpdf(post_niw, [μ], reshape([σ2], 1, 1))
            @test logpdf(nix2, μ, σ2) ≈ logpdf(niw, [μ], reshape([σ2], 1, 1))

        end


    end
    
    @testset "NormalWishart - MvNormal" begin

        mu_true = [2., 2.]
        Lam_true = Matrix(1.0I, 2, 2)
        Lam_true[1,2] = Lam_true[2,1] = 0.25

        X = rand(MvNormal(mu_true, inv(Lam_true)), n)
        Xbar = mean(X,dims=2)
        Xm = X .- Xbar

        mu0 = [2., 3.]
        kappa0 = 3.
        nu0 = 4.
        T0 = Matrix(1.0I, 2, 2)
        T0[1,2] = T0[2,1] = .5
        pri = NormalWishart(mu0, kappa0, T0, nu0)

        post = posterior(pri, MvNormal, X)

        @test post.mu ≈ (kappa0*mu0 + n*Xbar)./(kappa0 + n)
        @test post.kappa ≈ kappa0 + n
        @test post.nu ≈ nu0 + n
        @test Matrix(post.Tchol) ≈ T0 + Xm*transpose(Xm) + kappa0*n/(kappa0+n)*(Xbar-mu0)*(Xbar-mu0)'

        ps = posterior_randmodel(pri, MvNormal, X)

        @test isa(ps, MultivariateNormal)
        @test insupport(ps, ps.μ)
        @test insupport(InverseWishart, ps.Σ.mat)  # InverseWishart on purpose

    end
end
