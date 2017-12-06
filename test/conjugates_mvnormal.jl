# Conjugate models for multivariate normal

using Distributions
using ConjugatePriors

using ConjugatePriors: posterior, posterior_randmodel, NormalWishart, NormalInverseWishart

@testset "Conjugate models for multivariate normal" begin

    @testset "MvNormal -- Normal (known covariance)" begin

        n = 3
        p = 4
        X = reshape(Float64[1:12;], p, n)
        w = rand(n)
        Xw = X * diagm(w)

        Sigma = 0.75 * eye(p) + fill(0.25, 4, 4)
        ss = suffstats(MvNormalKnownCov(Sigma), X)
        ssw = suffstats(MvNormalKnownCov(Sigma), X, w)

        s_t = sum(X, 2)
        ws_t = sum(Xw, 2)
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
        Sig_true = eye(2)
        Sig_true[1,2] = Sig_true[2,1] = 0.25
        mu0 = [2.5, 2.5]
        Sig0 = eye(2)
        Sig0[1,2] = Sig0[2,1] = 0.5
        X = rand(MultivariateNormal(mu_true, Sig_true), n)
        pri = MultivariateNormal(mu0, Sig0)

        post = posterior((pri, Sig_true), MvNormal, X)
        @test isa(post, FullNormal)

        @test post.μ ≈ inv(inv(Sig0) + n*inv(Sig_true))*(n*inv(Sig_true)*mean(X,2) + inv(Sig0)*mu0)
        @test post.Σ.mat ≈ inv(inv(Sig0) + n*inv(Sig_true))

        # posterior_sample

        ps = posterior_randmodel((pri, Sig_true), MvNormal, X)
        @test isa(ps, FullNormal)
        @test insupport(ps, ps.μ)
        @test insupport(InverseWishart, ps.Σ.mat)

    end
    
    @testset "NormalInverseWishart - MvNormal" begin

        mu_true = [2., 2.]
        Sig_true = eye(2)
        Sig_true[1,2] = Sig_true[2,1] = 0.25

        X = rand(MultivariateNormal(mu_true, Sig_true), n)
        Xbar = mean(X,2)
        Xm = X .- mean(X,2)

        mu0 = [2., 3.]
        kappa0 = 3.
        nu0 = 4.
        T0 = eye(2)
        T0[1,2] = T0[2,1] = .5
        pri = NormalInverseWishart(mu0, kappa0, T0, nu0)

        post = posterior(pri, MvNormal, X)

        @test post.mu ≈ (kappa0*mu0 + n*Xbar)./(kappa0 + n)
        @test post.kappa ≈ kappa0 + n
        @test post.nu ≈ nu0 + n
        @test (post.Lamchol[:U]'*post.Lamchol[:U]) ≈ T0 + A_mul_Bt(Xm, Xm) + kappa0*n/(kappa0+n)*(Xbar-mu0)*(Xbar-mu0)'

        ps = posterior_randmodel(pri, MultivariateNormal, X)

        @test isa(ps, MultivariateNormal)
        @test insupport(ps, ps.μ) && insupport(InverseWishart, ps.Σ.mat)

    end
    
    @testset "NormalWishart - MvNormal" begin

        mu_true = [2., 2.]
        Lam_true = eye(2)
        Lam_true[1,2] = Lam_true[2,1] = 0.25

        X = rand(MvNormal(mu_true, inv(Lam_true)), n)
        Xbar = mean(X,2)
        Xm = X .- Xbar

        mu0 = [2., 3.]
        kappa0 = 3.
        nu0 = 4.
        T0 = eye(2)
        T0[1,2] = T0[2,1] = .5
        pri = NormalWishart(mu0, kappa0, T0, nu0)

        post = posterior(pri, MvNormal, X)

        @test post.mu ≈ (kappa0*mu0 + n*Xbar)./(kappa0 + n)
        @test post.kappa ≈ kappa0 + n
        @test post.nu ≈ nu0 + n
        @test (post.Tchol[:U]'*post.Tchol[:U]) ≈ T0 + A_mul_Bt(Xm, Xm) + kappa0*n/(kappa0+n)*(Xbar-mu0)*(Xbar-mu0)'

        ps = posterior_randmodel(pri, MvNormal, X)

        @test isa(ps, MultivariateNormal)
        @test insupport(ps, ps.μ)
        @test insupport(InverseWishart, ps.Σ.mat)  # InverseWishart on purpose

    end
end
