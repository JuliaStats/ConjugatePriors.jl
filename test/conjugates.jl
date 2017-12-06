using Distributions
using ConjugatePriors

import ConjugatePriors: posterior, posterior_rand, posterior_mode, posterior_randmodel, fit_map

n = 100
w = rand(100)

# auxiliary tools

function ccount(K, x)
	r = zeros(K)
	for i = 1:length(x)
		r[x[i]] += 1.0
	end
	r
end

function ccount(K, x, w)
	r = zeros(K)
	for i = 1:length(x)
		r[x[i]] += w[i]
	end
	r
end


@testset "Beta - Bernoulli" begin

    pri = Beta(1.0, 2.0)

    x = rand(Bernoulli(0.3), n)
    p = posterior(pri, Bernoulli, x)
    @test isa(p, Beta)
    @test p.α ≈ pri.α + sum(x)
    @test p.β ≈ pri.β + (n - sum(x))

    f = fit_map(pri, Bernoulli, x)
    @test isa(f, Bernoulli)
    @test succprob(f) ≈ mode(p)

    p = posterior(pri, Bernoulli, x, w)
    @test isa(p, Beta)
    @test p.α ≈ pri.α + sum(x .* w)
    @test p.β ≈ pri.β + (sum(w) - sum(x .* w))

    f = fit_map(pri, Bernoulli, x, w)
    @test isa(f, Bernoulli)
    @test succprob(f) ≈ mode(p)


    @testset "posterior_rand & posterior_randmodel" begin

        pri = Beta(1.0, 2.0)
        x = rand(Bernoulli(0.3), n)
        post = posterior(pri, Bernoulli, x)

        pv = posterior_rand(pri, Bernoulli, x)
        @test isa(pv, Float64)
        @test 0. <= pv <= 1.

        pv = posterior_rand(pri, Bernoulli, x, w)
        @test isa(pv, Float64)
        @test 0. <= pv <= 1.

        pm = posterior_randmodel(pri, Bernoulli, x)
        @test isa(pm, Bernoulli)
        @test 0. <= succprob(pm) <= 1.

        pm = posterior_randmodel(pri, Bernoulli, x, w)
        @test isa(pm, Bernoulli)
        @test 0. <= succprob(pm) <= 1.
    end
    
end

@testset "Beta - Binomial" begin

    pri = Beta(1.0, 2.0)

    x = rand(Binomial(10, 0.3), n)
    p = posterior(pri, Binomial, (10, x))
    @test isa(p, Beta)
    @test p.α ≈ pri.α + sum(x)
    @test p.β ≈ pri.β + (10n - sum(x))

    f = fit_map(pri, Binomial, (10, x))
    @test isa(f, Binomial)
    @test ntrials(f) == 10
    @test succprob(f) ≈ mode(p)

    p = posterior(pri, Binomial, (10, x), w)
    @test isa(p, Beta)
    @test p.α ≈ pri.α + sum(x .* w)
    @test p.β ≈ pri.β + (10 * sum(w) - sum(x .* w))

    f = fit_map(pri, Binomial, (10, x), w)
    @test isa(f, Binomial)
    @test ntrials(f) == 10
    @test succprob(f) ≈ mode(p)

end

@testset "Dirichlet - Categorical" begin

    pri = Dirichlet([1., 2., 3.])

    x = rand(Categorical([0.2, 0.3, 0.5]), n)
    p = posterior(pri, Categorical, x)
    @test isa(p, Dirichlet)
    @test p.alpha ≈ pri.alpha + ccount(3, x)

    f = fit_map(pri, Categorical, x)
    @test isa(f, Categorical)
    @test probs(f) ≈ mode(p)

    p = posterior(pri, Categorical, x, w)
    @test isa(p, Dirichlet)
    @test p.alpha ≈ pri.alpha + ccount(3, x, w)

    f = fit_map(pri, Categorical, x, w)
    @test isa(f, Categorical)
    @test probs(f) ≈ mode(p)

end

@testset "Dirichlet - Multinomial" begin

    pri = Dirichlet([1., 2., 3.])

    x = rand(Multinomial(100, [0.2, 0.3, 0.5]), 1)
    p = posterior(pri, Multinomial, x)
    @test isa(p, Dirichlet)
    @test p.alpha ≈ pri.alpha + x

    r = posterior_mode(pri, Multinomial, x)
    @test r ≈ mode(p)

    x = rand(Multinomial(10, [0.2, 0.3, 0.5]), n)
    p = posterior(pri, Multinomial, x)
    @test isa(p, Dirichlet)
    @test p.alpha ≈ pri.alpha + vec(sum(x, 2))

    r = posterior_mode(pri, Multinomial, x)
    @test r ≈ mode(p)

    p = posterior(pri, Multinomial, x, w)
    @test isa(p, Dirichlet)
    @test p.alpha ≈ pri.alpha + vec(x * w)

    r = posterior_mode(pri, Multinomial, x, w)
    @test r ≈ mode(p)

end

@testset "Gamma - Exponential" begin

    pri = Gamma(1.5, 2.0)

    x = rand(Exponential(2.0), n)
    p = posterior(pri, Exponential, x)
    @test isa(p, Gamma)
    @test shape(p) ≈ shape(pri) + n
    @test rate(p) ≈ rate(pri) + sum(x)

    f = fit_map(pri, Exponential, x)
    @test isa(f, Exponential)
    @test rate(f) ≈ mode(p)

    p = posterior(pri, Exponential, x, w)
    @test isa(p, Gamma)
    @test shape(p) ≈ shape(pri) + sum(w)
    @test rate(p) ≈ rate(pri) + sum(x .* w)

    f = fit_map(pri, Exponential, x, w)
    @test isa(f, Exponential)
    @test rate(f) ≈ mode(p)

end
