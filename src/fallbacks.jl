# Fallback functions for conjugates

posterior_canon(pri, G::IncompleteFormulation, x) = posterior_canon(pri, suffstats(G, x))
posterior_canon(pri, G::IncompleteFormulation, x, w) = posterior_canon(pri, suffstats(G, x, w))

posterior{P<:Distribution}(pri::P, ss::SufficientStats) = Base.convert(P, posterior_canon(pri, ss))
posterior{P<:Distribution}(pri::P, G::IncompleteFormulation, x) = Base.convert(P, posterior_canon(pri, G, x))
posterior{P<:Distribution}(pri::P, G::IncompleteFormulation, x, w) = Base.convert(P, posterior_canon(pri, G, x, w))

posterior_rand(pri, ss::SufficientStats) = Base.rand(posterior_canon(pri, ss))
posterior_rand(pri, G::IncompleteFormulation, x) = Base.rand(posterior_canon(pri, G, x))
posterior_rand(pri, G::IncompleteFormulation, x, w) = Base.rand(posterior_canon(pri, G, x, w))

posterior_rand!(r::Array, pri, ss::SufficientStats) = Base.rand!(posterior_canon(pri, ss), r)
posterior_rand!(r::Array, pri, G::IncompleteFormulation, x) = Base.rand!(posterior_canon(pri, G, x), r)
posterior_rand!(r::Array, pri, G::IncompleteFormulation, x, w) = Base.rand!(posterior_canon(pri, G, x, w), r)

posterior_mode(pri, ss::SufficientStats) = Distributions.mode(posterior_canon(pri, ss))
posterior_mode(pri, G::IncompleteFormulation, x) = Distributions.mode(posterior_canon(pri, G, x))
posterior_mode(pri, G::IncompleteFormulation, x, w) = Distributions.mode(posterior_canon(pri, G, x, w))

fit_map(pri, G::IncompleteFormulation, x) = complete(G, pri, posterior_mode(pri, G, x))
fit_map(pri, G::IncompleteFormulation, x, w) = complete(G, pri, posterior_mode(pri, G, x, w))

posterior_randmodel(pri, G::IncompleteFormulation, x) = complete(G, pri, posterior_rand(pri, G, x))
posterior_randmodel(pri, G::IncompleteFormulation, x, w) = complete(G, pri, posterior_rand(pri, G, x, w))

