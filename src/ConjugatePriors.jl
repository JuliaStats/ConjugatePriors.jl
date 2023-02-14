module ConjugatePriors

using Statistics
using LinearAlgebra

using PDMats
using Distributions
using StatsFuns
using SpecialFunctions
lgamma(x) = (logabsgamma(x))[1]

import Statistics: mean
import LinearAlgebra: Cholesky

import PDMats: PDMat

import StatsFuns:
    logmvgamma,
    log2π

import Distributions:
    BernoulliStats,
    BinomData,
    BinomialStats,
    CategoricalData,
    CategoricalStats,
    DirichletCanon,
    ExponentialStats,
    IncompleteFormulation,
    MultinomialStats,
    MvNormalStats,
    MvNormalKnownCovStats,
    NormalStats,
    NormalKnownSigma,
    NormalKnownMu,
    NormalKnownSigmaStats,
    NormalKnownMuStats,

    add!,
    add_categorical_counts!,
    shape,
    scale,
    rate,
    mode,
    rand,
    pdf,
    logpdf,
    params,
    isApproxSymmmetric

export
    # conjugate prior distributions defined here
    NormalInverseChisq,
    NormalGamma,
    NormalInverseGamma,
    NormalWishart,
    NormalInverseWishart,

    # inteface
    posterior,
    posterior_canon,
    posterior_rand,
    posterior_rand!,
    posterior_randmodel,
    posterior_mode,
    fit_map

include("fallbacks.jl")
include("beta_binom.jl")
include("dirichlet_multi.jl")
include("gamma_exp.jl")

include("normalgamma.jl")
include("normalinversegamma.jl")
include("normalinversechisq.jl")
include("normalwishart.jl")
include("normalinversewishart.jl")
include("normal.jl")
include("mvnormal.jl")

end # module
