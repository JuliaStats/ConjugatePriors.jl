__precompile__()

module ConjugatePriors

using PDMats
using Distributions
using StatsFuns

import Base: mean
import Base.LinAlg: Cholesky

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
    isApproxSymmmetric,
    hasCholesky


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
