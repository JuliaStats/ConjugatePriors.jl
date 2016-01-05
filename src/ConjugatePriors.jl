module ConjugatePriors

using Compat
using PDMats
using Distributions

import Base.LinAlg: Cholesky
import Base: scale, mean

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
    rate,
    mode,
    rand,
    pdf,
    logpdf
    
export
    NormalGamma,
    NormalWishart,
    NormalInverseGamma,
    NormalInverseWishart,
    posterior

include("fallbacks.jl")
include("beta_binom.jl")
include("dirichlet_multi.jl")
include("gamma_exp.jl")

include("normalgamma.jl")
include("normalinversegamma.jl")
include("normalwishart.jl")
include("normalinversewishart.jl")
include("normal.jl")
include("mvnormal.jl")

end # module
