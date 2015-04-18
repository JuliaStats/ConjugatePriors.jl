module ConjugatePriors

using Compat
using PDMats
using Distributions

import Base.LinAlg: Cholesky
import Distributions:
    IncompleteFormulation,
    BinomData,
    CategoricalData,

    BernoulliStats,
    BinomialStats,
    CategoricalStats,
    MultinomialStats,
    ExponentialStats,
    NormalStats,
    NormalKnownSigmaStats,
    NormalKnownMuStats,
    MvNormalStats,
    MvNormalKnownCovStats

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
