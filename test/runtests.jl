using Base.Test

@testset "Conjugate priors" begin
    tests = ["conjugates",
             "conjugates_normal",
             "conjugates_mvnormal"]

    for t in tests
        fpath = "$t.jl"
        println("running $fpath ...")
        include(fpath)
    end
end
