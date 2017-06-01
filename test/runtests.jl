# tests = ["conjugates",
#          "conjugates_normal",
#          "conjugates_mvnormal"]
tests = [
         "conjugates_mvnormal"]

for t in tests
    fpath = "$t.jl"
    println("running $fpath ...")
    include(fpath)
end
