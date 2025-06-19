module DFOTools

# Numerical value given to ∞
export INFTY
const INFTY = typemax(Float64)

# Structures
include("structures/problem.jl")

# Algorithms
include("auglag/auglag.jl")

# Profiles
include("profiles/convergence.jl")
include("profiles/performance.jl")

end
