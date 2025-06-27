export lhs

using Random
using ResumableFunctions

"""
    lhs(n, lb, ub, p)

Perform a latin hypercube sampling with given bounds, and return `p` sampled points. It is assumed that bounds are well-defined.
"""
@resumable function lhs(n::Int, lb::Vector{Float64}, ub::Vector{Float64}, p::Int; seed::Union{Int,Nothing})::Vector{Vector{Float64}}
    # If one of the variables is not bounded, throw error
    if (-INFTY in lb) || (INFTY in ub)
        throw(ArgumentError("Cannot perform LHS: one of the variables is not bounded."))
    end

    if seed
        Random.seed!(seed)
    end

    Π = [randperm(p) for _ in 1:n]

    for i in 1:p
        r = rand(n)
        x = lb .+ ((Π[i, :] .- r) ./ p) .* (ub .- lb)
        @yield x
    end
end