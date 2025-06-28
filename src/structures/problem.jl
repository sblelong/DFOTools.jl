export Blackbox, nb_ineqs, nb_eqs, add_ineq!, add_eq!, eval_obj, eval_ineq, eval_eq, satisfies_ineq, satisfies_eq

"""
    Blackbox

A structure that represents a (constrained) blackbox optimization problem under the form

``\\min f(x)\\text{ s.t. }c(x)\\leq 0, h(x)=0``

# Attributes
- `dim` (`Int`): the number of variables in the problem.
- `obj::Function`: objective function to be minimized.
- `ineqs::Vector{Function}`: vector of inequality constraints (c(x)≤0)
- `eqs::Vector{Function}`: vector of equality constraints (h(x)=0)
- `lbound` (`Vector{Float64}`): the lower bound constraints
- `ubound` (`Vector{Float64}`): the upper bound constraints
- `x0` (`Vector{Float64}`): the initial guess when solving the problem
- `name::String`: name of the problem.
"""
mutable struct Blackbox
    dim::Int
    obj::Function
    x0::Vector{Float64}
    ineqs::Vector{Function}
    eqs::Vector{Function}
    lbound::Vector{Float64}
    ubound::Vector{Float64}
    name::String
end
"""
    Blackbox(dim::Int, obj::Function, x0::Vector{Float64}; lbound::Vector{Float64}, ubound::Vector{Float64}, name::String)

Construct a `dim`-dimensional `Blackbox` problem with objective `obj`, name `name`, initial point `x0` and given bounds.
The constraints vectors and the scaling factors are initialized empty.
"""
function Blackbox(dim::Int, obj::Function, x0::Vector{Float64}; name::String="", lbound::Union{Vector{Float64},Nothing}=nothing, ubound::Union{Vector{Float64},Nothing}=nothing)
    lb = (lbound === nothing ? fill(-INFTY, dim) : lbound)
    ub = (ubound === nothing ? fill(-INFTY, dim) : ubound)
    return Blackbox(dim, obj, x0, Function[], Function[], lb, ub, name)
end

"""
    get_dim(bb::Blackbox) -> Int

Return the dimension of the blackbox problem `bb`, i.e. its number of variables.
"""
function get_dim(bb::Blackbox)::Int
    return bb.dim
end

"""
    get_x0(bb::Blackbox) -> Vector{Float64}

Return the initial guess of the blackbox problem `bb`.
"""
function get_x0(bb::Blackbox)::Vector{Float64}
    return bb.x0
end

"""
    get_lbound(bb::Blackbox) -> Vector{Float64}

Return the lower bounds of the blackbox problem `bb`.
"""
function get_lbound(bb::Blackbox)::Vector{Float64}
    return bb.lbound
end

"""
    get_ubound(bb::Blackbox) -> Vector{Float64}

Return the upper bounds of the blackbox problem `bb`.
"""
function get_ubound(bb::Blackbox)::Vector{Float64}
    return bb.ubound
end

"""
    nb_ineqs(bb::Blackbox) -> Int

Return the number of inequality constraints in the blackbox problem `bb`.
"""
function nb_ineqs(bb::Blackbox)::Int
    return length(bb.ineqs)
end

"""
    nb_eqs(bb::Blackbox) -> Int

Return the number of equality constraints in the blackbox problem `bb`.
"""
function nb_eqs(bb::Blackbox)::Int
    return length(bb.eqs)
end

"""
    add_ineq!(bb::Blackbox, ineq::Function)

Add an inequality constraint `ineq` (a function) to the blackbox problem `bb`.
"""
function add_ineq!(bb::Blackbox, ineq::Function)
    push!(bb.ineqs, ineq)
end

"""
    add_eq!(bb::Blackbox, eq::Function)

Add an equality constraint `eq` (a function) to the blackbox problem `bb`.
"""
function add_eq!(bb::Blackbox, eq::Function)
    push!(bb.eqs, eq)
end

"""
    eval_obj(bb::Blackbox, x::Vector{Float64}) -> Float64

Evaluate the objective function of `bb` at point `x`.
"""
function eval_obj(bb::Blackbox, x::Vector{Float64})::Float64
    return bb.obj(x)
end

"""
    eval_ineq(bb::Blackbox, idx::Int, x::Vector{Float64}) -> Float64

Evaluate the `idx`-th inequality constraint of `bb` at point `x`.
Throws an error if `idx` is out of bounds.
"""
function eval_ineq(bb::Blackbox, idx::Int, x::Vector{Float64})::Float64
    if (idx ≤ 0) || (idx > nb_ineqs(bb))
        error("When evaluating inequality constraint #$(idx) of blackbox $(bb.name). Has $(nb_ineqs(bb)) inequality constraints.")
    else
        return bb.ineqs[idx](x)
    end
end

"""
    eval_eq(bb::Blackbox, idx::Int, x::Vector{Float64}) -> Float64

Evaluate the `idx`-th equality constraint of `bb` at point `x`.
Throws an error if `idx` is out of bounds.
"""
function eval_eq(bb::Blackbox, idx::Int, x::Vector{Float64})::Float64
    if (idx ≤ 0) || (idx > nb_eqs(bb))
        error("When evaluating equality constraint #$(idx) of blackbox $(bb.name). Has $(nb_eqs(bb)) equality constraints.")
    else
        return bb.eqs[idx](x)
    end
end

"""
    satisfies_eq(bb::Blackbox, idx::Int, x::Vector{Float64}; ε::Float64=2e-12) -> Bool

Return `true` if the `idx`-th equality constraint of `bb` is satisfied at `x` within tolerance `ε`.
"""
function satisfies_eq(bb::Blackbox, idx::Int, x::Vector{Float64}; ε::Float64=2e-12)::Bool
    return isapprox(eval_eq(bb, idx, x), 0; atol=ε)
end

"""
    satisfies_ineq(bb::Blackbox, idx::Int, x::Vector{Float64}; ε::Float64=2e-12) -> Bool

Return `true` if the `idx`-th inequality constraint of `bb` is satisfied at `x` within tolerance `ε`.
"""
function satisfies_ineq(bb::Blackbox, idx::Int, x::Vector{Float64}; ε::Float64=2e-12)::Bool
    return eval_ineq(bb, idx, x) ≤ ε
end