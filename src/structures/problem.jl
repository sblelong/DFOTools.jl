export Blackbox, nb_ineqs, nb_eqs, add_ineq!, add_eq!, eval_obj, eval_ineq, eval_eq, satisfies_ineq, satisfies_eq

"""
    Blackbox

A structure that represents a (constrained) blackbox optimization problem under the form

``\\min f(x)\\text{ s.t. }c(x)\\leq 0, h(x)=0``

# Attributes
- `obj::Function`: objective function to be minimized.
- `ineqs::Vector{Function}`: vector of inequality constraints (c(x)≤0)
- `eqs::Vector{Function}`: vector of equality constraints (h(x)=0)
- `name::String`: name of the problem.
"""
mutable struct Blackbox
    dim::Int
    obj::Function
    ineqs::Vector{Function}
    eqs::Vector{Function}
    name::String
end

"""
    Blackbox(dim::Int, obj::Function, name::String)

Construct a `dim`-dimensional `Blackbox` problem with objective `obj` and name `name`.
The constraints vectors are initialized empty.
"""
function Blackbox(dim::Int, obj::Function, name::String)
    return Blackbox(dim, obj, Function[], Function[], name)
end

"""
    Blackbox(dim::Int, obj::Function)

Construct a `dim`-dimensional `Blackbox` problem with objective `obj` and an empty name.
The constraints vectors are initialized empty.
"""
function Blackbox(dim::Int, obj::Function)
    return Blackbox(dim, obj, "")
end

"""
    get_dim(bb::Blackbox) -> Int

Return the dimension of the blackbox problem `bb`, i.e. its number of variables.
"""
function get_dim(bb::Blackbox)::Int
    return bb.dim
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