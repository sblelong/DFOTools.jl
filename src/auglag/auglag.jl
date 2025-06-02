export dfauglag

using NOMAD

function auglag_bb_wrapper(bb::Blackbox, x::Vector{Float64}; ρ::Float64, λ::Vector{Float64}, μ::Vector{Float64})
    n, m, p = get_dim(bb), nb_eqs(bb), nb_ineqs(bb)

    L = eval_obj(bb, x) + 1 / 2 * ρ * (sum([(eval_eq(bb, i, x) + λ[i] / ρ)^2 for i in 1:m]) + sum([(max(0, eval_ineq(bb, i, x) + μ[i] / ρ))^2 for i in 1:p]))
    bb_outputs = [L]

    success = count_eval = true
    return (success, count_eval, bb_outputs)
end

"""
    dfauglag()

- Draft version.
- Solve a constrained problem with an augmented Lagrangian method, whose subproblem is solved by `NOMAD.jl`.
- Derivative-free version of Andreani et al.'s 2007 AUGLAG, as proposed by Diniz-Ehrhardt et al. in 2011.

# Arguments
- `bb::Blackbox`: the blackbox to be optimized. Should take `x::Vector{Float64}` as an input and return an output of the form `Tuple{Float64,Vector{Float64},Vector{Float64}}` where the first number is the objective value and the vectors of inequality and equality constraints, in this order.

# TODOS
- Handle the types of problems (equalities, inequalities, unconstrained) in a cleaner way.
"""
function dfauglag(bb::Blackbox, x0::Vector{Float64}; λ0::Union{Nothing,Vector{Float64}}=nothing, μ0::Union{Nothing,Vector{Float64}}=nothing, γ::Float64=1.5, τ::Float64=0.75, MAX_ITERS::Int=100)
    # Dimension, number of equalities and inequalities.
    n, m, p = get_dim(bb), nb_eqs(bb), nb_ineqs(bb)

    # Initialization of the Lagrange multipliers (default is all zeros).
    # Lagrange multipliers for equalities
    if λ0 === nothing
        λ = zeros(m)
    else
        λ = λ0
    end

    # Lagrange multipliers for inequalities
    if μ0 === nothing
        μ = zeros(p)
    else
        μ = μ0
    end

    # Penalty parameter
    ρ = 1.0

    # Outer iterations of the augmented Lagrangian method.
    k = 0
    x = x0
    options = NOMAD.NomadOptions(max_bb_eval=50)
    previous_gap = 0.0
    while k < MAX_ITERS
        # 1. Solve the augmented Lagrangian subproblem with a DFO method, here MADS.
        L(y) = auglag_bb_wrapper(bb, y; ρ, λ, μ)
        auglag_pb = NomadProblem(n, 1, ["OBJ"], L; options=options)
        result = solve(auglag_pb, x)
        x = result.x_best_feas

        # 2. Estimate new Lagrange multipliers.
        if nb_eqs(bb) > 0
            h = [eval_eq(bb, i, x) for i in 1:m]
            λ = λ .+ (ρ * h)
        end
        if nb_ineqs(bb) > 0
            g = [eval_ineq(bb, i, x) for i in 1:p]
            V = max.(g, -1 / ρ * μ)
            μ = max.(zeros(p), μ .+ (ρ * g))
        end

        # 3. Update the penalty parameter.
        # This rule uses a criterion that measures a combination of feasibility and dual-complementarity.
        if nb_ineqs(bb) > 0
            gap = max(maximum(h), maximum(V))
        else
            gap = maximum(h)
        end

        if gap > τ * previous_gap
            ρ *= γ
        end
        previous_gap = gap

        k += 1
        println(x)
    end
end