export dfauglag

using NOMAD
using LinearAlgebra

function auglag_bb_wrapper(bb::Blackbox, x::Vector{Float64}, ρ::Float64, λ::Vector{Float64}, μ::Vector{Float64})
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
- Add the handling of bound constraints through NOMAD.
- Add bounds for the Lagrange mutlipliers.
"""
function dfauglag(bb::Blackbox, x0::Vector{Float64}; λ0::Union{Nothing,Vector{Float64}}=nothing, μ0::Union{Nothing,Vector{Float64}}=nothing, ρ0::Float64=1.0, γ::Float64=1.5, τ::Float64=0.75, δ::Float64=1e-3, εfeas::Float64=1e-3, εfail::Float64=1e-3, MAX_ITERS::Int=50, NOMAD_MAX_EVALS::Int=150, NOMAD_print_point::Bool=false, MAX_SUCC_FAIL::Int=10)
    # Dimension, number of equalities and inequalities.
    n, m, p = get_dim(bb), nb_eqs(bb), nb_ineqs(bb)
    lb, ub = get_lbound(bb), get_ubound(bb)

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
    ρ = ρ0

    # Outer iterations of the augmented Lagrangian method.
    k = 0
    x = x0

    # Current strike of consecutive failures
    successive_fails = 0

    if NOMAD_print_point
        options = NOMAD.NomadOptions(max_bb_eval=NOMAD_MAX_EVALS, display_stats=["BBE", "SOL", "OBJ"])
    else
        options = NOMAD.NomadOptions(max_bb_eval=NOMAD_MAX_EVALS, display_stats=["BBE", "OBJ"])
    end

    previous_gap = 0.0
    while k < MAX_ITERS
        # 1. Solve the augmented Lagrangian subproblem with a DFO method, here MADS.
        L(y) = auglag_bb_wrapper(bb, y, ρ, λ, μ)
        auglag_pb = NomadProblem(n, 1, ["OBJ"], L; lower_bound=lb, upper_bound=ub, options=options)
        result = solve(auglag_pb, x)
        new_x = result.x_best_feas

        # Evaluation of feasibility through a rule that combines primal- and dual-feasibility.
        if p > 0
            g = [eval_ineq(bb, i, new_x) for i in 1:p]
            V = max.(g, -1 / ρ * μ)
            gap = max(maximum(h), maximum(V))
        elseif m > 0
            h = [eval_eq(bb, i, new_x) for i in 1:m]
            gap = maximum(h)
        else
            gap = 0.0
        end

        # 2. Estimate new Lagrange multipliers.
        if m > 0
            λ = λ .+ (ρ * h)
        end
        if p > 0
            μ = max.(zeros(p), μ .+ (ρ * g))
        end

        # 3. Update the penalty parameter according to the feasibility measure.
        if gap > τ * previous_gap
            ρ *= γ
        end

        # If no improvement, stop the procedure. This is not a desirable behavior, but is used here as a patch.
        if maximum(abs.(x .- new_x)) < εfail
            successive_fails += 1
        else
            successive_fails = 0
        end
        if successive_fails ≥ MAX_SUCC_FAIL
            println("## End of optimization (no more improvement): (x,f(x),h(x))=($x, $(eval_obj(bb, x)), $([eval_eq(bb, i, x) for i in 1:m]))")
            break
        end

        # 4. End of current iteration.
        previous_gap = gap
        x = new_x
        k += 1
        println("## End of iteration $k, (x,f(x),h(x))=($x, $(eval_obj(bb, x)), $([eval_eq(bb, i, x) for i in 1:m]))")
    end
end