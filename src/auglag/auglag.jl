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

# Arguments
- `bb::Blackbox`: the blackbox to be optimized. Should take `x::Vector{Float64}` as an input and return an output of the form `Tuple{Float64,Vector{Float64},Vector{Float64}}` where the first number is the objective value and the vectors of inequality and equality constraints, in this order.
"""
function dfauglag(bb::Blackbox, x0::Vector{Float64}; λ0::Union{Nothing,Vector{Float64}}=nothing, μ0::Union{Nothing,Vector{Float64}}=nothing, γ::Float64=0.5, MAX_ITERS::Int=100)
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
    options = NOMAD.NomadOptions(max_bb_eval=10)
    while k < MAX_ITERS
        # 1. Solve the augmented Lagrangian subproblem with a DFO method, here MADS.
        L(y) = auglag_bb_wrapper(bb, y; ρ, λ, μ)
        auglag_pb = NomadProblem(n, 1, ["OBJ"], L; options=options)
        result = solve(auglag_pb, x)
        x = result.x_best_feas

        # 2. Estimate new Lagrange multipliers.
        h = [eval_eq(bb, i, x) for i in 1:p]
        λ = λ .+ (ρ * h)
        g = [eval_ineq(bb, i, x) for i in 1:m]
        V = max.(g, -1 / ρ * μ)
        μ = max.(zeros(p), μ .+ (ρ * g))

        # 3. Update the penalty parameter.
        if k > 0
            if max(maximum(h), maximum(V)) > τ * max(maximum(hprev), maximum(Vprev)) # TODO: save h and V from previous iteration every time.
                ρ *= γ
            end
        end

        k += 1
    end
end