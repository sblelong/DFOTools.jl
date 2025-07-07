using NOMAD

function bb_wrapper(bb::Blackbox, εcon::Float64, x::Vector{Float64}; obj_scaling::Float64=1.0, ineqs_scaling::Union{Vector{Float64},Nothing}=nothing, eqs_scaling::Union{Vector{Float64},Nothing}=nothing)
    n, m, p = get_dim(bb), nb_ineqs(bb), nb_eqs(bb)
    bb_outputs = Float64[]

    if isnothing(ineqs_scaling)
        ineqs_scaling = [1.0 for _ in 1:m]
    end
    if isnothing(eqs_scaling)
        eqs_scaling = [1.0 for _ in 1:p]
    end

    scaled_obj = obj_scaling * eval_obj(bb, x)
    push!(bb_outputs, scaled_obj)

    for i in 1:m
        scaled_ineq = ineqs_scaling[i] * eval_ineq(bb, i, x)
        push!(bb_outputs, scaled_ineq - εcon)
    end
    for i in 1:p
        scaled_eq = eqs_scaling[i] * eval_eq(bb, i, x)
        push!(bb_outputs, abs(scaled_eq) - εcon)
    end

    success = count_eval = true

    return (success, count_eval, bb_outputs)
end

function solve_NOMAD_PB(bb::Blackbox, εcon::Float64, MAX_EVALS::Int; scaling::String="none")
    n, m, p = get_dim(bb), nb_ineqs(bb), nb_eqs(bb)
    lb, ub = get_lbound(bb), get_ubound(bb)
    x0 = bb.x0

    options = NOMAD.NomadOptions(display_stats=["BBE", "SOL", "BBO"], max_bb_eval=MAX_EVALS)

    obj_scaling, ineqs_scaling, eqs_scaling = sgrad_scaling(bb)
    println("Found constant for objective: ", obj_scaling)
    println("Found constants for equalities: ", eqs_scaling)
    println("Found constants for inequalities: ", ineqs_scaling)
    evaluator(x) = bb_wrapper(bb, εcon, x; obj_scaling=obj_scaling, ineqs_scaling=ineqs_scaling, eqs_scaling=eqs_scaling)
    problem = NomadProblem(n, m + p + 1, [["OBJ"]; ["PB" for _ in 1:(m+p)]], evaluator; lower_bound=lb, upper_bound=ub, options=options)
    solve(problem, x0)
end

function NOMAD_PB_wrapper(model::CUTEstModel, εcon::Float64; scaling::String="none")
    # 1. Create the bb
    n = model.meta.nvar # Dimension of the problem
    x0 = model.meta.x0
    lb, ub = model.meta.lvar, model.meta.uvar
    bb = Blackbox(n, x -> obj(model, x), x0; lbound=lb, ubound=ub)
    ncons = model.ncon[] # [] is here because model.ncon is a reference (NLPModels API).
    eqs_indices = get_jfix(model)
    lc, uc = model.meta.lcon, model.meta.ucon
    for i in 1:ncons
        if i in eqs_indices
            add_eq!(bb, x -> cons(model, x)[i])
        else
            # Check whether the inequality is g(x)≥0 or g(x)≤0
            if lc[i] != typemin(Float64) # The constraint is of type g(x)≥0
                add_ineq!(bb, x -> lc[i] - cons(model, x)[i])
            end
            if uc[i] != typemax(Float64) # The constraint is g(x)≤0
                add_ineq!(bb, x -> cons(model, x)[i] - uc[i])
            end
        end
    end
    # Solve with NOMAD and the PB
    solve_NOMAD_PB(bb, εcon, 1000 * (n + 1); scaling=scaling)
end

function mainPB(problems::Vector{String}, εcon::Float64; scaling::String="none")
    for pb in problems
        model = CUTEstModel(pb)
        # Redirect the inner_solve to a text file (yikes tho)
        Base.print("Solving problem $(pb)... ")
        try
            open("$(pb).log", "w") do out
                redirect_stdout(out) do
                    NOMAD_PB_wrapper(model, εcon; scaling=scaling)
                end
            end
        catch e
            println("✗")
            println(e)
            continue
        end
        finalize(model)
        Base.println("✓")
    end
end

function parse_NOMAD_log(file::IO, model::CUTEstModel)::Dict{Int,Dict{String,Union{Float64,Vector{Float64}}}}
    parsed_log = Dict{Int,Dict{String,Union{Float64,Vector{Float64}}}}()
    cons_indices = 1:model.meta.ncon
    eqs_indices = model.meta.jfix
    ineqs_indices = setdiff(cons_indices, eqs_indices)
    m, p = length(ineqs_indices), length(eqs_indices)
    if m > 0
        # Rebuild inequality constraints
        lc, uc = model.meta.lcon, model.meta.ucon
        ineqs_evaluators = Dict{Int,Vector{Function}}()
        for i in ineqs_indices
            ineq_evaluators = Function[]
            if lc[i] != typemin(Float64)
                push!(ineq_evaluators, x -> lc[i] - cons(model, x)[i])
            end
            if uc[i] != typemax(Float64)
                push!(ineq_evaluators, x -> cons(model, x)[i] - uc[i])
            end
            ineqs_evaluators[i] = ineq_evaluators
        end
    end

    for line in eachline(file)
        if occursin(r"^\d+", line)
            eval_data = Dict{String,Union{Float64,Vector{Float64}}}()
            numbers = [parse(Float64, m.match) for m in eachmatch(r"-?\d+\.?\d*", line)]

            if endswith(line, "-inf") || endswith(line, "inf")
                point = numbers[2:end]
                f = typemin(Float64)
            elseif endswith(strip(line), "1e+20")
                point = numbers[2:end-2]
                f = typemax(Float64)
            else
                point = numbers[2:end-1]
                try
                    f = obj(model, point)
                catch e
                    println(point)
                    rethrow(e)
                end
            end

            eval_data["x"] = point
            eval_data["f"] = f


        end
    end
end