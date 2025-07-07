# This file is designed to be executed in an environment where the package DFOTools is already imported.

using CUTEst
using NLPModels
using LinearAlgebra
import JSON

function parse_log(file::IO, model::CUTEstModel)::Dict{Int,Dict{String,Union{Float64,Vector{Float64}}}}
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

    total_evals_counter = 0

    for (i, line) in enumerate(eachline(file))
        if occursin(r"^\d+", line)
            eval_data = Dict{String,Union{Float64,Vector{Float64}}}()
            numbers = [parse(Float64, m.match) for m in eachmatch(r"-?\d+\.?\d*", line)]
            eval_nb = total_evals_counter + Int(numbers[1])
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
            if m > 0
                g = Float64[]
                for i in ineqs_indices
                    for gi in ineqs_evaluators[i]
                        push!(g, gi(point))
                        eval_data["g"] = g
                    end
                end
            end

            if p > 0
                h = cons(model, point)[eqs_indices]
                eval_data["h"] = h
            end

            parsed_log[eval_nb] = eval_data

        elseif startswith(line, "Blackbox evaluations")
            total_evals_counter += parse(Int, split(line)[end])
        end
    end
    return parsed_log
end

"""
Post-process the .txt data exported in a run.
Takes an instance of an algorithm that corresponds to the name of a folder, reads all the .log files inside the folder.
Creates a unique .json dictionnary where each entry corresponds to a CUTEst problem solved by the algorithm.
dict[problem] contains data corresponding to each evaluation of the blackbox: #eval, objectif, constraint violation.
Warning: only store iterations when there's been an improvement.

Intended JSON format:
{
    "pb": {
        1: {
            "x": Vector{Float64},
            "obj": Float64,
            "constraints": Vector{Float64},
        },
        ...
    },
}
"""
function post_process(dirname::String)
    data = Dict{String,Dict{Int,Dict{String,Union{Float64,Vector{Float64}}}}}()

    # Outer loop: iterate over problems.
    for filename in readdir(dirname)
        if !(endswith(filename, ".log"))
            continue
        end

        pb_name = split(filename, ".log")[1]
        Base.print("Post-processing problem $(pb_name)... ")
        model = CUTEstModel(pb_name)

        open(joinpath(dirname, filename), "r") do logf
            problem_log = parse_log(logf, model)
            data[pb_name] = problem_log
        end
        finalize(model)
        Base.println("✓")
        sleep(2)
    end
    return data
end

"""
Inner loop of the process:
- Construct the blackbox from the CUTEstModel
- Solve the problem with the AL, store the outputs of NOMAD in a file. Warning: this output is the AL value!
- Read the previously stored outputs and compute the true objective and constraints values with the original bb/CUTEstModel (both would work).
- Export the results, prefferably in a readable format, like not a .txt for instance (.json?).
"""
function inner_solve(model::CUTEstModel, γ::Float64, τ::Float64, α::Float64)
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

    # 2. Choose ρ0 as advised in Andreani et al..
    m, p = nb_ineqs(bb), nb_eqs(bb)
    f0 = eval_obj(bb, x0)
    if m > 0
        g0 = [eval_ineq(bb, i, x0) for i in 1:m]
    else
        g0 = [0.0]
    end
    if p > 0
        h0 = [eval_eq(bb, i, x0) for i in 1:p]
    else
        h0 = [0.0]
    end

    ρ0 = max(1.0e-6, min(10.0, 2 * abs(f0) / (norm(h0)^2 + norm(g0)^2)))

    # 3. Solve the problem with an augmented Lagrangian.
    # Max amount of evaluations: 50(n+1)
    # Max amount of iterations: 20(n+1)
    dfauglag(bb; ρ0=ρ0, γ=γ, τ=τ, MAX_EVALS=5000 * (n + 1), α=α, NOMAD_print_point=true, scaling="sgrad")
end

function main(problems::Vector{String}, γ::Float64, τ::Float64, α::Float64)
    for pb in problems
        model = CUTEstModel(pb)
        # Redirect the inner_solve to a text file (yikes tho)
        Base.print("Solving problem $(pb)... ")
        try
            open("$(pb).log", "w") do out
                redirect_stdout(out) do
                    inner_solve(model, γ, τ, α)
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

function json_to_profile(data::Dict{String,Any}, ref_dir::String; εcon::Float64=1e-3)::Dict{String,Vector{Float64}}
    println("HEYO")
    parsed_data = Dict{String,Vector{Float64}}()

    n_problems = length(data)

    # For each problem solved in the data dict
    for (problem, problem_data) in data

        if length(problem_data) == 0
            println("No data on problem $(problem)")
            continue
        end

        # Find the total amount of evals spent on that problem
        open(joinpath(ref_dir, "$(problem).log"), "r") do logf
            global n_evals
            n_evals = 0
            for line in eachline(logf)
                if startswith(line, "Blackbox evaluations")
                    n_evals += parse(Int64, split(line)[end])
                end
            end
        end

        # Initialize an array with typemax(Float64) and that size
        parsed_problem = fill(INFTY, n_evals)

        # Iterate over the evals and store the data only if the problem respects equalities with εcon
        eval_ids = sort(parse.(Int64, collect(keys(problem_data))))
        current_best = typemax(Float64)
        for eval in eval_ids
            eval_data = problem_data[string(eval)]
            if haskey(eval_data, "g")
                g = eval_data["g"]
                if maximum(g) > εcon
                    continue
                end
            end
            if haskey(eval_data, "h")
                h = eval_data["h"]
                if maximum(abs.(h)) > εcon
                    continue
                end
            end

            f = eval_data["f"]
            try
                success = (f < current_best)
                if success
                    parsed_problem[eval:end] .= f
                    current_best = f
                end
            catch e
                println("Issue encountered in problem $(problem) at eval $(eval)")
                rethrow(e)
                continue
            end
        end
        parsed_data[problem] = parsed_problem
    end
    return parsed_data
end

function generate_convergence(data::Vector{Dict{String,Vector{Float64}}}, opts::Dict{String,Any}, dir::String)
    n_algs = length(data)
    problems = collect(keys(data[1]))

    n_problems = length(problems)
    p = Progress(n_problems; barglyphs=BarGlyphs("[=> ]"))

    opts_keys = collect(keys(opts))

    for pb in problems
        # Retrieve problem data for all algorithms
        pb_data = Vector{Vector{Float64}}()
        for alg in 1:n_algs
            try
                data_alg_pb = data[alg][pb]
                if data_alg_pb[end] == INFTY
                    println("Algorithm $(alg) hasn't found feasible points on problem $(pb).")
                    continue
                end
                push!(pb_data, data_alg_pb)
            catch e
                println("! Problem $(pb) not found for algorithm $(alg).")
                continue
            end
        end

        if length(pb_data) == 0
            continue
        end

        # Plot convergence with the known optimal and save it
        known_opt = (pb in opts_keys ? opts[pb] : nothing)

        fig = convergence_plot(pb_data; algs_names=["Conf $i" for i in 1:n_algs], adjust_limit=true, ref_value=known_opt, filename=joinpath(dir, pb * ".pdf"))
        next!(p)
    end
end