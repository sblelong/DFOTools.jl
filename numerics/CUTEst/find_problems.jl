import CUTEst

function cutest_filter(pb::Dict, eqs::Int, max_var::Int, min_ineqs::Int, max_ineqs::Int)::Bool
    (pb["constraints"]["equality"] ≠ eqs) && return false
    (pb["variables"]["number"] > max_var) && return false
    nb_ineqs = pb["constraints"]["number"] - eqs
    (nb_ineqs < min_ineqs || nb_ineqs > max_ineqs) && return false
    return true
end

function find_equality_problems(nb_eqs::Int; max_var::Int=50, min_ineqs::Int=0, max_ineqs::Int=10)
    problems = String[]

    for eqs in 1:nb_eqs
        custom_filter(pb) = cutest_filter(pb, eqs, max_var, min_ineqs, max_ineqs)
        pbs = CUTEst.select_sif_problems(; custom_filter=custom_filter)
        append!(problems, pbs)
    end

    return problems
end

function find_data(pb::String)
    model = CUTEstModel(pb)
    model_data = Dict{String,Any}()
    n = model.meta.nvar
    model_data["nvar"] = n
    # Amount of bounded variables
    model_data["nbound"] = n - length(model.meta.ifree)
    # Total amount of constraints
    c = model.meta.ncon
    # Find linear and non-linear equalities and inequalities
    eqs_indices = model.meta.jfix
    ineqs_indices = setdiff(1:c, eqs_indices)
    lin_indices, nlin_indices = model.meta.lin, model.meta.nln
    lineq, nlineq, leq, nleq = 0, 0, 0, 0
    for i in ineqs_indices
        if i in lin_indices
            lineq += 1
        elseif i in nlin_indices
            nlineq += 1
        end
    end
    for i in eqs_indices
        if i in lin_indices
            leq += 1
        elseif i in nlin_indices
            nleq += 1
        end
    end
    model_data["ineqs"] = Dict{String,Int}()
    model_data["ineqs"]["lin"] = lineq
    model_data["ineqs"]["nlin"] = nlineq
    model_data["eqs"] = Dict{String,Int}()
    model_data["eqs"]["lin"] = leq
    model_data["eqs"]["nlin"] = nleq

    # Find the original guess
    x0 = model.meta.x0
    model_data["x0"] = x0
    # Find if the original guess is feasible
    c0 = cons(model, x0)
    lc, uc = model.meta.lcon, model.meta.ucon
    model_data["x0feas"] = all(lc .≤ c0 .≤ uc)
    f0 = obj(model, x0)
    model_data["f0"] = f0
    finalize(model)
    return model_data
end