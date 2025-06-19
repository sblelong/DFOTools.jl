using Plots
using LaTeXStrings

export convergence_plot

"""
    convergence_plot(data ; algs_names, adjust_limit, ref_value, ref_value_label)

Plot the convergence of a list of algorithms on the same problem, optionally displaying a known reference value for the objective.

# Arguments
- `data` (`Vector{Vector{Float64}}`): a vector of `n_algs` vectors, each containing the best objective value found so far at a given blackbox evaluation number.
- `algs_names` (`Vector{String}`): a vector of `n_algs` strings with the names of the algorithms.
- adjust_limit (`Bool`): if set to true, displayed evaluations will be cut once the final optimal is reached.
- ref_value (`Float64`): an optional reference value of the objective function to be displayed on the plot.
- ref_value_label (`String`): if a reference value is given, its label on the plot.

# Returns
- fig (`Plots.Plot`): a plot showing convergence.
"""
function convergence_plot(data::Vector{Vector{Float64}}; algs_names::Union{Vector{String},Nothing}=nothing, adjust_limit::Bool=false, ref_value::Union{Float64,Nothing}=nothing, ref_value_label::Union{String,LaTeXString,Nothing}=nothing, filename::Union{String,Nothing}=nothing)
    n_algs = length(data)

    if algs_names === nothing
        algs_names = ["Algo $(i)" for i in 1:n_algs]
    end

    fig = plot(size=(700, 450))

    # Find the last index to display for each algorithm. Include adjustement if needed.
    last_indices = Int[]
    for alg in 1:n_algs
        data_alg = data[alg]
        n_evals = length(data_alg)

        last_index = adjust_limit ? (Int(ceil(findfirst(data_alg .== data_alg[end]) + 0.05 * n_evals))) : n_evals
        push!(last_indices, last_index)
    end

    max_display = maximum(last_indices)

    # Retrieve data to be displayed for each algorithm, take the largest upper index.
    for alg in 1:n_algs
        data_alg = data[alg]
        n_evals = length(data[alg])

        first_feasible_index = findfirst(data_alg .≠ INFTY)
        eval_range = first_feasible_index:min(max_display, n_evals)

        plot!(eval_range, data[alg][eval_range], label=algs_names[alg], seriestype=:steppost, dpi=700, xtickfontsize=12, ytickfontsize=12)
    end

    xlims!(0, 1.05 * max_display)
    xlabel!("Number of function evaluations")
    ylabel!("Best objective function value")

    # Add an optional line to show a known reference value for the objective
    if !isnothing(ref_value)
        hline!([ref_value]; linewidth=3, label=ref_value_label, linestyle=:dot, color=:red)
    end

    if !isnothing(filename)
        savefig(fig, filename)
    end

    return fig
end