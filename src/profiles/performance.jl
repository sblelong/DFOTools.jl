export performance_profile

using Plots
using LaTeXStrings

"""
    performance_profile(data ; algs_names, τ, α_max, return_plot)

Compute the needed data and plot a performance profile from the results given by `n_algs` algorithms on a set of `n_pb`` problems, as defined in [Moré, Wild 2009](https://epubs.siam.org/doi/abs/10.1137/080724083?download=true&journalCode=sjope8).

*Note:* In the current implementation, the best objective value is assumed to be the best value found by any algorithm in the batch.

# Arguments
- `data` (`Vector{Dict{Union{Int,String},Vector{Float64}}}`): a vector of `n_algs` dictionaries, each containing the results of one algorithm on the test set. Each dict is organized as follows: `(id of the problem (Int or String) => vector of incumbent solution values at every iteration of the algorithm on the problem)`.
- `algs_names` (`Vector{String}`): a vector of `n_algs` strings with the names of the algorithms.
- τ (`Float64` ∈ (0,1)`): the tolerance factor in the convergence test (see the above-mentioned paper).
- α_max (`Float64`): the largest ratio of evaluations for which to compute the performance profile function.
- return_plot (`Bool`): whether or not to return the plot of the profile.

# Returns
- ρ (`Matrix{Float64}`): a table with `n_algs` rows and 100 columns. Each column contains the value of ρ(α) for a given α.
- fig (`Plots.Plot`): a plot with the performance profile.

# TODO
Allow the user to give known optimal values for some of/all the instances.
"""
function performance_profile(data::Vector{Dict{Union{Int,String},Vector{Float64}}}; algs_names::Union{Vector{String},Nothing}=nothing, τ::Real=1e-2, α_max::Float64=5.0, return_plot::Bool=true)
    n_algs = length(data)
    instances = collect(keys(data[1]))
    n_instances = length(instances)

    # Temporary: to work easily with matrices, need a mapping between instance names and indices in the matrix.
    num_id = Dict(instance => i for (i, instance) in enumerate(instances))

    # Work out algorithm names
    algs_names = (algs_names === nothing ? ["Alg. $i" for i in 1:n_algs] : algs_names)

    # 1. Compute optimal values for all problems
    optimals = compute_optimals(data)

    # 2. Compute the N_a^p values for each algorithm a on each problem p.
    # Assume N_a^p = ∞ if alg a never τ-solved problem p.
    smallest_solved_iterations = fill(typemax(Float64), (n_algs, n_instances))

    for alg in 1:n_algs
        for instance in instances
            if instance in keys(data[alg])
                f0 = data[alg][instance][1]
                instance_opt = optimals[instance]
                a_has_solved_p = data[alg][instance] .≤ instance_opt + τ * (f0 - instance_opt)
                if any(a_has_solved_p .== 1)
                    smallest_solved_iterations[alg, num_id[instance]] = findfirst(a_has_solved_p .== 1)
                end
            end
        end
    end

    # 3. Compute the performance ratios r_a^p.
    # As per described in the book, assume T_{a,p}=0 if N_a^p=∞.
    performance_ratios = fill(typemax(Float64), (n_algs, n_instances))

    best_iteration_solved = fill(typemax(Int), n_instances) # best_iteration_solved[num_id[instance]] = smallest N_a^p for the problem `instance`.

    for instance in instances
        best_iteration_solved[num_id[instance]] = minimum(smallest_solved_iterations[:, num_id[instance]])
    end

    for alg in 1:n_algs
        for instance in instances
            instance_id = num_id[instance]
            if smallest_solved_iterations[alg, instance_id] == typemax(Float64)
                performance_ratios[alg, instance_id] = typemax(Float64)
            else
                performance_ratios[alg, instance_id] = smallest_solved_iterations[alg, instance_id] / best_iteration_solved[instance_id]
            end
        end
    end

    # 4. Compute the values of the performance profile function
    αs = range(1, stop=α_max, length=100)
    ρ = fill(0.0, (n_algs, length(αs)))
    for alg in 1:n_algs
        for (i, α) in enumerate(αs)
            ρ[alg, i] = 1 / n_instances * sum(performance_ratios[alg, :] .≤ α)
        end
    end

    # 5. If desired, plot the performance profile
    if return_plot
        fig = plot(size=(700, 450))
        for alg in 1:n_algs
            plot!(αs, ρ[alg, :], label=algs_names[alg], ylimits=(0, 1), seriestype=:steppost, dpi=700, xtickfontsize=12, ytickfontsize=12)
        end
        xlabel!("Ratio of function evaluations " * L"\alpha")
        ylabel!("Portion of " * L"\tau" * "-solved instances " * L"\rho_a(\alpha)")

        return (ρ, fig)
    end

    return ρ

end