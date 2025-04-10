using JSON
using LinearAlgebra

"""
    KdVResult

A struct to hold the results of a KdV PINN simulation.

Fields:
- x: Spatial coordinates (vector)
- t: Time coordinates (vector)
- u_pred: Predicted solution matrix (rows = x points, cols = t points)
- u_exact: Exact solution matrix (rows = x points, cols = t points), if available
- losses: Dictionary of loss histories
"""
struct KdVResult
    x::Vector{Float64}
    t::Vector{Float64}
    u_pred::Matrix{Float64}
    u_exact::Union{Matrix{Float64}, Nothing}
    losses::Dict{String, Vector{Float64}}
end

"""
    load_kdv_result(filename)

Load KdV PINN results from a JSON file.

Returns a KdVResult struct with the data in a convenient format.
"""
function load_kdv_result(filename)
    # Parse the JSON file
    data = JSON.parsefile(filename)
    
    # Extract coordinates
    x = data["domain"]["x"]
    t = data["domain"]["t"]
    
    # Get solution matrix 
    # The JSON has u[x_idx][t_idx] format, but we want a matrix where
    # each column is a time point (standard in Julia)
    u_pred_list = data["solution"]["u_pred"]
    u_pred = Matrix{Float64}(undef, length(x), length(t))
    
    for i in 1:length(x)
        for j in 1:length(t)
            u_pred[i,j] = u_pred_list[i][j]
        end
    end
    
    # Extract exact solution if available
    u_exact = nothing
    if haskey(data["solution"], "u_exact")
        u_exact_list = data["solution"]["u_exact"]
        u_exact = Matrix{Float64}(undef, length(x), length(t))
        
        for i in 1:length(x)
            for j in 1:length(t)
                u_exact[i,j] = u_exact_list[i][j]
            end
        end
    end
    
    # Extract losses
    losses = Dict{String, Vector{Float64}}()
    if haskey(data, "losses")
        for (key, values) in data["losses"]
            losses[key] = Float64.(values)
        end
    end
    
    # Create and return the result struct
    return KdVResult(x, t, u_pred, u_exact, losses)
end

# Example: How to use
# result = load_kdv_result("kdv_result.json")
# x_coords = result.x
# solution_at_first_timepoint = result.u_pred[:,1]
# total_loss_history = result.losses["total"]