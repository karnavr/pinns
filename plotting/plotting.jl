using Plots
using LaTeXStrings
using Plots.PlotMeasures

MARGIN = 5mm

# LOSS  
function plot_losses(losses::Dict, components=("total",))
    
    # Get all available keys 
    available_keys = keys(losses)
    
    # Validate all components exist in losses dictionary
    for comp in components
        if !haskey(losses, comp)
            error("Component '$comp' not found in losses dictionary. Available components: $(join(available_keys, ", "))")
        end
    end
    
    # Get the number of epochs
    n_epochs = length(losses[first(available_keys)])
    epochs = 1:n_epochs
    
    # Create plot
    p = plot(size=(800, 400), legend=:topright, yscale=:log10, margin=MARGIN)

    # Plot each requested component
    for comp in components
        plot!(p, epochs, losses[comp], label=comp, linewidth=2)
    end

    xlabel!("Epoch"); ylabel!("Loss")

    return p
end


# SOLUTION PROFILES

# function plot_solution_over_time(solutions::Matrix{Float64}, t_steps::Vector{Float64})


# SPACETIME


# ANIMATION