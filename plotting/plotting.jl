using Plots
using LaTeXStrings
using Plots.PlotMeasures

MARGIN = 5mm

# LOSS  
function plot_losses(losses::Dict; components=("total",), optimizer_label = false)
    
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

    yticks!(10.0 .^ (-7:1:0))

    if optimizer_label
        vline!([1000], lw=2, color=:black, label=false)
        annotate!(1100, 1e-2, text("ADAM -> L-BFGS", 10, :black, rotation=90))
    end

    return p
end


# SOLUTION PROFILES

"""
    plot_wave_profiles(result, times; figsize=(800, 300), show_exact=true)

Plot KdV solution profiles at specified times as stacked subplots.

Parameters:
- result: KdVResult struct containing the solution data
- times: Tuple or Vector of time values to plot
- figsize: Tuple of (width, height) for the figure
- show_exact: Whether to show exact solution if available

Returns a plot with stacked subplots showing the wave profile at each time.
"""
function plot_wave_profiles(result, times; figsize=(600, 400), legendpos=:topright, plot_linear_combination=false)
    # Determine plot layout - one subplot per time
    n_plots = length(times)
    
    # Create the figure with appropriate size
    p = plot(layout=(n_plots, 1), size=figsize, legend=legendpos)
    
    for (i, target_time) in enumerate(times)
        # Find the closest time index
        time_idx = argmin(abs.(result.t .- target_time))
        actual_time = result.t[time_idx]
        
        # Plot predicted solution
        plot!(p[i], result.x, result.u_pred[:, time_idx], 
              linewidth=2, label="t = $(round(actual_time, digits=1))")

        if plot_linear_combination
            plot!(p[i], result.x, result.u_lin_comb[:, time_idx], 
                  linewidth=1, color=:black, linestyle=:dash, label=false)
        end
        
        # Add labels only to the bottom subplot
        if i == n_plots
            xlabel!(p[i], "x")
        else
            # Hide x-axis numbers on all plots except the last one
            xformatter = _ -> ""  # Function that returns empty string for any input
            plot!(p[i], xformatter=xformatter)
        end
        
        ylabel!(p[i], "u(x)")
    end
    
    return p
end



# SPACETIME


# ANIMATION