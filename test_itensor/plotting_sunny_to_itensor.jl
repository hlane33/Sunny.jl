#PLOTTING FUNCTIONS FOR sunny_toITensor - not currently required

"""
Generates site positions for plotting based on lattice type.
CORRECTED for honeycomb lattice.
"""
function get_site_positions(config::LatticeConfig, N_basis)
    if config.lattice_type == CHAIN_1D
        return [(Float64(x), 0.0) for x in 1:config.Lx]
    elseif config.lattice_type == SQUARE
        return [(Float64(x), Float64(y)) for y in 1:config.Ly, x in 1:config.Lx][:]
    elseif config.lattice_type == TRIANGULAR
        return [(x - 0.5*(y>1)*(y-1), y*√3/2) for y in 1:config.Ly, x in 1:config.Lx][:]
    elseif config.lattice_type == HONEYCOMB
       positions = Tuple{Float64, Float64}[]
    
        for y in 1:config.Ly
            for x in 1:config.Lx
                # Original parallelogram coordinates
                px = x - 0.5 * (y > 1) * (y - 1)
                base_x = px * config.a
                base_y = y * config.a * √3 / 2
                
                # Rotate 90 degrees clockwise: (x, y) → (y, -x)
                rotated_x = base_y
                rotated_y = -base_x
                
                # A sublattice atom
                push!(positions, (rotated_x, rotated_y))
                
                # B sublattice atom (apply same rotation to offset)
                offset_x = config.a / 2
                offset_y = -config.a * √3 / 6
                rotated_offset_x = offset_y  # Note the sign change from rotation
                rotated_offset_y = -offset_x
                
                push!(positions, (rotated_x + rotated_offset_x, 
                                rotated_y + rotated_offset_y))
            end
        end
    
        
        return positions
    else
        error("Unsupported lattice type: $(config.lattice_type)")
    end
end
"""
CORRECTED plotting function for honeycomb lattice bonds.
"""
function plot_lattice(results::DMRGResults; show_crystal=false, coupling_threshold=1e-10)
    plot_lattice(results.config, results.N_basis, results.bond_pairs, results.coupling_groups; 
                show_crystal=show_crystal, coupling_threshold=coupling_threshold)
end

function plot_lattice(config::LatticeConfig, N_basis::Int, bond_pairs=nothing, coupling_groups=nothing; 
                     show_crystal=false, coupling_threshold=1e-10)
    
    fig = Figure(resolution=(1200, 800))
    
    # Show crystal structure if requested
    if show_crystal && config.lattice_type != CHAIN_1D
        crystal = create_crystal(config)
        crystal_fig = view_crystal(crystal; ndims=2)
        display(crystal_fig)
    end
    
    # Get site positions
    sites = get_site_positions(config, N_basis)
    x_coords = [p[1] for p in sites]
    y_coords = [p[2] for p in sites]
    
    # Determine coloring based on basis
    colors = if N_basis == 1
        fill(:black, length(sites))
    else
        # For honeycomb: alternate colors for A and B sublattices
        [mod1(i, N_basis) == 1 ? :blue : :red for i in 1:length(sites)]
    end
    
    markersizes = fill(15, length(sites))

    # If we have bond information, create plots
    if bond_pairs !== nothing && coupling_groups !== nothing
        sorted_couplings = sort(collect(keys(coupling_groups)), by=abs, rev=true)
        print(sorted_couplings)
        significant_couplings = filter(J -> abs(J) > coupling_threshold, sorted_couplings)
        
        n_plots = length(significant_couplings) + 1
        n_cols = min(3, n_plots)
        n_rows = ceil(Int, n_plots / n_cols)
        
        # Create individual plots for each coupling strength
        axes = []
        for (idx, coupling) in enumerate(significant_couplings)
            print(coupling, "significant couplin")
            row = ceil(Int, idx / n_cols)
            col = mod1(idx, n_cols)
            
            ax = Axis(fig[row, col], 
                     title="J = $(round(coupling, digits=4))", 
                     aspect=DataAspect())
            push!(axes, ax)
            
            # Plot sites
            scatter!(ax, x_coords, y_coords, color=colors, markersize=markersizes)
            
            # Plot bonds for this coupling strength
            plot_bonds_from_pairs!(ax, bond_pairs, sites, coupling, :blue)
        end
        
        # Create combined plot
        combined_row = n_rows
        combined_col = n_cols
        if length(significant_couplings) % n_cols != 0
            combined_col = (length(significant_couplings) % n_cols) + 1
        else
            combined_row += 1
            combined_col = 1
        end
        
        ax_combined = Axis(fig[combined_row, combined_col], 
                          title="All Interactions", 
                          aspect=DataAspect())
        push!(axes, ax_combined)
        
        # Plot sites on combined plot
        scatter!(ax_combined, x_coords, y_coords, color=colors, markersize=markersizes)
        
        # Plot all bonds with different colors/styles
        colors_bonds = [:blue, :red, :green, :purple, :orange, :brown]
        styles = [:solid, :dash, :dot, :dashdot]
        
        for (idx, coupling) in enumerate(significant_couplings)
            bond_color = colors_bonds[mod1(idx, length(colors_bonds))]
            bond_style = styles[mod1(idx, length(styles))]
            plot_bonds_from_pairs!(ax_combined, bond_pairs, sites, coupling, bond_color, bond_style)
        end
        
        # Add legend to combined plot
        legend_elements = []
        for (idx, coupling) in enumerate(significant_couplings)
            bond_color = colors_bonds[mod1(idx, length(colors_bonds))]
            bond_style = styles[mod1(idx, length(styles))]
            push!(legend_elements, LineElement(color=bond_color, linestyle=bond_style))
        end
        legend_labels = ["J = $(round(J, digits=4))" for J in significant_couplings]
        axislegend(ax_combined, legend_elements, legend_labels, position=:rt)
        
        # Adjust limits for all plots
        for ax in axes
            xlims!(ax, minimum(x_coords)-0.5, maximum(x_coords)+0.5)
            ylims!(ax, minimum(y_coords)-0.5, maximum(y_coords)+0.5)
        end
        
    else
        # Simple single plot if no bond information available
        ax = Axis(fig[1, 1], title="$(config.lattice_type) Lattice", aspect=DataAspect())
        scatter!(ax, x_coords, y_coords, color=colors, markersize=markersizes)
        xlims!(ax, minimum(x_coords)-0.5, maximum(x_coords)+0.5)
        ylims!(ax, minimum(y_coords)-0.5, maximum(y_coords)+0.5)
    end
    
    display(fig)
    return fig
end

"""
CORRECTED helper function to plot bonds from bond_pairs.
Removed the problematic print statement and improved error handling.
"""
function plot_bonds_from_pairs!(ax, bond_pairs, sites, target_coupling, color, style=:solid)
    bonds_plotted = 0
    
    for (i, j, coupling) in bond_pairs
        # Only plot bonds with the target coupling strength
        coupling_val = coupling[1, 1]  # Assuming uniform coupling for simplicity

        if abs(coupling_val - target_coupling) < 1e-12
            # Ensure indices are valid
            if i > 0 && i <= length(sites) && j > 0 && j <= length(sites)
                p1 = sites[i]
                p2 = sites[j]
                
                lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]], 
                      color=color, linestyle=style, linewidth=2)
                bonds_plotted += 1

                # Add annotations for i and j at the midpoint of the bond
                midpoint = ((p1[1] + p2[1])/2, (p1[2] + p2[2] + 0.2)/2)
                text!(ax, "($i,$j)", position=midpoint, 
                      fontsize=8, color=:black, align=(:center, :center))
            else
                println("Warning: Invalid bond indices ($i, $j) for sites of length $(length(sites))")
            end
        end
    end
    
    println("Plotted $bonds_plotted bonds for coupling $target_coupling")
    return bonds_plotted
end