using Sunny, ITensors, ITensorMPS, GLMakie, LinearAlgebra

# Enum for lattice types
@enum LatticeType begin
    TRIANGULAR
    SQUARE
    CHAIN_1D
    HONEYCOMB
end
      
# Configuration structure to hold all parameters
struct LatticeConfig
    lattice_type::LatticeType
    Lx::Int
    Ly::Int  # Set to 1 for 1D chains
    Lz::Int  # Usually 1 for 2D systems
    a::Float64
    s::Float64
    J1::Float64
    J2::Float64
    J3::Float64  # Additional coupling for more complex lattices
    periodic_bc::Bool  # Periodic boundary conditions
end

struct DMRGConfig
    nsweeps::Int
    maxdim::Vector{Int}
    cutoff::Vector{Float64}
    noise::Tuple{Vararg{Float64}}
end

# Structure to hold calculation results for easy access
struct DMRGResults
    energy::Float64
    psi::MPS
    H::MPO
    sites::Vector
    bond_pairs::Vector
    coupling_groups::Dict
    config::LatticeConfig
    N_basis::Int
    crystal::Any  # Sunny Crystal object
    sys::Any      # Sunny System object
end

# Default configurations for different lattice types
function default_triangular_config()
    return LatticeConfig(TRIANGULAR, 8, 8, 1, 1.0, 1/2, 1.0, 0.0, 0.0, true)
end

function default_square_config()
    return LatticeConfig(SQUARE, 5, 5, 1, 1.0, 1/2, 1.0, 0.0, 0.0, true)
end

function default_chain_config()
    return LatticeConfig(CHAIN_1D, 16, 1, 1, 1.0, 1/2, 1.0, 0.0, 0.0, false)
end

function default_honeycomb_config()
    return LatticeConfig(HONEYCOMB, 6, 6, 1, 1.0, 1/2, 1.0, 0.0, 0.0, true)
end

function default_dmrg_config()
    return DMRGConfig(10, [10, 20, 100, 100, 200], [1E-8], (1E-7, 1E-8, 0.0))
end

"""
Creates appropriate Sunny crystal structure based on lattice type.
"""
function create_crystal(config::LatticeConfig)
    if config.lattice_type == TRIANGULAR
        latvecs = lattice_vectors(config.a, config.a, 1.0, 90, 90, 120)
        return Crystal(latvecs, [[0, 0, 0]])
    elseif config.lattice_type == SQUARE
        latvecs = lattice_vectors(config.a, config.a, 1.0, 90, 90, 90)
        return Crystal(latvecs, [[0, 0, 0]])
    elseif config.lattice_type == CHAIN_1D  
        latvecs = lattice_vectors(config.a, 10*config.a, 10*config.a, 90, 90, 90)
        return Crystal(latvecs, [[0, 0, 0]])
    elseif config.lattice_type == HONEYCOMB
        # Lattice vectors (a = b = 1.0 Å gives unit spacing, but you can scale this)
        a = 2.46  # Typical graphene lattice constant in Ångströms
        latvecs = lattice_vectors(a, a, 10.0, 90, 90, 120)  # c=10.0 to separate layers

        # Space group 191 (P6/mmm) generates the honeycomb positions automatically
        positions = [[1/3, 2/3, 0]]  # Fractional coordinates
        cryst = Crystal(latvecs, positions, 191) #Need only specify one position if using spacegroup
        print(cryst)
        return cryst
    else
        error("Unsupported lattice type: $(config.lattice_type)")
    end
end

"""
Sets up bonds based on lattice type.
"""
function get_lattice_bonds(config::LatticeConfig)
    if config.lattice_type == TRIANGULAR
        nn_bond = Bond(1, 1, [1, 0, 0])
        nnn_bond = Bond(1, 1, [2, 0, 0])
        return nn_bond, nnn_bond
    elseif config.lattice_type == SQUARE
        nn_bond = Bond(1, 1, [1, 0, 0])
        nnn_bond = Bond(1, 1, [1, 1, 0])
        return nn_bond, nnn_bond
    elseif config.lattice_type == CHAIN_1D
        nn_bond = Bond(1, 1, [1, 0, 0])
        nnn_bond = Bond(1, 1, [2, 0, 0])
        return nn_bond, nnn_bond
    elseif config.lattice_type == HONEYCOMB
        # Honeycomb nearest neighbor bonds (A-B sublattice connections)
        nn_bond = Bond(1, 2, [0, 0, 0])    # A to B in same cell
        # Next-nearest neighbor (A-A and B-B connections)
        nnn_bond = Bond(1, 1, [1, 0, 0])
        return nn_bond, nnn_bond
    else
        error("Unsupported lattice type: $(config.lattice_type)")
    end
end

"""
Sets up the Sunny system with exchange interactions based on lattice type.
"""
function setup_sunny_system(crystal, config::LatticeConfig)
    pbc = (true, !config.periodic_bc, true)
    sys = System(crystal, [1 => Moment(; s=config.s, g=2)], :dipole) #if you force space group you have to add other moment

    # Get the appropriate bonds for this lattice type
    nn_bond, nnn_bond = get_lattice_bonds(config)

    
    # Set nearest neighbor exchanges
    set_exchange!(sys, config.J1, nn_bond)
    set_exchange!(sys, config.J2, nnn_bond)

    # Set XXZ exchange
    J = 1.0
    Δ = 1.0  # Tune this for anisotropy (Δ < 1: easy-plane, Δ > 1: easy-axis)
    J_matrix = J * [1.0 0 0; 0 1.0 0; 0 0 Δ]

    set_exchange!(sys, J_matrix, nn_bond) #will currently override nn bond

    # Show crystal structure
    fig = view_crystal(sys; ndims=2, refbonds=20)
    display(fig)

    # Repeat periodically
    if config.lattice_type == CHAIN_1D
        sys = repeat_periodically(sys, (config.Lx, 1, 1))
    else
        sys = repeat_periodically(sys, (config.Lx, config.Ly, config.Lz))
    end

    sys_inhom = to_inhomogeneous(sys)
    remove_periodicity!(sys_inhom, pbc)

    return sys_inhom
end

"""
Maps CartesianIndex to 1D label for ITensor indexing.
"""
function cartind_to_label(cartind::CartesianIndex, dims, N_basis; perm=nothing)
    i, j = cartind[1], cartind[2]
    Lx, Ly = dims[1], dims[2] 
    
    k = cartind[4] # kth atom in basis
    n = (i-1)*Ly*N_basis + (j-1)*N_basis + k

    perm === nothing ? n : perm(n)
end

function get_unique_bonds(sys::System, config::LatticeConfig)
    pbc = (true, !config.periodic_bc, true)
    Sunny.is_homogeneous(sys) && error("Use `to_inhomogeneous` first.")
    ints = Sunny.interactions_inhomog(sys)

    sites = Sunny.eachsite(sys)
    N_basis = length(sys.crystal.positions)
    bond_pairs = []
    
    for (j, int) in enumerate(ints)
        for pc in int.pair
            (; bond, isculled) = pc
            isculled && continue
            siteᵢ = sites[j]
            siteⱼ = Sunny.bonded_site(siteᵢ, bond, sys.dims)
        
            # Get linear labels
            labelᵢ = cartind_to_label(siteᵢ, sys.dims, N_basis)
            labelⱼ = cartind_to_label(siteⱼ, sys.dims, N_basis)

            # Convert scalar couplings to diagonal matrices
            coupling_matrix = if pc.bilin isa AbstractMatrix
                pc.bilin  # Keep as-is if already a matrix
            else
                # Create diagonal matrix from uniform coupling
                Matrix(pc.bilin * I, 3, 3)  # 3x3 identity scaled by coupling -- prevents errors when forming hamiltonian
            end
            
            # Create bond pair with coupling constant
            push!(bond_pairs, (labelᵢ, labelⱼ, coupling_matrix))
        end
    end
    println("$(length(bond_pairs)) unique bonds found.")
    return unique(bond_pairs), N_basis
end

"""
Classify bonds by their coupling strength and organize them.
"""
function organize_bonds_for_itensor(bond_pairs)
    # New version that handles both Float64 and Matrix couplings
    coupling_groups = Dict{Any, Vector{Tuple{Int,Int}}}()  # Can store both matrices and norms

    for (i, j, coupling) in bond_pairs
        # Use Frobenius norm as the grouping key
        coupling_key = coupling[1,1]
        
        if !haskey(coupling_groups, coupling_key)
            coupling_groups[coupling_key] = Vector{Tuple{Int,Int}}()
        end
        push!(coupling_groups[coupling_key], (i, j))
    end

    # Sort by absolute value (for floats) or matrix norm (for matrices)
    sorted_couplings = sort(collect(keys(coupling_groups)), by=x -> x isa AbstractMatrix ? norm(x) : abs(x), rev=true)
    
    println("Found $(length(sorted_couplings)) different coupling strengths:")
    for (idx, coupling) in enumerate(sorted_couplings)
        println("  Group $idx: J = $coupling, $(length(coupling_groups[coupling])) bonds")
    end
    
    return coupling_groups, sorted_couplings
end

"""
Builds the ITensor Hamiltonian from processed bond pairs.
"""
function build_hamiltonian_from_bonds(bond_pairs, config::LatticeConfig, N_basis; conserve_qns=true)
    if config.lattice_type == CHAIN_1D
        N = config.Lx
    else
        N = config.Lx * config.Ly * N_basis
    end
    
    sites = siteinds("S=1/2", N; conserve_qns=conserve_qns)
    
    os = OpSum()
    
    for (i, j, coupling) in bond_pairs
        #Assumes no off diagonal coupling
        J_xx = coupling[1, 1]  # Top-left element (SxSx coupling)
        J_yy = coupling[2, 2]  # Middle element (SySy coupling)
        J_zz = coupling[3, 3]  # Bottom-right element (SzSz coupling)
        J_xy = coupling[1, 2]  # SxSy coupling
        J_yx = coupling[2, 1]  # SySx coupling
        J_xz = coupling[1, 3]  # SxSz coupling
        J_zx = coupling[3, 1]  # SzSx coupling
        J_yz = coupling[2, 3]  # SySz coupling
        J_zy = coupling[3, 2]  # SzSy coupling


        os += (J_xx + J_yy)/4, "S+", i, "S-", j
        os += (J_xx + J_yy)/4, "S-", i, "S+", j
        # SzSz term
        os += J_zz, "Sz", i, "Sz", j
        # Off-diagonal terms - offdiagonal terms will break QN conservation anyway so may as well keep them
        # in terms of Sx and Sy
        if !conserve_qns
            os += J_xy, "S+", i, "S-", j  # J_xy SxSy
            os += J_yx, "S-", i, "S+", j  # J_yx SySx 
            os += J_xz, "S+", i, "Sz", j  # J_xz SxSz
            os += J_zx, "Sz", i, "S+", j  # J_zx SzSx
            os += J_yz, "S-", i, "Sz", j  # J_yz SySz
            os += J_zy, "Sz", i, "S-", j  # J_zy SzSy
        end
    end
    
    H = MPO(os, sites)
    return H, sites
end

"""
Creates initial state for DMRG.
"""
function create_initial_state(sites, config::LatticeConfig, N_basis)
    if config.lattice_type == CHAIN_1D
        N = config.Lx * N_basis
    else
        N = config.Lx * config.Ly * N_basis
    end
    
    if config.lattice_type == TRIANGULAR
        state = [isodd(n) ? "Up" : "Dn" for n=1:N]
    else
        state = [isodd(n) ? "Up" : "Dn" for n=1:N]
    end
    
    return MPS(sites, state)
end

"""
Performs DMRG calculation.
"""
function run_dmrg(H, psi0, dmrg_config::DMRGConfig)
    energy, psi = dmrg(H, psi0; 
                      nsweeps=dmrg_config.nsweeps, 
                      maxdim=dmrg_config.maxdim, 
                      cutoff=dmrg_config.cutoff, 
                      noise=dmrg_config.noise)
    return energy, psi
end

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

# Analysis function to examine bond structure
function analyze_bond_structure(bonds, couplings, config)
    println("\nBond Structure Analysis for $(config.lattice_type):")
    println("Total bonds: $(length(bonds))")
    
    for (coupling, bond_list) in couplings
        println("  Coupling J = $coupling: $(length(bond_list)) bonds")
        if length(bond_list) ≤ 10
            for (i, j) in bond_list[1:min(5, length(bond_list))]
                println("    ($i, $j)")
            end
            if length(bond_list) > 5
                println("    ... and $(length(bond_list) - 5) more")
            end
        end
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

"""
Main function that orchestrates the entire calculation.
"""
function main_calculation(config::LatticeConfig = default_triangular_config(), 
                         dmrg_config::DMRGConfig = default_dmrg_config();
                         show_plots=false, show_crystal=false)
    
    println("START OF CALCULATION ==============================")
    println("Periodic boundary conditions in y: ", config.periodic_bc)
    println("Lattice type: $(config.lattice_type)")
    println("Dimensions: $(config.Lx) × $(config.Ly)")
    
    # 1. Set up the crystal structure
    crystal = create_crystal(config)
    
    # 2. Set up the Sunny system
    sys = setup_sunny_system(crystal, config)
    
    # 3. Extract bonds directly from Sunny system
    bond_pairs, N_basis = get_unique_bonds(sys, config)
    
    # 4. Organize bonds by coupling strength
    coupling_groups, sorted_couplings = organize_bonds_for_itensor(bond_pairs)

    # 6. Create initial state and form hamiltonian
    if config.lattice_type == CHAIN_1D
        linkdims = 10
        H, sites = build_hamiltonian_from_bonds(bond_pairs, config, N_basis;conserve_qns=false)
        #can't conserve qns for random site
        psi0 = random_mps(sites; linkdims)
    else
        H, sites = build_hamiltonian_from_bonds(bond_pairs, config, N_basis; conserve_qns=true)
        psi0 = create_initial_state(sites, config, N_basis)
    end
   
    # 7. Calculate initial energy
    initial_energy = inner(psi0, Apply(H, psi0))
    println("Initial energy = ", initial_energy)
        
    # 8. Run DMRG
    energy, psi = run_dmrg(H, psi0, dmrg_config)
    
    # 9. Report results
    println("\nGround State Energy = ", energy)
    println("Energy per site = ", energy / length(sites))
    println("Using overlap = ", inner(psi, Apply(H, psi)))
    println("Total QN of Ground State = ", totalqn(psi))
    
    # 10. Create results structure
    results = DMRGResults(energy, psi, H, sites, bond_pairs, coupling_groups, 
                         config, N_basis, crystal, sys)
    
    # 11. Show plots if requested
    if show_plots
        plot_lattice(results; show_crystal=show_crystal)
    end
    
    return results
end

"""
# Example usage:
println("=== SUNNY to ITensor ===")
honey_results = main_calculation(default_honeycomb_config();show_crystal=false)
#tri_results = main_calculation(default_triangular_config(); show_crystal=false)

# Analyze the bond structure:
analyze_bond_structure(honey_results.bond_pairs, honey_results.coupling_groups, honey_results.config)
#analyze_bond_structure(tri_results.bond_pairs, tri_results.coupling_groups, tri_results.config)

"""