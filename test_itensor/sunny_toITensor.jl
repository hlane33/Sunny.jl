using Sunny, ITensors, ITensorMPS, GLMakie

# Enum for lattice types
@enum LatticeType begin
    TRIANGULAR
    SQUARE
    CHAIN_1D
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

function default_dmrg_config()
    return DMRGConfig(10, [10, 20, 100, 100, 200], [1E-8], (1E-7, 1E-8, 0.0))
end

"""
Maps a 2D lattice to a 1D index using snake ordering for 2D or simple linear for 1D.
Returns the site index for given x and y coordinates.
    NOT CURRENTLY CALLED
"""
function get_site_index(x::Int, y::Int, config::LatticeConfig)
    if config.lattice_type == CHAIN_1D
        return x  # Simple linear indexing for 1D
    elseif config.lattice_type == TRIANGULAR
        # Snake ordering for triangular lattice
        if y % 2 == 1  # Odd rows: left-to-right
            return (y - 1) * config.Lx + x
        else            # Even rows: right-to-left
            return (y - 1) * config.Lx + (config.Lx - x + 1)
        end
    else  # SQUARE or other 2D lattices
        # Standard row-major ordering
        return (y - 1) * config.Lx + x
    end
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
    else  # CHAIN_1D
        latvecs = lattice_vectors(config.a, 10*config.a, 10*config.a, 90, 90, 90)
        return Crystal(latvecs, [[0, 0, 0]])
    end
end

"""
Sets up bonds based on lattice type.
Doesnt currently generalise to multi atom basis
Just an object of bonds for different lattice types
"""
function get_lattice_bonds(config::LatticeConfig)
    if config.lattice_type == TRIANGULAR
        # Triangular lattice bonds
        nn_bonds = [Bond(1, 1, [1, 0, 0]), Bond(1, 1, [0, 1, 0]), Bond(1, 1, [-1, 1, 0])]
        nnn_bonds = [Bond(1, 1, [2, 0, 0]), Bond(1, 1, [1, 1, 0]), Bond(1, 1, [-1, 2, 0])]
        return nn_bonds, nnn_bonds
    elseif config.lattice_type == SQUARE
        # Square lattice bonds
        nn_bonds = [Bond(1, 1, [1, 0, 0]), Bond(1, 1, [0, 1, 0])]
        nnn_bonds = [Bond(1, 1, [1, 1, 0]), Bond(1, 1, [1, -1, 0])]  # Diagonal
        return nn_bonds, nnn_bonds
    else  # CHAIN_1D
        # 1D chain bonds
        nn_bonds = [Bond(1, 1, [1, 0, 0])]
        nnn_bonds = [Bond(1, 1, [2, 0, 0])]  # Next-nearest neighbor
        return nn_bonds, nnn_bonds
    end
end

"""
Sets up the Sunny system with exchange interactions based on lattice type.

Don't like this: shouldnt need to for loop over the bonds
"""
function setup_sunny_system(crystal, config::LatticeConfig)
    pbc = (true,!config.periodic_bc,true)
    sys = System(crystal, [1 => Moment(; s=config.s, g=2)], :dipole)

    
    # Get the appropriate bonds for this lattice type
    nn_bonds, nnn_bonds = get_lattice_bonds(config)
    
    # Set nearest neighbor exchanges
    set_exchange!(sys, config.J1, nn_bonds[1])

    
    # Set next-nearest neighbor exchanges
    set_exchange!(sys, config.J2, nnn_bonds[1])

    
    # Repeat periodically
    if config.lattice_type == CHAIN_1D
        sys = repeat_periodically(sys, (config.Lx, 1, 1))
    else
        sys = repeat_periodically(sys, (config.Lx, config.Ly, config.Lz))
    end

    sys_inhom = to_inhomogeneous(sys)
    #true means remove periodicity in that direction
    remove_periodicity!(sys_inhom,pbc)

    return sys_inhom
end
"""
Maps CartesianIndex to 1D label with optional permutation for different ordering schemes.
This creates the standard labelling and will eventually permute according to some rule.
"""
function cartind_to_label(cartind::CartesianIndex, dims; perm = nothing)
    # perm will be some rule to move from conventional labelling i.e.
    # 1 to Lx*Ly where we go 1→Lx first i.e. (2,1) → 2 to the scheme
    # of choice e.g. snake
    x, y = cartind[1], cartind[2]  # Now x=column, y=row
    Lx, Ly = dims[1], dims[2]
    n = (x-1)*Ly + y  # Match square_lattice's formula
    
    # Apply permutation if needed
    perm === nothing ? n : perm(n)
    
    # Apply permutation if specified
    if perm !== nothing
        # Apply custom permutation logic here
        # For now, return standard indexing
        return n
    else
        return n
    end
end


function get_unique_bonds(sys::System, config::LatticeConfig)
    pbc = (true,!config.periodic_bc,true)
    Sunny.is_homogeneous(sys) && error("Use `to_inhomogeneous` first.")
    ints = Sunny.interactions_inhomog(sys)
    sites = Sunny.eachsite(sys)
    bond_pairs = []
    
    for (j, int) in enumerate(ints)
        for pc in int.pair
            (; bond, isculled) = pc
            isculled && continue
            
            siteᵢ = sites[j]
            siteⱼ = Sunny.bonded_site(siteᵢ, bond, sys.dims)
            
            push!(bond_pairs, (
                cartind_to_label(siteᵢ, sys.dims),
                cartind_to_label(siteⱼ, sys.dims),
                pc.bilin
            ))
        end
    end
    print(length(unique(bond_pairs)), " unique bonds found.\n")
    # Remove duplicates by converting to a set
    print(bond_pairs)
    return unique(bond_pairs)
    
end

"""
Classify bonds by their coupling strength and organize them.
Returns organized bond information with coupling constants.

This function is not used in Itensor calculation but is useful for analysis.

"""
function organize_bonds_for_itensor(bond_pairs)
    # Group bonds by coupling strength
    coupling_groups = Dict{Float64, Vector{Tuple{Int,Int}}}()
    
    for (i, j, coupling) in bond_pairs
        if !haskey(coupling_groups, coupling)
            coupling_groups[coupling] = Vector{Tuple{Int,Int}}()
        end
        push!(coupling_groups[coupling], (i, j))
    end
    
    # Sort coupling strengths to identify primary interactions
    sorted_couplings = sort(collect(keys(coupling_groups)), by=abs, rev=true)
    
    println("Found $(length(sorted_couplings)) different coupling strengths:")
    for (idx, coupling) in enumerate(sorted_couplings)
        println("  Group $idx: J = $coupling, $(length(coupling_groups[coupling])) bonds")
    end
    
    return coupling_groups, sorted_couplings
end


"""
Builds the ITensor Hamiltonian from processed bond pairs.
Automatically handles multiple coupling strengths.
"""
function build_hamiltonian_from_bonds(bond_pairs, config::LatticeConfig)
    if config.lattice_type == CHAIN_1D
        N = config.Lx
    else
        N = config.Lx * config.Ly
    end
    
    sites = siteinds("S=1/2", N; conserve_qns=false)
    
    os = OpSum()
    
    # Add all bond terms (works for s dot s terms)
    for (i, j, coupling) in bond_pairs
        # Heisenberg interaction terms
        os += coupling * 1.0, "Sz", i, "Sz", j
        os += coupling * 0.5, "S+", i, "S-", j
        os += coupling * 0.5, "S-", i, "S+", j
    end
    
    H = MPO(os, sites)
    return H, sites
end

"""
Creates initial state for DMRG (can be customized per lattice type).
"""
function create_initial_state(sites, config::LatticeConfig)
    if config.lattice_type == CHAIN_1D
        N = config.Lx
    else
        N = config.Lx * config.Ly
    end
    
    # Different initial states for different lattices
    if config.lattice_type == TRIANGULAR
        # For triangular lattice, might want a more complex initial state
        state = [isodd(n) ? "Up" : "Dn" for n=1:N]
    else
        #might want some 'if random state'
        # Standard Neel state for square and 1D
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
"""
function get_site_positions(config::LatticeConfig)
    if config.lattice_type == CHAIN_1D
        return [(Float64(x), 0.0) for x in 1:config.Lx]
    elseif config.lattice_type == SQUARE
        return [(Float64(x), Float64(y)) for y in 1:config.Ly, x in 1:config.Lx][:]
    else  # TRIANGULAR
        return [(x + 0.5*(y%2), y*√3/2) for y in 1:config.Ly, x in 1:config.Lx][:]
    end
end


# Analysis function to examine bond structure
function analyze_bond_structure(bonds, couplings, config)
    println("\nBond Structure Analysis for $(config.lattice_type):")
    println("Total bonds: $(length(bonds))")
    
    for (coupling, bond_list) in couplings
        println("  Coupling J = $coupling: $(length(bond_list)) bonds")
        if length(bond_list) ≤ 10  # Show first few bonds for small lists
            for (i, j) in bond_list[1:min(5, length(bond_list))]
                println("    ($i, $j)")
            end
            if length(bond_list) > 5
                println("    ... and $(length(bond_list) - 5) more")
            end
        end
    end
end

function plot_lattice(config::LatticeConfig, nn_bonds, nnn_bonds)
    fig = Figure(resolution=(800, 600))
    
    if config.lattice_type == CHAIN_1D
        # Simple 1D plot
        ax = Axis(fig[1, 1], title="1D Chain", aspect=DataAspect())
        sites = get_site_positions(config)
        x_coords = [p[1] for p in sites]
        y_coords = [p[2] for p in sites]
        
        scatter!(ax, x_coords, y_coords, color=:black, markersize=15)
        
        # Draw bonds
        for i in 1:length(sites)-1
            lines!(ax, [sites[i][1], sites[i+1][1]], [sites[i][2], sites[i+1][2]], 
                  color=:blue, linewidth=3)
        end
        
    else
        # 2D lattice plots
        ax1 = Axis(fig[1, 1], title="Nearest Neighbors (NN)", aspect=DataAspect())
        ax2 = Axis(fig[1, 2], title="Next-Nearest Neighbors (NNN)", aspect=DataAspect())
        ax3 = Axis(fig[2, 1:2], title="Full $(config.lattice_type) Lattice", aspect=DataAspect())

        sites = get_site_positions(config)
        x_coords = [p[1] for p in sites]
        y_coords = [p[2] for p in sites]

        # Plot sites
        for ax in [ax1, ax2, ax3]
            scatter!(ax, x_coords, y_coords, color=:black, markersize=10)
        end

        # Helper function to plot bonds
        function plot_bonds!(ax, bonds, color, style=:solid)
            for bond in bonds
                dx, dy, _ = bond.n
                for y in 1:config.Ly, x in 1:config.Lx
                    x2 = config.periodic_bc ? mod1(x + dx, config.Lx) : x + dx
                    y2 = config.periodic_bc ? mod1(y + dy, config.Ly) : y + dy
                    
                    if (!config.periodic_bc && (x2 < 1 || x2 > config.Lx || y2 < 1 || y2 > config.Ly))
                        continue
                    end
                    
                    i = LinearIndices((config.Lx, config.Ly))[x, y]
                    j = LinearIndices((config.Lx, config.Ly))[mod1(x2, config.Lx), mod1(y2, config.Ly)]
                    
                    if i < j
                        p1 = sites[i]
                        p2 = sites[j]
                        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]], 
                              color=color, linestyle=style, linewidth=2)
                    end
                end
            end
        end

        # Plot bonds
        plot_bonds!(ax1, nn_bonds, :blue)
        plot_bonds!(ax3, nn_bonds, :blue)
        plot_bonds!(ax2, nnn_bonds, :red, :dash)
        plot_bonds!(ax3, nnn_bonds, :red, :dash)
    end

    display(fig)
    return fig
end

"""
Main function that orchestrates the entire calculation using the improved bond extraction.
"""
function main_calculation(config::LatticeConfig = default_triangular_config(), 
                         dmrg_config::DMRGConfig = default_dmrg_config())
    
    println("START OF CALCULATION ==============================")
    println("Periodic boundary conditions in y: ", config.periodic_bc)
    println("Lattice type: $(config.lattice_type)")
    println("Dimensions: $(config.Lx) × $(config.Ly)")
    
    # 1. Set up the crystal structure
    crystal = create_crystal(config)
    if config.lattice_type != CHAIN_1D
        view_crystal(crystal)
    end
    
    # 2. Set up the Sunny system
    sys = setup_sunny_system(crystal, config)

    
    # 4. Extract bonds directly from Sunny system
    bond_pairs = get_unique_bonds(sys, config)
    
    # 5. Organize bonds by coupling strength
    coupling_groups, sorted_couplings = organize_bonds_for_itensor(bond_pairs)

    # 6. Visualize the lattice and bonds
    nn_bonds, nnn_bonds = get_lattice_bonds(config)
    plot_lattice(config, nn_bonds, nnn_bonds)

    # 7. Build the Hamiltonian directly from bond pairs
    H, sites = build_hamiltonian_from_bonds(bond_pairs, config)

    # 8. Create initial state (MAKE THIS BIT MORE GENERAL)
    if config.lattice_type == CHAIN_1D
        linkdims = 10
        psi0 = random_mps(sites;linkdims)
    else
        psi0 = create_initial_state(sites, config)
    end
   
    # 9. Calculate initial energy
    initial_energy = inner(psi0, Apply(H, psi0))
    println("Initial energy = ", initial_energy)
        
    # 10. Run DMRG
    energy, psi = run_dmrg(H, psi0, dmrg_config)
    
    # 11. Report results
    println("\nGround State Energy = ", energy)
    println("Energy per site = ", energy / length(sites))
    println("Using overlap = ", inner(psi, Apply(H, psi)))
    println("Total QN of Ground State = ", totalqn(psi))
    
    # 12. Return additional bond information for analysis
    return energy, psi, H, sites, bond_pairs, coupling_groups
end



"""
# Square lattice
println("\n=== SQUARE LATTICE ===")
sq_config = default_square_config()
energy_sq, psi_sq, H_sq, sites_sq, bonds_sq, couplings_sq = main_calculation(sq_config)
analyze_bond_structure(bonds_sq, couplings_sq, sq_config)


# Triangular lattice dmrg calc
println("=== TRIANGULAR LATTICE ===")
tri_config = default_triangular_config()
energy_tri, psi_tri, H_tri, sites_tri, bonds_tri, couplings_tri = main_calculation(tri_config)
# Analyze the bond structures
analyze_bond_structure(bonds_tri, couplings_tri, tri_config)
"""