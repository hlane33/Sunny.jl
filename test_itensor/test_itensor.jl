using Sunny, ITensors, ITensorMPS, GLMakie

# Configuration structure to hold all parameters
struct LatticeConfig
    Lx::Int # Number of sites in x-direction
    Ly::Int # Number of sites in y-direction
    a::Float64 # Lattice spacing
    s::Float64 # Spin quantum number
    J1::Float64 # Nearest neighbor coupling
    J2::Float64 # Next-nearest neighbor coupling
end

struct DMRGConfig
    nsweeps::Int # Number of DMRG sweeps
    maxdim::Vector{Int} # Maximum bond dimensions
    cutoff::Vector{Float64} # Cutoff values for truncation
    noise::Tuple{Vararg{Float64}} # Noise parameters
end

# Default configurations
function default_lattice_config()
    return LatticeConfig(8, 8, 1.0, 1/2, 1.0, 0)
end

function default_dmrg_config()
    return DMRGConfig(5, [10, 20, 100, 100, 200], [1E-8], (1E-7, 1E-8, 0.0))
end

"""
Maps a 2D triangular lattice to a 1D index using snake ordering.
The lattice is defined by its dimensions Lx and Ly.
Returns the site index for given x and y coordinates.
"""
function get_site_index(x::Int, y::Int, Lx::Int)
    if y % 2 == 1  # Odd rows: left-to-right
        return (y - 1) * Lx + x
    else            # Even rows: right-to-left
        return (y - 1) * Lx + (Lx - x + 1)
    end
end

"""
Creates a Sunny crystal structure for the triangular lattice.
"""
function create_triangular_crystal(config::LatticeConfig)
    latvecs = lattice_vectors(config.a, config.a, 1.0, 90, 90, 120)
    return Crystal(latvecs, [[0, 0, 0]])
end

"""
Sets up the Sunny system with exchange interactions.
"""
function setup_sunny_system(crystal, config::LatticeConfig)
    sys = System(crystal, [1 => Moment(; s=config.s, g=2)], :dipole)
    
    # Set nearest neighbor exchange
    set_exchange!(sys, config.J1, Bond(1, 1, [1, 0, 0]))
    
    # Set next-nearest neighbor exchange
    set_exchange!(sys, config.J2, Bond(1, 1, [2, 0, 0]))
    
    # Repeat periodically
    sys = repeat_periodically(sys, (config.Lx, config.Ly, 1))
    
    return sys
end

"""
Classifies bonds into nearest neighbor and next-nearest neighbor categories.
Returns (nn_bonds, nnn_bonds, J_nn, J_nnn).
"""
function classify_bonds(sys)
    pairs = sys.interactions_union[1].pair
    
    nn_bonds = []
    nnn_bonds = []
    J_nn_bilinear = 0.0
    J_nnn_bilinear = 0.0
    
    for pair in pairs
        bond = pair.bond
        disp = bond.n  # Displacement vector [dx, dy, dz]
        d = abs(maximum(disp[1:2]))  # this is currently quite crude and just makes it nnn 
        # if the max displacement in any direction is more than one
        if d <= 1.0  # Nearest neighbors
            push!(nn_bonds, bond)
            J_nn_bilinear = pair.bilin
        else  # Next-nearest neighbors
            push!(nnn_bonds, bond)
            J_nnn_bilinear = pair.bilin
        end
    end
    
    return nn_bonds, nnn_bonds, J_nn_bilinear, J_nnn_bilinear
end

"""
Converts bond information to site pair indices for ITensor.
"""
function bonds_to_site_pairs(bonds, config::LatticeConfig)
    site_positions = [(x, y) for y in 1:config.Ly, x in 1:config.Lx][:]
    pairs = Tuple{Int,Int}[]
    
    for (x, y) in site_positions
        for bond in bonds
            dx, dy, _ = bond.n
            x2, y2 = x + dx, y + dy
            if 1 ≤ x2 ≤ config.Lx && 1 ≤ y2 ≤ config.Ly
                i = get_site_index(x, y, config.Lx)
                j = get_site_index(x2, y2, config.Lx)
                push!(pairs, (i, j))
            end
        end
    end
    
    # Remove duplicates and ensure i < j ordering
    unique_pairs = unique([i < j ? (i, j) : (j, i) for (i, j) in pairs])
    return unique_pairs
end

"""
Builds the ITensor Hamiltonian from bond information.
"""
function build_hamiltonian(nn_pairs, nnn_pairs, J_nn, J_nnn, config::LatticeConfig)
    N = config.Lx * config.Ly
    sites = siteinds("S=1/2", N; conserve_qns=true)
    
    os = OpSum()
    
    # Add nearest neighbor terms
    for (i, j) in nn_pairs
        os += J_nn * 1.0, "Sz", i, "Sz", j
        os += J_nn * 0.5, "S+", i, "S-", j
        os += J_nn * 0.5, "S-", i, "S+", j
    end
    
    # Add next-nearest neighbor terms
    for (i, j) in nnn_pairs
        os += J_nnn * 1.0, "Sz", i, "Sz", j
        os += J_nnn * 0.5, "S+", i, "S-", j
        os += J_nnn * 0.5, "S-", i, "S+", j
    end
    
    H = MPO(os, sites)
    return H, sites
end

"""
Creates initial Neel state for DMRG.
"""
function create_initial_state(sites, config::LatticeConfig)
    N = config.Lx * config.Ly
    state = [isodd(n) ? "Up" : "Dn" for n=1:N]
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
Plotting function for visualizing the triangular lattice.
"""
function plot_triangular_lattice(config::LatticeConfig, nn_bonds, nnn_bonds)
    fig = Figure(resolution=(800, 800))
    
    # Create subplots
    ax1 = Axis(fig[1, 1], title="Nearest Neighbors (NN)", aspect=DataAspect())
    ax2 = Axis(fig[1, 2], title="Next-Nearest Neighbors (NNN)", aspect=DataAspect())
    ax3 = Axis(fig[2, 1:2], title="Full Triangular Lattice", aspect=DataAspect())

    # Generate site positions
    sites = vec([(x + 0.5*(y%2), y*√3/2) for y in 1:config.Ly, x in 1:config.Lx])
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
                x2 = mod1(x + dx, config.Lx)
                y2 = mod1(y + dy, config.Ly)
                
                i = LinearIndices((config.Lx, config.Ly))[x, y]
                j = LinearIndices((config.Lx, config.Ly))[x2, y2]
                
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

    display(fig)
    return fig
end

"""
Main function that orchestrates the entire calculation.
"""
function main_calculation(lattice_config=default_lattice_config(), 
                         dmrg_config=default_dmrg_config())
    
    println("START OF CALCULATION ==============================")
    
    # 1. Set up the crystal structure
    crystal = create_triangular_crystal(lattice_config)
    view_crystal(crystal; ndims=2)
    
    # 2. Set up the Sunny system
    sys = setup_sunny_system(crystal, lattice_config)
    
    # 3. Classify bonds and extract coupling constants
    nn_bonds, nnn_bonds, J_nn, J_nnn = classify_bonds(sys)
    
    # 4. Convert to site pairs for ITensor
    nn_pairs = bonds_to_site_pairs(nn_bonds, lattice_config)
    nnn_pairs = bonds_to_site_pairs(nnn_bonds, lattice_config)
    
    # 5. Visualize the lattice
    plot_triangular_lattice(lattice_config, nn_bonds, nnn_bonds)
    
    # 6. Build the Hamiltonian
    H, sites = build_hamiltonian(nn_pairs, nnn_pairs, J_nn, J_nnn, lattice_config)
    
    # 7. Create initial state
    psi0 = create_initial_state(sites, lattice_config)
    
    # 8. Calculate initial energy
    initial_energy = inner(psi0, Apply(H, psi0))
    println("Initial energy = ", initial_energy)
    
    # 9. Run DMRG
    energy, psi = run_dmrg(H, psi0, dmrg_config)
    
    # 10. Report results
    println("\nGround State Energy = ", energy)
    println("Using overlap = ", inner(psi, Apply(H, psi)))
    println("Total QN of Ground State = ", totalqn(psi))
    
    return energy
end

# Run the calculation
energy = main_calculation()
println("Calculation is done")