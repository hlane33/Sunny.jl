using Sunny, ITensors, ITensorMPS, GLMakie, LinearAlgebra

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
    N_basis::Int
    crystal::Any  # Sunny Crystal object
    sys::Any      # Sunny System object
end

function default_dmrg_config()
    return DMRGConfig(10, [10, 20, 100, 100, 200], [1E-8], (1E-7, 1E-8, 0.0))
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

function get_unique_bonds(sys::System)
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
                Matrix(pc.bilin * I, 3, 3)  # 3x3 identity scaled by coupling
            end
            
            # Create bond pair with coupling constant
            push!(bond_pairs, (labelᵢ, labelⱼ, coupling_matrix))
        end
    end
    println("$(length(bond_pairs)) unique bonds found.")
    return unique(bond_pairs), N_basis
end

"""
Builds the ITensor Hamiltonian from processed bond pairs.
"""
function build_hamiltonian_from_bonds(bond_pairs, sys::System; conserve_qns=true)
    # Calculate total number of sites
    N_basis = length(sys.crystal.positions)
    N = prod(sys.dims) * N_basis
    
    sites = siteinds("S=1/2", N; conserve_qns=conserve_qns)
    
    os = OpSum()
    #Currently only does nearest neighbour coupling
    for (i, j, coupling) in bond_pairs
        print("Processing bond pair ($i, $j) with coupling: ")
        # Extract coupling matrix elements
        J_xx = coupling[1, 1]  # SxSx coupling
        J_yy = coupling[2, 2]  # SySy coupling
        J_zz = coupling[3, 3]  # SzSz coupling
        J_xy = coupling[1, 2]  # SxSy coupling
        J_yx = coupling[2, 1]  # SySx coupling
        J_xz = coupling[1, 3]  # SxSz coupling
        J_zx = coupling[3, 1]  # SzSx coupling
        J_yz = coupling[2, 3]  # SySz coupling
        J_zy = coupling[3, 2]  # SzSy coupling

        # Add terms to OpSum
        os += (J_xx + J_yy)/4, "S+", i, "S-", j
        os += (J_xx + J_yy)/4, "S-", i, "S+", j
        os += J_zz, "Sz", i, "Sz", j
        
        # Off-diagonal terms (break QN conservation)
        if !conserve_qns
            os += J_xy, "S+", i, "S-", j
            os += J_yx, "S-", i, "S+", j
            os += J_xz, "S+", i, "Sz", j
            os += J_zx, "Sz", i, "S+", j
            os += J_yz, "S-", i, "Sz", j
            os += J_zy, "Sz", i, "S-", j
        end
    end
    
    H = MPO(os, sites)
    return H, sites
end

"""
Creates initial state for DMRG.
"""
function create_AFM_state(sites, sys::System)
    N = length(sites)
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
Main DMRG calculation function - works with any Sunny system.
Takes a pre-constructed Sunny system and runs DMRG on it.
"""
function calculate_ground_state(sys::System, 
                               dmrg_config::DMRGConfig = default_dmrg_config();
                               conserve_qns::Bool = true,
                               linkdims::Int = 10,
                               show_crystal::Bool = false)
    
    println("START OF CALCULATION ==============================")
    println("System dimensions: $(sys.dims)")
    println("Basis sites per unit cell: $(length(sys.crystal.positions))")
    
    # Ensure system is inhomogeneous
    if Sunny.is_homogeneous(sys)
        error("System must be inhomogeneous. Use `to_inhomogeneous(sys)` first.")
    end
    
    # Extract bonds from the system
    bond_pairs, N_basis = get_unique_bonds(sys)

    # Build Hamiltonian
    H, sites = build_hamiltonian_from_bonds(bond_pairs, sys; conserve_qns=conserve_qns)
    
    # Create initial state
    if conserve_qns
       psi0 = create_AFM_state(sites,sys)
    else
        psi0 = random_mps(sites; linkdims)
    end
   
    # Calculate initial energy
    initial_energy = inner(psi0, Apply(H, psi0))
    println("Initial energy = ", initial_energy)
        
    # Run DMRG
    energy, psi = run_dmrg(H, psi0, dmrg_config)
    
    # Report results
    println("\nGround State Energy = ", energy)
    println("Energy per site = ", energy / length(sites))
    println("Using overlap = ", inner(psi, Apply(H, psi)))
    
    # Create results structure
    results = DMRGResults(energy, psi, H, sites, bond_pairs, 
                         N_basis, sys.crystal, sys)
    
    return results
end

# ============================================================================
# HELPER FUNCTIONS FOR CONSTRUCTING COMMON LATTICE TYPES 
# JUST HELPS ME WHEN TESTING, COULD FEASIBLY DO ANY LATTICE
# ============================================================================

"""
Helper function to create a triangular lattice system.
"""
function create_triangular_system(Lx::Int, Ly::Int, Lz::Int=1; 
                                 a::Float64=1.0, s::Float64=0.5, 
                                 J1::Float64=1.0, J2::Float64=0.0,
                                 periodic_bc::Bool=false)
    
    # Create crystal
    latvecs = lattice_vectors(a, a, 1.0, 90, 90, 120)
    crystal = Crystal(latvecs, [[0, 0, 0]])
    
    # Create system
    pbc = (true, !periodic_bc, true)
    sys = System(crystal, [1 => Moment(; s=s, g=2)], :dipole)
    
    # Set exchanges
    nn_bond = Bond(1, 1, [1, 0, 0])
    nnn_bond = Bond(1, 1, [2, 0, 0])
    set_exchange!(sys, J1, nn_bond)
    set_exchange!(sys, J2, nnn_bond)
    
    # Repeat and make inhomogeneous
    sys = repeat_periodically(sys, (Lx, Ly, Lz))
    sys_inhom = to_inhomogeneous(sys)
    remove_periodicity!(sys_inhom, pbc)
    
    return sys_inhom
end

"""
Helper function to create a square lattice system.
"""
function create_square_system(Lx::Int, Ly::Int, Lz::Int=1; 
                             a::Float64=1.0, s::Float64=0.5, 
                             J1::Float64=1.0, J2::Float64=0.0,
                             periodic_bc::Bool=false)
    
    # Create crystal
    latvecs = lattice_vectors(a, a, 1.0, 90, 90, 90)
    crystal = Crystal(latvecs, [[0, 0, 0]])
    
    # Create system
    pbc = (true, !periodic_bc, true)
    sys = System(crystal, [1 => Moment(; s=s, g=2)], :dipole)
    
    # Set exchanges
    nn_bond = Bond(1, 1, [1, 0, 0])
    nnn_bond = Bond(1, 1, [1, 1, 0])
    set_exchange!(sys, J1, nn_bond)
    set_exchange!(sys, J2, nnn_bond)
    
    # Repeat and make inhomogeneous
    sys = repeat_periodically(sys, (Lx, Ly, Lz))
    sys_inhom = to_inhomogeneous(sys)
    remove_periodicity!(sys_inhom, pbc)
    
    return sys_inhom
end

"""
Helper function to create a 1D chain system.
"""
function create_chain_system(Lx::Int; 
                            a::Float64=1.0, s::Float64=0.5, 
                            J1::Float64=1.0, J2::Float64=0.0,
                            periodic_bc::Bool=false)
    
   # Create crystal
    latvecs = lattice_vectors(a, 10*a, 10*a, 90, 90, 90)
    crystal = Crystal(latvecs, [[0, 0, 0]])
    
    # Create system
    pbc = (!periodic_bc, true, true)
    sys = System(crystal, [1 => Moment(; s=s, g=2)], :dipole; dims=(Lx,1,1))
    
    # Set exchanges
    nn_bond = Bond(1, 1, [1, 0, 0])
    nnn_bond = Bond(1, 1, [2, 0, 0])
    set_exchange!(sys, J1, nn_bond)
    set_exchange!(sys, J2, nnn_bond)
    
    # Repeat and make inhomogeneous
    sys_inhom = to_inhomogeneous(sys)
    remove_periodicity!(sys_inhom, pbc)
    
    return sys_inhom
end

"""
Helper function to create a honeycomb lattice system.
"""
function create_honeycomb_system(Lx::Int, Ly::Int, Lz::Int=1; 
                                a::Float64=2.46, s::Float64=0.5, 
                                J1::Float64=1.0, J2::Float64=0.0,
                                periodic_bc::Bool=false)
    
    # Create crystal
    latvecs = lattice_vectors(a, a, 10.0, 90, 90, 120)
    positions = [[1/3, 2/3, 0]]
    crystal = Crystal(latvecs, positions, 191)
    
    # Create system
    pbc = (true, !periodic_bc, true)
    sys = System(crystal, [1 => Moment(; s=s, g=2)], :dipole)
    
    # Set exchanges
    nn_bond = Bond(1, 2, [0, 0, 0])    # A to B in same cell
    nnn_bond = Bond(1, 1, [1, 0, 0])   # A-A connections
    set_exchange!(sys, J1, nn_bond)
    set_exchange!(sys, J2, nnn_bond)
    
    # Repeat and make inhomogeneous
    sys = repeat_periodically(sys, (Lx, Ly, Lz))
    sys_inhom = to_inhomogeneous(sys)
    remove_periodicity!(sys_inhom, pbc)
    
    return sys_inhom
end

function create_dimerized_spin_chain(Lx::Int, Ly::Int=1, Lz::Int=1; 
                                a::Float64=2.46, s::Float64=0.5, 
                                J1::Float64=1.0, J2::Float64=0.0,
                                periodic_bc::Bool=false)
    # Spin chain system with Spin 1/2  frustrated system that mimics https://arxiv.org/pdf/2507.19412v1

    # CuGeO3 crystallizes in orthorhombic Pnmm space group
    # Approximate lattice parameters (Angstrom)
    a, b, c = 4.8, 8.5, 2.9
    
    latvecs = lattice_vectors(a, b, c, 90, 90, 90)
    
    # Simplified positions - focus on Cu chain along c-axis
    # In reality, there are more atoms, but we focus on magnetic Cu sites
    positions = [[0, 0, 0], [0, 0, 0.5]]  # Cu positions along chain
    
    crystal = Crystal(latvecs, positions, "P m m m")  # Approximate space group
    
    # Create system - extend along c-axis (chain direction)
    sys = System(crystal, (1, 1, 20), [1 => Moment(; s=s, g=2), 2 => Moment(; s=1/2, g=2)], :SUN)

  # Clear any existing interactions
    
    # Nearest-neighbor interactions with dimerization
    # J1 interactions alternate as J1(1±δ) due to dimerization
    
    # For a dimerized chain, we need to handle the alternating bonds
    # This is simplified - in practice, you might need to set bonds individually
    
    # Intra-dimer bonds (stronger): J1(1+δ)
    delta = 0.04

    set_exchange!(sys, J1 * (1 + delta), Bond(1, 2, [0, 0, 0]))
    
    # Inter-dimer bonds (weaker): J1(1-δ)  
    set_exchange!(sys, J1 * (1 - delta), Bond(1, 1, [0, 0, 1]))

    # Next-nearest-neighbor interactions J2
    set_exchange!(sys, J2, Bond(1, 1, [0, 0, 2]))

    sys_inhom = to_inhomogeneous(sys)
    remove_periodicity!(sys_inhom, pbc)
   
    return sys_inhom

# ============================================================================
#  TEST USAGE
# ============================================================================
end 


println("=== DMRG Calculation ===")


#create Kagome
Lx = 4
Ly = 7
pbc = (true, false, true)

units = Units(:meV, :angstrom)
latvecs = lattice_vectors(6, 6, 5, 90, 90, 120)
positions = [[1/2, 0, 0]]
cryst = Crystal(latvecs, positions, 147)
sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
J = -1.0
set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
sys = repeat_periodically(sys, (Lx, Ly, 1))
sys_inhom = to_inhomogeneous(sys)
remove_periodicity!(sys_inhom, pbc)

chain_sys = create_dimerized_spin_chain(20; a=4.2, s=0.5, J1=13.79, J2=4.83, periodic_bc=false)

# Calculate ground state

custom_results = calculate_ground_state(chain_sys; 
                                      conserve_qns=true,  # Off-diagonal terms break QN conservation
                                      )
println("END")


