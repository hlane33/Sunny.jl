using Sunny, ITensors, ITensorMPS, GLMakie, LinearAlgebra


#########
# This code integrates Sunny.jl with ITensors.jl, attempting to interface the two such that
# it is easy to create the systems using Sunny whilst leveraging the DMRG ability in ITensors
# The code does this by extracting the bond information from Sunny and processing it appropriately 
# into ITensor to construct the hamilton and then perform DMRG.
########

"""
    DMRGConfig

Configuration parameters for DMRG calculations.

# Fields
- `nsweeps::Int`: Number of DMRG sweeps to perform
- `maxdim::Vector{Int}`: Maximum bond dimensions for each sweep
- `cutoff::Vector{Float64}`: Truncation cutoffs for each sweep  
- `noise::Tuple{Vararg{Float64}}`: Noise parameters for each sweep

# Example
```julia
config = default_dmrg_config()
config.nsweeps = 15
config.maxdim = [20, 50, 100]
```
"""
mutable struct DMRGConfig
    nsweeps::Int
    maxdim::Vector{Int}
    cutoff::Vector{Float64}
    noise::Tuple{Vararg{Float64}}
end

"""
    DMRGResults

Container for DMRG calculation results and associated system information.

# Fields
- `energy::Float64`: Ground state energy
- `psi::MPS`: Ground state wavefunction as Matrix Product State
- `H::MPO`: Hamiltonian as Matrix Product Operator
- `sites::Vector`: ITensor site indices
- `bond_pairs::Vector`: List of bond pairs with coupling matrices
- `N_basis::Int`: Number of basis sites per unit cell
- `crystal::Any`: Sunny Crystal object
- `sys::Any`: Sunny System object
"""
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

"""
    default_dmrg_config()

Create default DMRG configuration with reasonable parameters for most calculations.

Returns a `DMRGConfig` with 10 sweeps, bond dimensions [10,20,100,100,200], 
cutoff 1E-8, and noise schedule (1E-7, 1E-8, 0.0).
"""
function default_dmrg_config()
    return DMRGConfig(10, [10, 20, 100, 100, 200], [1E-8], (1E-7, 1E-8, 0.0))
end

"""
    cartind_to_label(cartind, dims, N_basis; perm=nothing)

Convert CartesianIndex to 1D label for ITensor indexing.

# Arguments
- `cartind::CartesianIndex`: Cartesian index from Sunny system
- `dims`: System dimensions (Lx, Ly, Lz)
- `N_basis::Int`: Number of basis sites per unit cell
- `perm=nothing`: Optional permutation function to apply to result

Maps a 4D CartesianIndex (i,j,1,k) to a linear site index suitable for ITensor.
"""
function cartind_to_label(cartind::CartesianIndex, dims, N_basis; perm=nothing)
    i, j = cartind[1], cartind[2]
    Lx, Ly = dims[1], dims[2] 
    
    k = cartind[4] # kth atom in basis
    n = (i-1)*Ly*N_basis + (j-1)*N_basis + k

    perm === nothing ? n : perm(n)
end

"""
    get_unique_bonds(sys)

Extract unique bond pairs and coupling matrices from a Sunny system.

# Arguments
- `sys::System`: Inhomogeneous Sunny system

Returns a tuple `(bond_pairs, N_basis)` where `bond_pairs` contains tuples of 
`(site_i, site_j, coupling_matrix)`, `N_basis` is the number of basis sites,
n_to_cartind is the mapping from n to [i,j,k,l] to be inverted later.

# Notes
The system must be inhomogeneous. Scalar couplings are converted to 3×3 diagonal matrices.
Culled bonds are automatically filtered out.
"""
function get_unique_bonds(sys::System)
    Sunny.is_homogeneous(sys) && error("Use `to_inhomogeneous` first.")
    ints = Sunny.interactions_inhomog(sys)
    dims = sys.dims

    sites = Sunny.eachsite(sys)
    N_basis = length(sys.crystal.positions)
    bond_pairs = []
    
    #creates object to save mapping to for inversion after Time evolution
     n_to_cartind = Vector{NTuple{4, Int}}(undef, prod(dims) * N_basis)  # Maps labelᵢ → (i,j,k,l)
    for (j, int) in enumerate(ints)
        for pc in int.pair
            (; bond, isculled) = pc
            #saves mapping n --> [i,j,k,l] for all sites
            siteᵢ = sites[j] 
            # This mapping doesn't care about the culling
            label = cartind_to_label(siteᵢ, sys.dims, N_basis)
            n_to_cartind[label] = (siteᵢ[1], siteᵢ[2], siteᵢ[3], siteᵢ[4])

            # now culls bonds so no double counting
            isculled && continue
            siteⱼ = Sunny.bonded_site(siteᵢ, bond, sys.dims)
        
            # Get linear labels
            #you care about the bonds here and therefore the culling
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
    return unique(bond_pairs), N_basis, n_to_cartind
end

"""
    build_hamiltonian_from_bonds(bond_pairs, sys; conserve_qns=true)

Build ITensor Hamiltonian MPO from processed bond pairs.

# Arguments
- `bond_pairs`: Vector of `(site_i, site_j, coupling_matrix)` tuples
- `sys::System`: Sunny system (used for size information)
- `conserve_qns::Bool=true`: Whether to conserve quantum numbers (total Sz)

Returns `(H, sites)` where `H` is the Hamiltonian MPO and `sites` are the physical indices.

# Notes
When `conserve_qns=false`, 
includes all off-diagonal coupling terms that break Sz conservation.
"""
function build_hamiltonian_from_bonds(bond_pairs, sys::System; conserve_qns=true)
    # Calculate total number of sites
    N_basis = length(sys.crystal.positions)
    N = prod(sys.dims) * N_basis
    
    sites = siteinds("S=1/2", N; conserve_qns=conserve_qns)
    
    os = OpSum()
    for (i, j, coupling) in bond_pairs
        #print("Processing bond pair ($i, $j) with coupling:  $coupling\n")
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
    create_AFM_state(sites, sys)

Create antiferromagnetic initial state for DMRG.

Alternates spins up/down for each site. Used as initial guess when quantum numbers are conserved.
"""
function create_AFM_state(sites, sys::System)
    N = length(sites)
    state = [isodd(n) ? "Up" : "Dn" for n=1:N]
    return MPS(sites, state)
end

"""
    run_dmrg(H, psi0, dmrg_config)

Perform DMRG calculation with given Hamiltonian and initial state.

# Arguments
- `H`: Hamiltonian as ITensor MPO
- `psi0`: Initial state as ITensor MPS  
- `dmrg_config::DMRGConfig`: DMRG parameters

Returns `(energy, psi)` where `energy` is the ground state energy and `psi` is the converged MPS.
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
    calculate_ground_state(sys; kwargs...)

Main DMRG calculation function for any Sunny system.

# Arguments
- `sys::System`: Pre-constructed inhomogeneous Sunny system

# Keyword Arguments
- `dmrg_config::DMRGConfig=default_dmrg_config()`: DMRG parameters
- `conserve_qns::Bool=true`: Whether to conserve total Sz quantum number
- `linkdims::Int=10`: Bond dimension for random initial state (when `conserve_qns=false`)
- `show_crystal::Bool=false`: Whether to display crystal structure

Returns a `DMRGResults` object containing the ground state energy, wavefunction, 
and all associated system information, and the mapping of site coords to DMRG mapping.
see [`cartind_to_label`]@ref

# Example
```julia
sys = create_square_system(4, 4; J1=1.0, J2=0.2)
results = calculate_ground_state(sys; conserve_qns=true)
println("Ground state energy: ", results.energy)
```

# Notes
The input system must be inhomogeneous (use `to_inhomogeneous(sys)` if needed).
When `conserve_qns=true`, uses antiferromagnetic initial state. When `false`, 
uses random initial state and includes all coupling terms.
"""
function calculate_ground_state(sys::System;
                               dmrg_config::DMRGConfig = default_dmrg_config(),
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
    bond_pairs, N_basis, n_to_cartind = get_unique_bonds(sys)

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
    
    return results, n_to_cartind
end

