# ============================================================================
# HELPER FUNCTIONS FOR CONSTRUCTING COMMON LATTICE Systems in Sunny
# that integrates with set up by returning sys_inhom
# ============================================================================

"""
    create_triangular_system(Lx, Ly, Lz=1; kwargs...)

Create triangular lattice system with nearest and next-nearest neighbor couplings.

# Arguments
- `Lx::Int`, `Ly::Int`, `Lz::Int=1`: System dimensions
- `a::Float64=1.0`: Lattice constant  
- `s::Float64=0.5`: Spin magnitude
- `J1::Float64=1.0`: Nearest neighbor coupling
- `J2::Float64=0.0`: Next-nearest neighbor coupling
- `periodic_bc::Bool=false`: Whether to use periodic boundary conditions (cylindrical or not)

Returns an inhomogeneous Sunny system ready for DMRG calculation.
"""
function create_triangular_system(Lx::Int, Ly::Int, Lz::Int=1; 
                                 a::Float64=1.0, s::Float64=0.5, 
                                 J1::Float64=1.0, J2::Float64=0.0,
                                 periodic_bc::Bool=false)
    
    # Create crystal
    latvecs = lattice_vectors(a, a, 1.0, 90, 90, 120)
    crystal = Crystal(latvecs, [[0, 0, 0]])
    
    # Create system
    pbc = (true, !periodic_bc, true) #if true then not periodic in that direction
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
    create_square_system(Lx, Ly, Lz=1; kwargs...)

Create square lattice system with nearest and next-nearest neighbor couplings.

# Arguments  
- `Lx::Int`, `Ly::Int`, `Lz::Int=1`: System dimensions
- `a::Float64=1.0`: Lattice constant
- `s::Float64=0.5`: Spin magnitude  
- `J1::Float64=1.0`: Nearest neighbor coupling
- `J2::Float64=0.0`: Next-nearest neighbor (diagonal) coupling
- `periodic_bc::Bool=false`: Whether to use periodic boundary conditions

Returns an inhomogeneous Sunny system ready for DMRG calculation.
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
    create_chain_system(Lx; kwargs...)

Create 1D spin chain system with nearest and next-nearest neighbor couplings.

# Arguments
- `Lx::Int`: Chain length
- `a::Float64=1.0`: Lattice spacing
- `s::Float64=0.5`: Spin magnitude
- `J1::Float64=1.0`: Nearest neighbor coupling  
- `J2::Float64=0.0`: Next-nearest neighbor coupling
- `periodic_bc::Bool=false`: Whether to use periodic boundary conditions

Returns an inhomogeneous Sunny system ready for DMRG calculation.
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
    create_honeycomb_system(Lx, Ly, Lz=1; kwargs...)

Create honeycomb lattice system with nearest and next-nearest neighbor couplings.

# Arguments
- `Lx::Int`, `Ly::Int`, `Lz::Int=1`: System dimensions
- `a::Float64=2.46`: Lattice constant  
- `s::Float64=0.5`: Spin magnitude
- `J1::Float64=1.0`: Nearest neighbor coupling (A-B sublattices)
- `J2::Float64=0.0`: Next-nearest neighbor coupling (within sublattice)
- `periodic_bc::Bool=false`: Whether to use periodic boundary conditions

Returns an inhomogeneous Sunny system ready for DMRG calculation.
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

"""
    create_dimerized_spin_chain(Lx=4, Ly=1, Lz=1; kwargs...)

Create dimerized spin chain with alternating bond strengths (J₁±δ model).

# Arguments  
- `Lx::Int=4`, `Ly::Int=1`, `Lz::Int=1`: System dimensions
- `a::Float64=2.46`: Lattice constant
- `s::Float64=0.5`: Spin magnitude
- `J1::Float64=1.0`: Average nearest neighbor coupling
- `J2::Float64=0.0`: Next-nearest neighbor coupling  
- `delta::Float64=0.04`: Dimerization parameter (bond alternation)
- `periodic_bc::Bool=false`: Whether to use periodic boundary conditions

Creates a frustrated spin chain with alternating strong (J₁+δ) and weak (J₁-δ) bonds.
Based on the J₁-J₂-δ dimerized chain model.

Returns an inhomogeneous Sunny system ready for DMRG calculation.
"""
function create_dimerized_spin_chain(Lx::Int=4, Ly::Int=1, Lz::Int=1;
                                      a::Float64=2.46, s::Float64=0.5,
                                      J1::Float64=1.0, J2::Float64=0.0,
                                      delta::Float64=0.04,
                                      periodic_bc::Bool=false)
    
    # Spin chain system with Spin 1/2 frustrated system that mimics https://arxiv.org/pdf/2507.19412v1
    # Using J1-J2 δ dimerized chain model
    
    # For a proper dimerized chain, we need a 4-site unit cell to capture the alternating pattern
    # Unit cell: A-B-A-B with alternating strong/weak bonds
    # Strong bonds: A(1)-B(2) and A(3)-B(4) 
    # Weak bonds: B(2)-A(3) and B(4)-A(1+1) (next unit cell)

    # Lattice vectors (monoclinic, P2₁/m)
    a = 2.0  # doubled due to dimerization
    b = 1.0  # arbitrary
    c = 1.0  # arbitrary
    β = 90.0 # monoclinic angle
    γ = 95.0 # monoclinic angle
    latvecs = lattice_vectors(a, b, c, β, γ, β) # Simple orthorhombic for simplicity

    # Atomic positions (2 spins per cell)
    positions = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]  # in fractional coordinates
    types = ["Cu","Cu"]  # Assuming spin-1/2 Cu²⁺ ions
    cryst = Crystal(latvecs, positions; types)  # 11 = P2₁/m space group number

    # Spin properties (e.g., S=1/2, g=2)
    S = 1/2
    g = 2.0
    sys = System(crystal, [1 => Moment(; s=s, g=2)], :dipole; dims=(Lx,1,1))  # 1x1x1 supercell (minimal for dimerization)

    # Exchange interactions (J₁ ± δ)
    J1 = 1.0   # Average nearest-neighbor coupling
    J2 = 0.2  # Example value
    δ = 0.04   # Dimerization parameter
    J1_strong = J1 + δ
    J1_weak = J1 - δ

    # Set alternating bonds along a-axis
    set_exchange!(sys, J1_strong, Bond(1, 2, [0,0,0]))  # Intra-cell strong bond
    set_exchange!(sys, J1_weak, Bond(2, 1, [1,0,0]))    # Inter-cell weak bond (PBC wraps)

    # Optional: Add J₂ (next-nearest-neighbor)
    
    set_exchange!(sys, J2, Bond(1, 1, [1,0,0]))  # J₂ couples spin 1 to itself in next cell
    set_exchange!(sys, J2, Bond(2, 2, [1,0,0]))
    # Visualize the crystal structure
    fig = view_crystal(sys; ndims=2)
    display(fig)
    
    # Convert to inhomogeneous system
    sys_inhom = to_inhomogeneous(sys)
    
    # Set boundary conditions
    # Remove periodicity in x-direction (chain direction) only
    pbc = (!periodic_bc, true, true)  # (x, y, z) - remove periodicity in x
    remove_periodicity!(sys_inhom, pbc)
    println("Applied open boundary conditions in chain direction")

    println("Using periodic boundary conditions")

    
    return sys_inhom
end