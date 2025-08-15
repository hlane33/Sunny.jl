# ============================================================================
# HELPER FUNCTIONS FOR CONSTRUCTING COMMON LATTICE Systems in Sunny
# that integrates with set up by returning sys_inhom
# ============================================================================

"""
    create_triangular_system(Lx, Ly, Lz=1; kwargs...)

Create triangular lattice system with nearest and next-nearest neighbor couplings.

# Arguments
- `Lx::Int`, `Ly::Int`, `Lz::Int=1`: System dimensions

#Kwargs
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
#Kwargs
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

# Kwargs
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

#Kwargs
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

