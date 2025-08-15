include("../ITensors_integration.jl")

# This example is simply to demonstrate that the DMRG should feasibly work for any 2D system 
# with heisenberg type terms (though not necessarily isotoropic) in the hamiltonian

println("=== KAGOME DMRG Calculation ===")
#create simple Kagome lattice
Lx = 4
Ly = 7

#Choose whether you want periodic boundary conditions in (x,y,z)
pbc = (true, false, true)

units = Units(:meV, :angstrom)
# Set up simple Kagome lattice, see `Sunny/examples/spinw_tutorials/SWO5_Simple_Kagome_FM.jl`
latvecs = lattice_vectors(6, 6, 5, 90, 90, 120)
positions = [[1/2, 0, 0]]
cryst = Crystal(latvecs, positions, 147)
sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
J = -1.0
set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
sys = repeat_periodically(sys, (Lx, Ly, 1))

# Set the system to inhomogenous so that the DMRG in `sunny_toITensor` can be computed
sys_inhom = to_inhomogeneous(sys)
# Applies periodic boundary conditions
remove_periodicity!(sys_inhom, pbc)

# Calculate ground state - see DMRGResults type for what this contains
dmrg_results, _ = calculate_ground_state(sys_inhom; 
                                      conserve_qns=true,  # Off-diagonal terms break QN conservation
                                      )


