inlcude("../ITensors_integration")


println("=== KAGOME DMRG Calculation ===")
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

# Calculate ground state

custom_results = calculate_ground_state(sys_inhom; 
                                      conserve_qns=true,  # Off-diagonal terms break QN conservation
                                      )


