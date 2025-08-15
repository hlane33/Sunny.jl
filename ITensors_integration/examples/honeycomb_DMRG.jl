include("../ITensors_integration.jl")

#Alternatively, just use one of the lattice helpers to set up the system for you
sys = create_honeycomb_system(8,10; periodic_bc = false)
# Calculate ground state
dmrg_results, _ = calculate_ground_state(sys)

print(dmrg_results.energy)