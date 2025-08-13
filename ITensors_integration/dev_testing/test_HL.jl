using ITensors, ITensorMPS, GLMakie, FFTW
using Sunny
include("../ITensors_integration.jl")

########
#Harry test -- have left here for posterity but no real functionality
########


units = Units(:meV, :angstrom)
N = 50
η = 0.1
tstep = 0.5
tmax = 5.0
cutoff = 1E-10
maxdim = 300  

sys = create_chain_system(N; periodic_bc = true)
DMRG_results = calculate_ground_state(sys)
ψ = DMRG_results.psi
H = DMRG_results.H
sites = DMRG_results.sites

# Prepare time evolution
ts = 0.0:tstep:tmax
N_timesteps = size(ts,1)
c = div(N, 2)
ϕ = apply_op(ψ, "Sz", sites, c)  # Excited state

# Compute correlation function using TDVP
G = compute_G(N, ψ, ϕ, H, sites, η, collect(ts), tstep, cutoff, maxdim)
energies = range(0, 5, N_timesteps)
println("energies size: ", size(energies))
############################################################
############################################################


cryst = sys.crystal
qs = [[0,0,0], [1,0,0]]
path = q_space_path(cryst, qs, 401)

# UNINTEGRATED WAY USING QuantumCorrelations
qc = QuantumCorrelations(sys; 

                    measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false),
                    energies=energies)

add_sample!(qc,G)


res = intensities(qc, path; energies = :available)
plot_intensities(res; units, title="Dynamic structure factor for 1D chain with qc")
