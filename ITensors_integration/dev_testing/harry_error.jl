using ITensors, ITensorMPS, GLMakie, FFTW
using Sunny, Serialization

#Decide where you want these to actually be included
include("../ITensors_integration.jl")

units = Units(:meV, :angstrom)
N = 31
η = 0.3
tstep = 0.2
tmax = 10.0
ts = 0.0:tstep:tmax
Lt = length(ts)
cutoff = 1E-10
maxdim = 300  

sys = create_chain_system(N; periodic_bc = false)
cryst = sys.crystal
q_ends = [[0,0,0], [1,0,0]]
path = q_space_path(cryst, q_ends, 400)
qpts = path.qs  
path_qs = [2pi*q[1] for q in qpts]
c = div(N,2)
positions = 1:N
new_allowed_qs = (2pi/N) * (0:(N-1))
allowed_qs = 0:(1/N):2π
FT_params = (allowed_qs = path_qs,energies = range(0, 5, 500),positions = positions,c = c,ts = ts)
linear_predict_params = (n_predict = Lt, n_coeff = 20)
dmrg_config = default_dmrg_config()
DMRG_results = calculate_ground_state(sys)
ψ = DMRG_results.psi
ϕ = apply_op(ψ, "Sz", sites, c)  # Excited state
E0 = DMRG_results.energy
H = DMRG_results.H
sites = DMRG_results.sites

ts = 0.0:tstep:tmax
G = compute_G(N, ψ, ϕ, H, sites, η, ts, tstep, cutoff, maxdim)