using ITensors, ITensorMPS, GLMakie, FFTW
using Sunny, Serialization

#Decide where you want these to actually be included
include("/Users/adam/Sunny.jl-1/ITensors_integration/ITensors_integration.jl")


units = Units(:meV, :angstrom)
# Lattice configuration
N = 15
# Time evolution parameters
η = 0.1
tstep = 0.2
tmax = 10.0
ts = 0.0:tstep:tmax
Lt = length(ts)
cutoff = 1E-10
maxdim = 300  

#Create system 
sys = create_chain_system(N; periodic_bc = false)
cryst = sys.crystal
q_ends = [[0,0,0], [1,0,0]]
path = q_space_path(cryst, q_ends, 400)
qpts = path.qs #qs in SVector 
path_qs = [2pi*q[1] for q in qpts]
c = div(N,2)
L = div(N,2)
positions = 1:N #Enforces symmetry

#Fourier transform params
new_allowed_qs = (2pi/N) * (0:(N-1))
allowed_qs = 0:(1/N):2π
FT_params = (
    allowed_qs = path_qs,
    energies = range(0, 5, 500),
    positions = positions,
    c = c,
    ts = ts
)
#Linear prediciton params: n_predict is the number of future time steps to predict, 
# n_coeff is the number of coefficients used in linear prediction
linear_predict_params = (
    n_predict = Lt, # Half the number of time steps
    n_coeff = 20 # Half the number of coefficients
)


# File to save/load G array
g_filename = "G_array_$(N)sites_$(tmax)tmax.jls"
dmrg_config = default_dmrg_config()
G = load_G(g_filename, compute_G_wrapper, 
            N, sys, FT_params, η, ts, tstep, cutoff, maxdim, dmrg_config)


# UNINTEGRATED WAY USING QuantumCorrelations
qs_length = length(FT_params.allowed_qs)
energies_length = length(FT_params.energies) #energies after FT 
qc = QuantumCorrelations(sys, qs_length, energies_length ; 
                    measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false),
                    initial_energies = range(0, 5, length(ts))
                    )

add_sample!(qc, G, FT_params, linear_predict_params)
# Generate linearly spaced q-points and intensities params
# Standard Sunny plotting with FT done in accum_sample! and plotting by plot_intensities
res = intensities(qc, path)
fig = plot_intensities(res; units, title="Dynamic structure factor for 1D chain length $N", saturation=0.9)

display(fig)