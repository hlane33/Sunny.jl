using ITensors, ITensorMPS, GLMakie, FFTW
using Sunny, Serialization

include("../ITensors_integration.jl")
include("2D_FT_testing.jl")


#----------------------
# This attempts to use 2D FT to do the fourier transform for a square lattice 
# It runs but fails to produce a plot, probably because the FT is set up wrong
#------------------------


# Create system 
Lx = Ly = 4
periodic_bc = true
sys = create_square_system(Lx, Ly; periodic_bc = periodic_bc)
cryst = sys.crystal
q_ends = [[0,0,0], [1,0,0]]
path = q_space_path(cryst, q_ends, 400)
qpts = path.qs
path_qs = [2π*q[1] for q in qpts]

c = div(Lx, 2)

# Define your momentum and frequency grids
qxs = range(-π, π, length=50)  # Or whatever range you want
qys = range(-π, π, length=50)
ωs = range(0, 5, length=100)   # Energy range
positions = extract_positions_from_sunny(sys)
print(positions)
tstep = 0.5
tmax  = 5.0
ts = 0.0:tstep:tmax

# Create FT parameters using the Sunny system
ft_params = FTParams(
    allowed_qxs = collect(qxs),
    allowed_qys = collect(qys),
    c = c,
    energies = ωs,
    positions = positions,
    ts = collect(ts)
)

linear_predict_params = LinearPredictParams(
    n_predict = 5,
    n_coeff   = 5
)

# ----------------------
# Load or compute G
# ----------------------
tdvp_params = TDVPParams(
    N       = prod(sys.dims),
    η       = 0.1,
    tstep   = tstep,
    tmax    = tmax,
    cutoff  = 1e-10,
    maxdim  = 300
)
ts = 0.0:tdvp_params.tstep:tdvp_params.tmax
g_filename = "G_array_$(tdvp_params.N)sites_$(tdvp_params.tmax)tmax.jls"

G, n_to_cartind = load_G(
    g_filename,
    compute_G_wrapper,
    sys,
    tdvp_params,
    ft_params,
    linear_predict_params
)


#returns G that accounts for 2D
G_2D = map_G_to_2D(G, sys, length(ts))

# ----------------------
# Post-processing & plotting
# ----------------------
qxs_length = length(ft_params.allowed_qxs)
qys_length = length(ft_params.allowed_qys)
energies_length = length(ft_params.energies)

qc = QuantumCorrelations(
    sys,
    energies_length,
    qxs_length,
    qys_length;
    measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys; apply_g=false),
    num_timesteps = length(ts)
)

add_sample!(qc, G_2D, ft_params, linear_predict_params, assume_real_S=true)

res = intensities(qc, ft_params.energies, path)
fig = plot_intensities(
    res;
    units = Units(:meV, :angstrom),
    title = "Dynamic structure factor for 1D chain length $(tdvp_params.N)",
    saturation = 0.9
)

display(fig)
