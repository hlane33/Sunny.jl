using ITensors, ITensorMPS, GLMakie, FFTW
using Sunny, Serialization

include("../ITensors_integration.jl")


# ----------------------
# Parameter setup
# ----------------------
tdvp_params = TDVPParams(
    N       = 17,
    η       = 0.1,
    tstep   = 0.2,
    tmax    = 10.0,
    cutoff  = 1e-10,
    maxdim  = 300
)

ts = 0.0:tdvp_params.tstep:tdvp_params.tmax

# Create system 
sys = create_chain_system(tdvp_params.N; periodic_bc = tdvp_params.periodic_bc)
cryst = sys.crystal
q_ends = [[0,0,0], [1,0,0]]
path = q_space_path(cryst, q_ends, 400)
qpts = path.qs
path_qs = [2π*q[1] for q in qpts]

c = div(tdvp_params.N, 2)
positions = 1:tdvp_params.N

ft_params = FTParams(
    allowed_qs = path_qs,
    energies   = range(0, 5, 500),
    positions  = positions,
    c          = c,
    ts         = ts
)

linear_predict_params = LinearPredictParams(
    n_predict = 5,
    n_coeff   = 5
)

# ----------------------
# Load or compute G
# ----------------------
g_filename = "G_array_$(tdvp_params.N)sites_$(tdvp_params.tmax)tmax.jls"

G = load_G(
    g_filename,
    compute_G_wrapper,
    sys,
    tdvp_params,
    ft_params,
    linear_predict_params
)

# ----------------------
# Post-processing & plotting
# ----------------------
qs_length = length(ft_params.allowed_qs)
energies_length = length(ft_params.energies)

qc = QuantumCorrelations(
    sys,
    qs_length,
    energies_length;
    measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys; apply_g=false),
    num_timesteps = length(ts)
)

add_sample!(qc, G, ft_params, linear_predict_params)

res = intensities(qc, ft_params.energies, path)
fig = plot_intensities(
    res;
    units = Units(:meV, :angstrom),
    title = "Dynamic structure factor for 1D chain length $(tdvp_params.N)",
    saturation = 0.9
)

display(fig)
