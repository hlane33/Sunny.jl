using ITensors, ITensorMPS, GLMakie, FFTW
using Sunny, Serialization

include("../ITensors_integration.jl")

# ----------------------
# Parameter setup
# ----------------------

#This example shows how using the code within `ITensors_integration/.` a system can be created in Sunny
# converted into a form on which DMRG/TEBD/TDVP... can be performed and then integrated back into the Sunny architecture
# This full process currently only works for 1D systems due to the limitations on the fourier transform in Compute_S()
# But see dev_testing/square_lattice_TDVP and dev_testing/2D_FT_testing/ for ideas on the 2D case.

# Start by defining the size of your chain
N = 32 #size of chain 

# Choose the parameters for your time evolution
tdvp_params = TDVPParams(
    N       = N,
    η       = 0.1,
    tstep   = 0.2,
    tmax    = 10.0,
    cutoff  = 1e-10,
    maxdim  = 300
)

#Make an array of the time evolution for use in the Fourier transform later
ts = 0.0:tdvp_params.tstep:tdvp_params.tmax

# Create system using `sunny_toITensor.jl`
# This uses one of the helpers from `lattice_helpers.jl`
# I would recommend looking there for guidance on how to set up the system initially
sys = create_chain_system(N; periodic_bc = false)
cryst = sys.crystal

# Here you choose the qs you want during the Fourier transform
# you should be able to choose any array of qs, but here using Sunny's q_space_path 
# functionality can be helpful, though slightly redundant in this 1D case
q_ends = [[0,0,0], [1,0,0]]
path = q_space_path(cryst, q_ends, 400)
qpts = path.qs
# We want the qs that go into the Fourier Transform to be in units of 2π rather than RLU
path_qs = [2π*q[1] for q in qpts]

# c will mess things up more than you might expect, this divides the chain in two and is important
# for setting the reference spin when you calculate the correlations and then later the Fourier Transform
# I would reccommend offsetting it by 1 to see the impact on the final plot
c = div(N, 2)

# Positions are extracted using Sunny's get_global_positions() function
# but offset by [1,1,0] to match the numbering in TDVP code
positions = extract_positions_from_sunny(sys) 

# Set the parameters for the fourier transform
# Allowed_qxs, energies are the externally applied energy arrays that increase the resolution
# of the plotting (Experiment with this!).
ft_params = FTParams(
    allowed_qxs = path_qs,
    energies   = range(0, 5, 500),
    positions  = positions,
    c          = c,
    ts         = ts
)


# LinearPredictParams is a struct that passes the parameters
# to linear_predict() to apply linear regression as a form of windowing
linear_predict_params = LinearPredictParams(
    n_predict = 20,
    n_coeff   = 15
)

# ----------------------
# Load or compute G
# ----------------------
#It may be that you don't want to recompute G every time when trying to modify plot settings 
# or the fourier transform, so this offers a way to save the correlations to be reused
g_filename = "G_array_$(tdvp_params.N)sites_$(tdvp_params.tmax)tmax.jls"

#Use load G to either load a preexisting G with N=N and tmax=tmax or compute G for a new set of parameters
# n_to_cartind is a vector of tuples that stores the mapping of site coords to linear index
G, n_to_cartind = load_G(
    g_filename,
    compute_G_wrapper,
    sys,
    tdvp_params,
    ft_params,
    linear_predict_params
)



function lattice_fourier_out(G::AbstractMatrix{<:Number};dt)
    # Drop final time slice to avoid double-counting t=0 and t=T
    Gt = G[:, 1:end-1]
    Nx, Nt = size(Gt)

    # Spatial FFT (along x, dimension 1): implements exp(-i q x)
    G_qt = ifft(Gt, 1)/Nx

    # Temporal FFT (along t, dimension 2): convention exp(+i ω t)
    G_qw = fft(G_qt, 2)

    # Shift to center q=0 and ω=0 for plotting
    G_qw_shifted = fftshift(fftshift(G_qw, 1), 2)

    # Momentum and frequency axes

    return  G_qw_shifted
end

Gpad =zeros(Float64,size(G))
Gpadded = hcat(G,Gpad)
Gw = lattice_fourier_out(Gpadded;dt=tdvp_params.tstep)
heatmap(real.(Gw);colorrange = (0,0.001))

# When DMRG and TEBD are performed, 2D systems are treated as 1D by mapping this coordinate
# to a single site number. This function undoes that mapping once DMRG/TDVP is complete.
# For details on how the mapping is set in the first place, see cartind_to_label()
G_2D = map_G_to_2D(G, sys, length(ts))

# ----------------------
# Post-processing & plotting
# ----------------------

# This takes the length of the imposed momenta and energy grids to be passed to QuantumCorrelations
# such that the array where the S(q,ω) is stored is initialised at the correct size
qs_length = length(ft_params.allowed_qxs)
energies_length = length(ft_params.energies)

# Creates a QuantumCorrelations object that mimics SampledCorrelations to compute and store S(q,ω)
qc = QuantumCorrelations(
    sys,
    energies_length,
    qs_length;
    measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys; apply_g=false),
    num_timesteps = length(ts)
)

# Mimics add_sample!(sc) in Sunny to add the correlations to the buffer and then compute the structure factor
add_sample!(qc, G_2D, ft_params, linear_predict_params; assume_real_S=true)

# Uses a modified version of Sunny.intensities to process the structure factor data
# For a 1D chain this doesn't do much beyond allow for the plotting using plot_intensities!
res = intensities(qc, ft_params.energies, path)
#Plot your structure factor!
fig = plot_intensities(
    res;
    units = Units(:meV, :angstrom),
    title = "Dynamic structure factor for 1D chain length $(tdvp_params.N)",
    saturation = 0.9
)

display(fig)
