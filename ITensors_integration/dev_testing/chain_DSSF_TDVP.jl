using ITensors, ITensorMPS, GLMakie, Sunny, Serialization
include("../ITensors_integration.jl")

################
# AFM TDVP Example Chain#
# This was an example I used for testing to compare with the Sunny integrated method
# that the /examples/ use since this does the FT and plotting externally although
# Sunny_to_ITensor.jl is used to create the system and then create the Hamiltonian
################

# TDVP Parameters
η = 0.1 # Damping factor
tstep = 0.2
tmax = 10.0
cutoff = 1E-10 # SVD cutoff for TDVP
maxdim = 300  # Maximum bond dimension for TDVP


N = 20 #Number of lattice sites
sys = create_chain_system(N; periodic_bc = false)
cryst = sys.crystal

q_ends = [[0,0,0], [1,0,0]]
# These are the q-points that would be used in Sunny.intensities()
# But any choice of qs should be fine
path = q_space_path(cryst, q_ends, 500)
qpts = path.qs 
#converts into units of 2pi so that it matches what Fourier transform expects
path_qs = [2pi*q[1] for q in qpts]

# uses Sunny_to_ITensor to create the system and Hamiltonian
# can input a different set DMRG parameters if desired
DMRG_results = calculate_ground_state(sys)
ψ = DMRG_results.psi
H = DMRG_results.H
sites = DMRG_results.sites

# Prepare time evolution
ts = 0.0:tstep:tmax
Lt = length(ts)

#parameters for linear prediction, I found this is really 
#just trial and error to avoid artifacts but https://journals.aps.org/prb/abstract/10.1103/PhysRevB.77.134437
# provides more detail on instances of param choices
linear_predict_params = (
    n_predict = 0, # Matches
    n_coeff = div(Lt,2) # Half the number of coefficients
)


N_timesteps = size(ts,1)
c = div(N, 2)
ϕ = apply_op(ψ, "Sz", sites, c)  # Excited state

# Compute correlation function using TDVP -- or uses stored G array to save time
g_filename = "G_array_$(N)sites_$(tmax)tmax.jls"
G = load_object(g_filename)

# Compute structure factor
allowed_energies = range(0, 5,N_timesteps)
allowed_qs = 0:(1/N):2π
new_allowed_qs = (2π/N) * (0:(N-1))  # [0, 2π/N, 4π/N, ..., 2π(N-1)/N] - should match sunny_to_itensor?
positions = 1:N
# Computes structure factor using Compute_S from useful_TDVP_functions.jl
out = compute_S(G, allowed_qs, allowed_energies, positions, c, ts; linear_predict_params=linear_predict_params)
print("Shape of out: ", size(out))

# Plotting
fig = Figure()
ax = Axis(fig[1, 1],
        xlabel = "qₓ",
        xticks = ([0, allowed_qs[end]], ["0", "2π"]),
        ylabel = "Energy (meV)",
        title = "S=1/2 AFM DMRG/TDVP for Chain lattice w/0 sunny")

# Create heatmap with controlled color range
vmax = 0.4 * maximum(out)  # Set upper limit for better contrast
hm = heatmap!(ax, allowed_qs, allowed_energies, out,
            colorrange = (0, vmax),
            colormap = :viridis)  # :viridis is perceptually uniform

# Add colorbar with clear labeling
cbar = Colorbar(fig[1, 2], hm,
            label = "Intensity (a.u.)",
            vertical = true,
            ticks = LinearTicks(5),
            flipaxis = false)

# Indicate clipped values in colorbar (optional)
cbar.limits = (0, vmax)  # Explicitly show the range
cbar.highclip = :red      # Color for values > vmax
cbar.lowclip = :blue      # Color for values < 0 (if needed)

# Set axis limits
ylims!(ax, 0, 5)


display(fig)
