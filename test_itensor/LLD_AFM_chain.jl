# # 4. Generalized spin dynamics of FeI₂ at finite *T*

# The [previous FeI₂ tutorial](@ref "3. Multi-flavor spin wave simulations of
# FeI₂") used multi-flavor spin wave theory to calculate the dynamical spin
# structure factor. This tutorial performs an analogous calculation at finite
# temperature using the [classical dynamics of SU(_N_) coherent
# states](https://doi.org/10.1103/PhysRevB.106.054423).
#
# Compared to spin wave theory, classical spin dynamics in real-space is
# typically much slower and is limited in ``𝐪``-space resolution. The approach,
# however, allows for thermal fluctuations. This allows to explore [finite
# temperature phases](https://doi.org/10.1103/PhysRevB.109.014427) and enables
# the study of [non-equilibrium
# processes](https://doi.org/10.1103/PhysRevB.106.235154).
#
# The structure of this tutorial largely follows the [previous study of CoRh₂O₄
# at finite *T*](@ref "2. Landau-Lifshitz dynamics of CoRh₂O₄ at finite *T*").
# The main difference is that CoRh₂O₄ can be well described with `:dipole` mode,
# whereas FeI₂ is best modeled using `:SUN` mode, owing to its strong easy-axis
# anisotropy.
#
# Construct the FeI₂ system as [previously described](@ref "3. Multi-flavor spin
# wave simulations of FeI₂").

using Sunny, GLMakie

units = Units(:meV, :angstrom)

# Lattice (1D chain along x)
a = 1.0  # Lattice constant
latvecs = lattice_vectors(a, 10*a, 10*a, 90, 90, 90)
crystal = Crystal(latvecs, [[0, 0, 0]])  # 1 spin per unit cell

# System with AFM-compatible supercell
sys = System(crystal, [1 => Moment(; s=1/2, g=2)], :SUN; dims=(5, 1, 1))

# AFM exchange coupling (J1 > 0)
J1 = 1.0  # meV
set_exchange!(sys, J1, Bond(1, 1, [1, 0, 0]))  # Nearest-neighbor AFM

# Repeat supercell to desired length (e.g., 20 spins = 10 supercells)
Lx = 5  # Number of unit cells (each with 2 spins)
repeat_periodically(sys, (Lx, 1, 1))  # Now has 20 spins



# Direct optimization via [`minimize_energy!`](@ref) is susceptible to trapping
# in a local minimum. An alternative approach is to simulate the system using
# [`Langevin`](@ref) spin dynamics. This requires a bit more set-up, but allows
# sampling from thermal equilibrium at any target temperature. Select the
# temperature 2.3 K ≈ 0.2 meV. This temperature is small enough to magnetically
# order, but large enough so that the dynamics can readily overcome local energy
# barriers and annihilate defects.

langevin = Langevin(; damping=0.2, kT=0.001*units.K)

# Use [`suggest_timestep`](@ref) to select an integration timestep for the error
# tolerance `tol=1e-2`. Initializing `sys` to some low-energy configuration
# usually works well.

randomize_spins!(sys)
minimize_energy!(sys; maxiters=10)
suggest_timestep(sys, langevin; tol=1e-2)
langevin.dt = 0.03;

# Run a Langevin trajectory for 10,000 time-steps and plot the spins. The
# magnetic order is present, but may be difficult to see.

for _ in 1:10_000
    step!(sys, langevin)
end
plot_spins(sys; color=[S[3] for S in sys.dipoles])

# Verify the expected two-up, two-down spiral magnetic order by calling
# [`print_wrapped_intensities`](@ref). A single propagation wavevector ``±𝐤``
# dominates the static intensity in ``\mathcal{S}(𝐪)``, indicating the expected
# 2 up, 2 down magnetic spiral order. A smaller amount of intensity is spread
# among many other wavevectors due to thermal fluctuations.

print_wrapped_intensities(sys)

# Thermalization has not substantially altered the suggested `dt`.

suggest_timestep(sys, langevin; tol=1e-2)

# ### Structure factor in the paramagnetic phase

# The remainder of this tutorial will focus on the paramagnetic phase.
# Re-thermalize the system to the temperature of 3.5 K ≈ 0.30 meV.

langevin.kT = 0.001 * units.K
for _ in 1:10_000
    step!(sys, langevin)
end

# The suggested timestep has increased slightly. Following this suggestion will
# make the simulations a bit faster.

suggest_timestep(sys, langevin; tol=1e-2)
langevin.dt = 0.040;

# Collect dynamical spin structure factor data using
# [`SampledCorrelations`](@ref). This procedure involves sampling spin
# configurations from thermal equilibrium and using the [spin dynamics of
# SU(_N_) coherent states](https://arxiv.org/abs/2204.07563) to estimate
# dynamical correlations. With proper classical-to-quantum corrections, the
# associated structure factor intensities ``S^{αβ}(q,ω)`` can be compared with
# finite-temperature inelastic neutron scattering data. Incorporate the
# [`FormFactor`](@ref) appropriate to Fe²⁺. 

dt = 2*langevin.dt
energies = range(0, 5, 2)
sc = SampledCorrelations(sys; dt, energies, measure=ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false))

# The function [`add_sample!`](@ref) will collect data by running a dynamical
# trajectory starting from the current system configuration. 

Sunny.add_sample!(sc, sys)

# To collect additional data, it is required to re-sample the spin configuration
# from the thermal distribution. Statistical error is reduced by fully
# decorrelating the spin configurations between calls to `add_sample!`.


# Perform an intensity calculation for two special ``𝐪``-points in reciprocal
# lattice units (RLU). A classical-to-quantum rescaling of normal mode
# occupations will be performed according to the temperature `kT`. The large
# statistical noise could be reduced by averaging over more thermal samples.

qs = [[0,0,0], [1,0,0]]
cryst = sys.crystal
path = q_space_path(cryst, qs, 401)
res = Sunny.intensities(sc, path; energies, kT = langevin.kT  )
fig = plot_intensities(res; units, title="LLD AFM chain Sqw plot")
display(fig)

