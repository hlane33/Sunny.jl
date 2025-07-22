# # SW02 - AFM Heisenberg chain BUT with measure changed
#
# This is a Sunny port of [SpinW Tutorial
# 2](https://spinw.org/tutorials/02tutorial), originally authored by Bjorn Fak
# and Sandor Toth. It calculates the spin wave spectrum of the antiferromagnetic
# Heisenberg nearest-neighbor spin chain.

# Load Sunny and the GLMakie plotting package.

using Sunny, GLMakie

# Define the chemical cell for a 1D chain following the [previous tutorial](@ref
# "SW01 - FM Heisenberg chain").

units = Units(:meV, :angstrom)

# Lattice (1D chain along x)
a = 1.0  # Lattice constant
latvecs = lattice_vectors(a, 10*a, 10*a, 90, 90, 90)
crystal = Crystal(latvecs, [[0, 0, 0]])  # 1 spin per unit cell

# System with AFM-compatible supercell
sys = System(crystal, [1 => Moment(; s=1/2, g=2)], :dipole; dims=(2, 1, 1))

# AFM exchange coupling (J1 > 0)
J1 = 1.0  # meV
set_exchange!(sys, J1, Bond(1, 1, [1, 0, 0]))  # Nearest-neighbor AFM

# Repeat supercell to desired length (e.g., 20 spins = 10 supercells)
Lx = 10  # Number of unit cells (each with 2 spins)
repeat_periodically(sys, (Lx, 1, 1))  # Now has 20 spins

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys; ndims=2, ghost_radius=8)

# Perform a [`SpinWaveTheory`](@ref) calculation for a path between ``[0,0,0]``
# and ``[1,0,0]`` in RLU.

swt = SpinWaveTheory(sys; measure=ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false))
energies = 0:0.05:5
qs = [[0,0,0], [1,0,0]]
cryst = sys.crystal
path = q_space_path(cryst, qs, 401)
res = intensities(swt, path; energies, kernel =gaussian(fwhm=0.25))
fig = plot_intensities(res; units, title="SWT AFM chain Sqw plot")
display(fig)

"""

# This system includes two bands that are fully degenerate in their dispersion.

isapprox(res.disp[1, :], res.disp[2, :])

# Plot the intensities summed over the two degenerate bands using the [Makie
# `lines` function](https://docs.makie.org/stable/reference/plots/lines).

xs = [q[1] for q in path.qs]
ys = res.data[1, :] + res.data[2, :]
lines(xs, ys; axis=(; xlabel="[H, 0, 0]", ylabel="Intensity", yscale=log10))
"""