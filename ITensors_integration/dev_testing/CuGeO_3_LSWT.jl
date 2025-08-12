using ITensors, ITensorMPS, GLMakie, Sunny
include("sunny_toITensor.jl")

sys = create_dimerized_spin_chain(N; a=4.2, s=0.5, J1=13.79, J2=4.83, periodic_bc=false)
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys; ndims=2, ghost_radius=8)

# Perform a [`SpinWaveTheory`](@ref) calculation for a path between ``[0,0,0]``
# and ``[1,0,0]`` in RLU.

swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
energies = 0:0.05:5
qs = [[0,0,0], [1,0,0]]
cryst = sys.crystal
path = q_space_path(cryst, qs, 401)
res = Sunny.intensities(swt, path; energies, kernel =gaussian(fwhm=0.25))
fig = plot_intensities(res; units, title="SWT AFM J1-J2-delta chain")
display(fig)