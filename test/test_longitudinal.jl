using Sunny, LinearAlgebra, GLMakie, Random
units = Units(:K, :angstrom)

function make_bcc_system(; L=20, Nc=10)
    J1 = -100.0
    a = 2.8665
    crystal = Sunny.bcc_crystal(; a)
    sys_conv = System(crystal, [1 => Moment(; s=1, g=2)], :dipole)
    set_exchange!(sys_conv, J1, Bond(1, 2, [0, 0, 0]))
    sys_prim = reshape_supercell(sys_conv, primitive_cell(crystal))
    return crystal, repeat_periodically(sys_prim, (L, L, Nc))
end

crystal, sys = make_bcc_system(;)

plot_spins(sys)

# Integrator 
dt = 0.00012
damping = 0.1

# Tc = 1.232J for alpha = 0.032pi
# Tc = 1.4J for alpha = 0.25pi
# Tc = 1.6J for alpha = 0.602pi
# Tc = 2.05333J for alpha = 0.602pi

kT = 112.0
langevin = Langevin(dt; kT, damping)
suggest_timestep(sys, langevin; tol=1e-2)

################################################################################
# Set up itinerancy parameters
################################################################################
J0 = 800
#αs = range(0.001, 3π/4, 6) |> collect

α = 0.602π
a = J0/(1+tan(α))
b = a*tan(α) 

################################################################################
# Set up SampledCorrelations 
################################################################################
energies = range(0.0, 2000.0, 150)
measure = ssf_perp(sys)

integrator = LongitudinalLangevin(; damping, kT, A=a/2, B=b/4, C=0, planck_statistics=sys);
# integrator = Langevin(dt; damping, kT)
polarize_spins!(sys, [0, 0, 1])

# typeof(integrator)
# Sunny.step!(sys, integrator)
# plot_spins(sys)

sc = SampledCorrelations(sys; dt, measure, energies, integrator)
sc.measperiod

polarize_spins!(sys, [0, 0, 1])
# Thermalize
@time for _ in 1:1000
    Sunny.step!(sys, integrator)
end

@time for _ in 1:10
    for _ in 1:100 
        Sunny.step!(sys, integrator)
    end
    add_sample!(sc, sys)
end

qs = q_space_path(crystal,[[1/2, 1/2, 0], [0, 0, 0], [1, 0, 0], [1/2, 1/2, 1/2], [0, 0, 0]], 200)
res = intensities(sc, qs; energies, kT=kT)
plot_intensities(res; saturation=0.6,colormap=:coolwarm)

