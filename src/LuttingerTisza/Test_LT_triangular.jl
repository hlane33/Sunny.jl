# Test LT
import Pkg
Pkg.activate("C:\\Users\\hlane34\\.julia\\dev\\Sunny.jl")  #step needed until we add to dependences properly


using Sunny
using Profile
################################################################################
# Model
################################################################################
begin
    a, b, c = 1.0, 1.0, 4.0
    latvecs = lattice_vectors(a, b, c, 90, 90, 120)
    # positions = [[0,0,0],[0,0,0.1]]
    positions = [[0,0,0]]
    cryst = Crystal(latvecs, positions, 1)
    print_symmetry_table(cryst,1.5)

    dims = (1,1,1)
    S1 = 1
    sys    = System(cryst, dims, [SpinInfo(1; S=S1, g=1)], :dipole)

    J = 1.0
    K = 0.2
    h = 0


    J₁= [J+K 0 0;
         0 J 0;
         0 0 J];

    J₂= [J 0 0;
         0 J+K 0;
         0 0 J];

    J₃= [J 0 0;
         0 J 0;
         0 0 J+K];


    set_exchange!(sys, J₁, Bond(1, 1, [-1,0,0]))
    set_exchange!(sys, J₂, Bond(1, 1, [0,-1,0]))
    set_exchange!(sys, J₃, Bond(1, 1, [-1,-1,0]))


end
################################################################################
# Find ground state 
################################################################################
begin
    Δt = 0.02
    λ = 0.1
    langevin = Langevin(Δt; kT=0, λ)

    randomize_spins!(sys)
    for kT in range(5, 0, 30_000)
        langevin.kT = kT
        step!(sys, langevin)
    end

end
langevin.kT = 0.0
for _ ∈ 1:20_000
    step!(sys, langevin)
end

################################################################################
# Calculate SWT
################################################################################
print_wrapped_intensities(sys)


q_opt, val, vect = Sunny.Min_Q_new(sys,10)
Qvectors, Evecs =  Sunny.suggest_initial_parameters(sys,20)

Qvectors
Evecs

Qvectors_in = [Qvectors[2],Qvectors[3],Qvectors[4]]
Evecs_in = [Evecs[2],Evecs[3],Evecs[4]]


L=6

@time res, vectors, minim, A_value, B_value, T_matrix1 =Sunny.optimize_fourier_components(sys,Qvectors_in,L,Evecs/sqrt(3),1e4;time_limit=5000.0)

Qs_list=[]
for arr in vectors[:,1]
    push!(Qs_list,sum([arr...] .*Q))
end
