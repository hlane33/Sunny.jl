# Test LT
import Pkg
Pkg.activate("C:\\Users\\hlane34\\.julia\\dev\\Sunny.jl")  #step needed until we add to dependences properly

using Sunny

################################################################################
# Model
################################################################################
begin
    a, b, c = 1.0, 2.0, 3.0
    latvecs = lattice_vectors(a, b, c, 90, 90, 90)
    # positions = [[0,0,0],[0,0,0.1]]
    positions = [[0,0,0]]
    cryst = Crystal(latvecs, positions, 1)

    dims = (1,1,1)
    S1 = 1
    S2 = 1
    sys    = System(cryst, dims, [SpinInfo(1; S=S1, g=1)], :dipole)

    J = -1.0
    D = 0.3
    h = 0


    J₁= [J 0 0;
         0 J D;
         0 -D J];
    

    J₂=0.0*J₁
    J₃=0.0*J₁
    μ = -0.2

    set_exchange!(sys, J₁, Bond(1, 1, [-1,0,0]))
    # set_exchange!(sys, J₂, Bond(1, 2, [0,0,0]))
    # set_exchange!(sys, J₃, Bond(2, 2, [-1,0,0]))
    set_external_field!(sys, [0.0, 0.0, h])
    S = spin_operators(sys, 1)
    set_onsite_coupling!(sys, -μ*S[3]^2, 1)


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

q_opt, val, vect = Sunny.Min_Q(sys)
atan(-D/J)
-(q_opt[1][1]-1)*2*pi # picked up a degenerate wavevector so we need to do some work!
@assert round(atan(-D/J),digits=6) ≈ round(-(q_opt[1][1]-1)*2*pi,digits=6)
# check this transformation is legit:
@assert abs(Sunny.lambda_min(sys,[-(q_opt[1][1]-1),0,0],1)[1]-Sunny.lambda_min(sys,q_opt[1],1)[1]) < 1e-7

####################################################################

Evals = fill(zeros(ComplexF64,3), length(unique_els))
Qs, Energies, EigV = Sunny.Min_Q(sys)
Qvectors, Evals =  Sunny.suggest_initial_parameters(sys)
L=8

@time res, vectors, minim=Sunny.optimize_fourier_components(sys,Qvectors,L,Evals,1000.0;iterations = 5)

T=Sunny.build_empty_T_matrix(Qvectors,L)
unique_els = Sunny.get_unique_elements(T,L)
empty_list = fill(zeros(ComplexF64,length(initial_guess[1])),length(unique_els))
init_dict = Dict(zip(unique_els,empty_list))
 dims = Tuple([length(unique_els)])
for q=1:length(Primary_Qs)
    address= zeros(Int64,length(Primary_Qs))
    address[q]=1
    init_dict[Tuple(address)]=initial_guess[q]
end
Sunny.update_T_matrix!(T,init_dict)

Sunny.ℒ(sys,Qvectors,L,T,0.0)
