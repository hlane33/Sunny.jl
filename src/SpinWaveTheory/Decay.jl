using Sunny, GLMakie, LinearAlgebra

function print_something()
    print("Do something")
end


function non_interacting_greens_function(ω, swt::SpinWaveTheory, q_reshaped;ϵ=0.0001)
    H = Sunny.dynamical_matrix(swt,q_reshaped)
    L = Sunny.natoms(swt.sys.crystal)
    A = diagm([ones(2L)...,-ones(2L)...])
    return inv( -(ω+im*ϵ)*A + H)
end

function V_first_level(swt::SpinWaveTheory)
    (; sys, data) = swt
    N = 3
    # (; local_rotations, stevens_coefs, sqrtS) = data
    M = 1
    for (i, int) in enumerate(sys.interactions_union)
        for coupling in int.pair
            # at this level we want to calculate Vij(123) Vα(123)
            # put definition for V1m(i,j) into expression for Vij(1,2,3)
            (; isculled, bond) = coupling
            isculled && break

            @assert i == bond.i
            j = bond.j
            V1 = zeros(ComplexF64,N) #may need to replace with N-1 
            V2 = zeros(ComplexF64,N,N,N)
            Ṽ = zeros(ComplexF64,N)

            phase = exp(2π*im * dot(q_reshaped, bond.n)) # Phase associated with periodic wrapping

            # Set "general" pair interactions of the form Aᵢ⊗Bⱼ. Note that Aᵢ
            # and Bᵢ have already been transformed according to the local frames
            # of sublattice i and j, respectively.
            for (Ai, Bj) in coupling.general.data 
                for m in 1:N-1, n in 1:N-1
                    c = -0.5Ai[N,N] * Bj[N,n] 
                    d = Ai[N,1:N] * Bj[m,n]
                    V1[m] += c
                    V2[:,m,n] += d
                end
            end
            V123 = √(M)*(V1[σ3]*delta_function(α1,j)*delta_function(α2,j)*delta_function(α3,j) 
            +V1[σ3])
        end

    end
    
end

function delta_function(α1,α2)
    if α1 == α2
        return 1
    else
        return 0
    end
end

 V1,V2 = V_first_level(swt)
