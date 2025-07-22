using Sunny, GLMakie, LinearAlgebra, Combinatorics


function non_interacting_greens_function(ω, swt::SpinWaveTheory, q_reshaped;ϵ=0.0001)
    H = Sunny.dynamical_matrix(swt,q_reshaped)
    L = Sunny.natoms(swt.sys.crystal)
    A = diagm([ones(2L)...,-ones(2L)...])
    return inv( -(ω+im*ϵ)*A + H)
end

function delta_function(α1,α2)
    if α1 == α2
        return 1
    else
        return 0
    end
end

# Define a helper function to compute the contributions
function add_to_Ṽ(perm123::NTuple{3,Int}, α::NTuple{3,Int}, σ::NTuple{3,Int}, 
                   i, j, Ai, Bj, phase, Ṽ, M, N)

    α1, α2, α3 = α[perm123[1]], α[perm123[2]], α[perm123[3]]
    σ1, σ2, σ3 = σ[perm123[1]], σ[perm123[2]], σ[perm123[3]]

    Ṽ[perm123][α1, α2, α3, σ1, σ2, σ3] += √(M) * (
        -0.5*delta_function(α1,j)*delta_function(α2,j)*delta_function(α3,j)*
             delta_function(σ1,σ2)*Ai[N,N]*Bj[N,σ3]
        -0.5*delta_function(α1,i)*delta_function(α2,i)*delta_function(α3,i)*
             delta_function(σ1,σ2)*Ai[N,σ3]*Bj[N,N]
        +delta_function(α1,j)*delta_function(α2,j)*delta_function(α3,j)*
             conj(phase[3])*Ai[N,σ3]*Bj[σ1,σ2]
        +delta_function(α1,j)*delta_function(α2,j)*delta_function(α3,j)*
             phase[3]*Ai[N,σ3]*Bj[σ1,σ2]
    )
end

function add_to_Ṽ_onsite(perm123::NTuple{3,Int}, α::NTuple{3,Int}, σ::NTuple{3,Int}, 
                   i, op, Ṽ, M, N)
    α1, α2, α3 = α[perm123[1]], α[perm123[2]], α[perm123[3]]
    σ1, σ2, σ3 = σ[perm123[1]], σ[perm123[2]], σ[perm123[3]]

    Ṽ[perm123][α1, α2, α3, σ1, σ2, σ3] += 1/(2√(M))*op[N,σ3]* delta_function(α1,i)* delta_function(α2,i)* delta_function(α3,i)*delta_function(σ1,σ2)
end

function Vtildes(swt::SpinWaveTheory,q_reshaped1,q_reshaped2,q_reshaped3)
    (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = N
    # Store Ṽ tensors indexed by permutations
    Ṽ = Dict{NTuple{3, Int}, Array{ComplexF64, 6}}()
    for perm in permutations(1:3)
        Ṽ[Tuple(perm)] = zeros(ComplexF64, Na, Na, Na, N-1, N-1, N-1)
    end
    for (i, int) in enumerate(sys.interactions_union)
        for coupling in int.pair
            # at this level we want to calculate Vij(123) Vα(123)
            # put definition for V1m(i,j) into expression for Vij(1,2,3)
            (; isculled, bond) = coupling
            isculled && break

            @assert i == bond.i
            j = bond.j

            phase1 = exp(2π*im * dot(q_reshaped1, bond.n)) # Phase associated with periodic wrapping
            phase2 = exp(2π*im * dot(q_reshaped2, bond.n)) # Phase associated with periodic wrapping
            phase3 = exp(2π*im * dot(q_reshaped3, bond.n)) # Phase associated with periodic wrapping
            phase=[phase1,phase2,phase3]

            # Set "general" pair interactions of the form Aᵢ⊗Bⱼ. Note that Aᵢ
            # and Bᵢ have already been transformed according to the local frames
            # of sublattice i and j, respectively.
            for (Ai, Bj) in coupling.general.data 
                for α1 in 1:Na, α2 in 1:Na, α3 in 1:Na
                    for σ1 in 1:N-1, σ2 in 1:N-1, σ3 in 1:N-1
                        α = (α1, α2, α3)
                        σ = (σ1, σ2, σ3)
                        for perm in permutations(1:3)
                            add_to_Ṽ(Tuple(perm), α, σ, i, j, Ai, Bj, phase, Ṽ, M,N)
                        end
                    end
                end
            end
            op = int.onsite
            for α1 in 1:Na, α2 in 1:Na, α3 in 1:Na
                for σ1 in 1:N-1, σ2 in 1:N-1, σ3 in 1:N-1
                    α = (α1, α2, α3)
                    σ = (σ1, σ2, σ3)
                    for perm in permutations(1:3)
                        add_to_Ṽ_onsite(Tuple(perm), α, σ, i, op,  Ṽ, M,N)
                    end
                end
            end
        end
    end
    return Ṽ
end

function Vns(swt::SpinWaveTheory,q_reshaped1,q_reshaped2,q_reshaped3,perm)
    (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = N
    Nf = Na*(N-1)
    V1out = zeros(ComplexF64,Nf,Nf,Nf)
    V2out = zeros(ComplexF64,Nf,Nf,Nf)
    Ṽ_list = []  
    Hs = []
    Ts = []
    W11s = []
    W12s = []
    W21s = []
    W22s = []
    qs = [q_reshaped1,q_reshaped2,q_reshaped3,-q_reshaped1,-q_reshaped2,-q_reshaped3]
      Ṽ = Dict{NTuple{3, Int}, Array{ComplexF64, 6}}()
    for perm in permutations(1:3)
        Ṽ[Tuple(perm)] = zeros(ComplexF64, Na, Na, Na, N-1, N-1, N-1)
    end
    

    for (qi,q) ∈ enumerate(qs)
        H, T = excitations(swt,q)
        W11 = T[1:Nf,1:Nf]
        W21 = T[Nf+1:end,1:Nf]
        W12 = T[1:Nf,Nf+1:end]
        W22 = T[Nf+1:end,Nf+1:end]
        push!(Hs,H)
        push!(Ts,T)
        push!(W11s,W11)
        push!(W12s,W12)
        push!(W21s,W21)
        push!(W22s,W22)
    end
    qtriplets = [(q_reshaped1,q_reshaped2,q_reshaped3),(q_reshaped3,q_reshaped2,q_reshaped1),(q_reshaped2,q_reshaped1,q_reshaped3),
    (-q_reshaped1,-q_reshaped2,-q_reshaped3),(-q_reshaped3,-q_reshaped2,-q_reshaped1),(-q_reshaped3,-q_reshaped1,-q_reshaped2)]
    for qt in qtriplets
        V = Vtildes(swt,qt[1],qt[2],qt[3]) 
        push!(Ṽ_list, V[perm])
    end
    for α1 ∈ 1:Na, α2 ∈ 1:Na, α3 ∈ 1:Na 
        for σ1 ∈ 1:N-1, σ2 ∈ 1:N-1, σ3 ∈ 1:N-1
            for n1 ∈ 1:Nf, n2 ∈ 1:Nf, n3  ∈ 1:Nf
            V1out[n1,n2,n3] += Ṽ_list[1][α1, α2, α3,σ1, σ2, σ3] * W22s[1][α1+(N-1)*σ1,n1] * W11s[2][α2+(N-1)*σ2,n2] * W11s[3][α3+(N-1)*σ3,n3]  # Fix the 3 indices
                 +  Ṽ_list[2][α1, α2, α3,σ1, σ2, σ3] * W21s[3][α1+(N-1)*σ1,n3] * W11s[2][α2+(N-1)*σ2,n2] * W12s[1][α3+(N-1)*σ3,n1]
                 +  Ṽ_list[3][α1, α2, α3,σ1, σ2, σ3] * W21s[2][α1+(N-1)*σ1,n2] * W12s[1][α2+(N-1)*σ2,n1] * W11s[3][α3+(N-1)*σ3,n3]
                
                 +  Ṽ_list[4][α1, α2, α3,σ1, σ2, σ3] * conj(W21s[4][α1+(N-1)*σ1,n1]) * conj(W12s[5][α2+(N-1)*σ2,n2]) * conj(W12s[6][α3+(N-1)*σ3,n3])
                 +  Ṽ_list[5][α1, α2, α3,σ1, σ2, σ3] * conj(W22s[6][α1+(N-1)*σ1,n3]) * conj(W12s[5][α2+(N-1)*σ2,n2]) * conj(W11s[4][α3+(N-1)*σ3,n1])
                 +  Ṽ_list[6][α1, α2, α3,σ1, σ2, σ3] * conj(W22s[6][α1+(N-1)*σ1,n3]) * conj(W11s[4][α2+(N-1)*σ2,n1]) * conj(W12s[5][α3+(N-1)*σ3,n2])

            V2out[n1,n2,n3] =  Ṽ_list[1][α1, α2, α3,σ1, σ2, σ3] * W22s[1][α1+(N-1)*σ1,n1] * W12s[2][α2+(N-1)*σ2,n2] * W12s[3][α3+(N-1)*σ3,n3]
                        +conj(Ṽ_list[6][α1, α2, α3,σ1, σ2, σ3]) * conj(W21s[6][α1+(N-1)*σ1,n3]) * conj(W11s[5][α2+(N-1)*σ2,n2]) * conj(W12s[4][α3+(N-1)*σ3,n1])  
            end
        end
    end
    return V1out, V2out 
end


@time Vns(swt,q_reshaped,q_reshaped,q_reshaped,[])
H, T =excitations(swt,q_reshaped)

T
function Ws(swt::SpinWaveTheory,q_reshaped)
    H, T = excitations(swt,q_reshaped) 
    Nf = size(T)[1]
    W11 = T[1:Nf,1:Nf]
    W21 = T[Nf+1:end,1:Nf]
    W12 = T[1:Nf,Nf+1:end]
    W22 = T[Nf+1:end,Nf+1:end]
    return W11,W12,W21,W22 
end




V = Vtildes(swt,q_reshaped,q_reshaped,q_reshaped)
V[(1,2,3)][1,1,1,1,1,1]

Vs = [V[(1,2,3)],V[(1,2,3)],V[(1,2,3)]]

Vs[1][1,1,1,1,1,1]