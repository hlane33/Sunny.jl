using Sunny, GLMakie, LinearAlgebra, Combinatorics

PERMUTATIONS2 = collect(permutations(1:3))
PERMUTATIONS1 = [[1,li...] for li in permutations(2:3)]

function non_interacting_greens_function(ω, swt::SpinWaveTheory, q;ϵ=0.0001)
    H = Sunny.dynamical_matrix(swt,q)
    L = Sunny.natoms(swt.sys.crystal)
    A = diagm([ones(2L)...,-ones(2L)...])
    return inv( -(ω+im*ϵ)*A + H)
end
@inline δ(x, y) = (x==y)

# function idx(α, σ, N)
#     return α + (N - 1) * (σ - 1)
# end

function idx(α, σ, N)
    return σ + (N - 1) * (α - 1)
end



function add_Ṽ123(α::NTuple{3,Int}, σ::NTuple{3,Int}, 
                   i, j, Ai, Bj, phase, Ṽ, M, N)
    α1, α2, α3 = α
    σ1, σ2, σ3 = σ
    Ṽ[α1, α2, α3, σ1, σ2, σ3] += √(M) * (
        -0.5*δ(α1,j)*δ(α2,j)*δ(α3,j)*δ(σ1,σ2)*Ai[N,N]*Bj[N,σ3]
        -0.5*δ(α1,i)*δ(α2,i)*δ(α3,i)*δ(σ1,σ2)*Ai[N,σ3]*Bj[N,N]
        +δ(α1,j)*δ(α2,j)*δ(α3,i)*conj(phase[3])*Ai[N,σ3]* (Bj[σ1,σ2]
        # -δ(σ1,σ2)*Bj[N,N]     # this factor appears in notes
        )
        +δ(α1,i)*δ(α2,i)*δ(α3,j)*phase[3]*Bj[N,σ3]*(Ai[σ1,σ2]
        # -δ(σ1,σ2)*Ai[N,N]  # this factor appears in notes
        )   )
end


function add_Ṽ123_onsite( α::NTuple{3,Int}, σ::NTuple{3,Int}, 
                   i, op, Ṽ, M, N)
    α1, α2, α3 = α
    σ1, σ2, σ3 = σ

    Ṽ[α1, α2, α3, σ1, σ2, σ3] += 1/(2√(M))*op[N,σ3]* δ(α1,i)* δ(α2,i)* δ(α3,i)*δ(σ1,σ2)
end


function vertex_initial(swt::SpinWaveTheory,q1,q2,q3)
    (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = 1
    Ṽ = zeros(ComplexF64, Na, Na, Na, N-1, N-1, N-1)
    for (i, int) in enumerate(sys.interactions_union)
        for coupling in int.pair
            (; isculled, bond) = coupling
            isculled && break
            j = bond.j
            phase1 = exp(2π*im * dot(q1, bond.n)) # Phase associated with periodic wrapping
            phase2 = exp(2π*im * dot(q2, bond.n)) # Phase associated with periodic wrapping
            phase3 = exp(2π*im * dot(q3, bond.n)) # Phase associated with periodic wrapping
            phase=[phase1,phase2,phase3]
            for (Ai, Bj) in coupling.general.data 
                for α1 in 1:Na, α2 in 1:Na, α3 in 1:Na
                    for σ1 in 1:N-1, σ2 in 1:N-1, σ3 in 1:N-1
                        α = (α1, α2, α3)
                        σ = (σ1, σ2, σ3)
                        add_Ṽ123(α, σ, i, j, Ai, Bj, phase, Ṽ, M,N)
                    end
                end
            end
            op = int.onsite
            for α1 in 1:Na, α2 in 1:Na, α3 in 1:Na
                for σ1 in 1:N-1, σ2 in 1:N-1, σ3 in 1:N-1
                    α = (α1, α2, α3)
                    σ = (σ1, σ2, σ3)
                    add_Ṽ123_onsite(α, σ, i, op,  Ṽ, M,N)
                end
            end
        end
    end
    return Ṽ
end

function vertex_diag_precalc(swt::SpinWaveTheory,q1,q2,q3)
    (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = 1
    Nf = Na*(N-1)
    qs = [q1,q2,q3,-q1,-q2,-q3]
    Ṽ_list = zeros(ComplexF64, Na, Na, Na, N-1, N-1, N-1,length(qs))
    # Ts = zeros(ComplexF64,2Nf,2Nf,length(qs))
    energies_list = zeros(ComplexF64,2Nf,length(qs))
    W11s = zeros(ComplexF64,Nf,Nf,length(qs))
    W12s = zeros(ComplexF64,Nf,Nf,length(qs))
    W21s = zeros(ComplexF64,Nf,Nf,length(qs))
    W22s = zeros(ComplexF64,Nf,Nf,length(qs))
    for (qi,q) ∈ enumerate(qs)
        energies, T = excitations(swt,q)
        W11s[:,:,qi] .= T[1:Nf,1:Nf]
        W21s[:,:,qi] .= T[(Nf+1):end,1:Nf]
        W12s[:,:,qi] .= T[1:Nf,(Nf+1):end]
        W22s[:,:,qi] .= T[(Nf+1):end,(Nf+1):end] #should be trivial but it seems like this is a little weird
        energies_list[:,qi] .= energies
        # Ts[:,:,qi] .= T
    end
    qtriplets = [(q1,q2,q3),(q3,q2,q1),(q2,q1,q3),
    (-q1,-q2,-q3),(-q3,-q2,-q1),(-q3,-q1,-q2)]
    for (qi,qt) in enumerate(qtriplets)
        V = vertex_initial(swt,qt[1],qt[2],qt[3]) 
        Ṽ_list[:,:,:,:,:,:,qi]  = V
    end
    return Ṽ_list, W11s, W12s, W21s, W22s, energies_list
end

function vertex_diag_extract1(swt,Ṽ_list, W11s, W12s, W21s, W22s)
       (; sys, data) = swt
        Na = Sunny.natoms(sys.crystal)
        N = sys.Ns[1]
        M = N
        Nf = Na*(N-1)
        V1out = zeros(ComplexF64,Nf,Nf,Nf)
        v1 = Ṽ_list[:,:,:,:,:,:,1]
        v2 = Ṽ_list[:,:,:,:,:,:,2]
        v3 = Ṽ_list[:,:,:,:,:,:,3]
        v4 = Ṽ_list[:,:,:,:,:,:,4]
        v5 = Ṽ_list[:,:,:,:,:,:,5]
        v6 = Ṽ_list[:,:,:,:,:,:,6]
       for α1 ∈ 1:Na, α2 ∈ 1:Na, α3 ∈ 1:Na 
        for σ1 ∈ 1:N-1, σ2 ∈ 1:N-1, σ3 ∈ 1:N-1
            ind1 = idx(α1, σ1, N)
            ind2 = idx(α2, σ2, N)
            ind3 = idx(α3, σ3, N)
            for n1 ∈ 1:Nf, n2 ∈ 1:Nf, n3  ∈ 1:Nf
            V1out[n1,n2,n3] += v1[α1, α2, α3,σ1, σ2, σ3] * W22s[ind1,n1,1] * W11s[ind2,n2,2] * W11s[ind3,n3,3]  # Fix the 3 indices
                 +  v2[α1, α2, α3,σ1, σ2, σ3] * W21s[ind1,n3,3] * W11s[ind2,n2,2] * W12s[ind3,n1,1]
                 +  v3[α1, α2, α3,σ1, σ2, σ3] * W21s[ind1,n2,2] * W12s[ind2,n1,1] * W11s[ind3,n3,3]
                
                 +  conj(v4[α1, α2, α3,σ1, σ2, σ3]) * conj(W21s[ind1,n1,4]) * conj(W12s[ind2,n2,5]) * conj(W12s[ind3,n3,6])
                 +  conj(v5[α1, α2, α3,σ1, σ2, σ3]) * conj(W22s[ind1,n3,6]) * conj(W12s[ind2,n2,5]) * conj(W11s[ind3,n1,4])
                 +  conj(v6[α1, α2, α3,σ1, σ2, σ3]) * conj(W22s[ind1,n3,6]) * conj(W11s[ind2,n1,4]) * conj(W12s[ind3,n2,5])
            end
        end
    end
    return V1out
end

function vertex_diag_extract2(swt,Ṽ_list, W11s, W12s, W21s, W22s)
       (; sys, data) = swt
        Na = Sunny.natoms(sys.crystal)
        N = sys.Ns[1]
        M = 1
        Nf = Na*(N-1)
        V2out = zeros(ComplexF64,Nf,Nf,Nf)
        v1 = Ṽ_list[:,:,:,:,:,:,1]
        v5 = Ṽ_list[:,:,:,:,:,:,5]
       for α1 ∈ 1:Na, α2 ∈ 1:Na, α3 ∈ 1:Na 
        for σ1 ∈ 1:N-1, σ2 ∈ 1:N-1, σ3 ∈ 1:N-1
            ind1 = idx(α1, σ1, N)
            ind2 = idx(α2, σ2, N)
            ind3 = idx(α3, σ3, N)
            for n1 ∈ 1:Nf, n2 ∈ 1:Nf, n3  ∈ 1:Nf
                V2out[n1,n2,n3] +=  v1[α1, α2, α3,σ1, σ2, σ3] * W22s[ind1,n1,1] * W12s[ind2,n2,2] * W12s[ind3,n3,3]
                                        +conj(v5[α1, α2, α3,σ1, σ2, σ3]) * conj(W21s[ind1,n3,6]) * conj(W11s[ind2,n2,5]) * conj(W11s[ind3,n1,4])  
            end
        end
    end
    return V2out
end

function vertex_diag(swt::SpinWaveTheory,q1,q2,q3)
    (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = 1
    Nf = Na*(N-1)
    qs = [q1,q2,q3]

    Ṽ_list, W11s, W12s, W21s, W22s, Hs = vertex_diag_precalc(swt,q1,q2,q3)
    VS1 = zeros(ComplexF64,Nf,Nf,Nf)
    VS2 = zeros(ComplexF64,Nf,Nf,Nf)
    for (pi,perm) ∈ enumerate(PERMUTATIONS1)
        q1, q2, q3 = qs[perm]
        V1 = vertex_diag_extract1(swt,Ṽ_list, W11s, W12s, W21s, W22s)
        VS1+= permutedims(V1,perm)
    end
        for (pi,perm) ∈ enumerate(PERMUTATIONS2)
        q1, q2, q3 = qs[perm]
        V2 = vertex_diag_extract2(swt,Ṽ_list, W11s, W12s, W21s, W22s)
        VS2+= permutedims(V2,perm)
    end
    return VS1, VS2, Hs

end


function self_energy(swt::SpinWaveTheory,q;dq = 0.2,ϵ=0.05)
     (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = 1
    Nf = Na*(N-1)
    ks = Sunny.make_q_grid(sys, dq)
    Σa = zeros(ComplexF64,Nf)
    Σb = zeros(ComplexF64,Nf) 
    for (ki,k) ∈ enumerate(ks)
        VS1, VS2, energies = vertex_diag(swt,-q,k,q-k)
        for n1 ∈ 1:Nf, n2 ∈ 1:Nf, n3 ∈ 1:Nf
            Σa[n1] +=  ((VS1[n1,n2,n3]*conj(VS1[n1,n2,n3]))^2)/(energies[n1,4]-energies[n2,2]-energies[n3,3]+im*ϵ) # 4 because -q is q1 so +q = 4
            Σb[n1] +=  ((VS2[n1,n2,n3]*conj(VS2[n1,n2,n3]))^2)/(energies[n1,4]+energies[n2,2]+energies[n3,3]-im*ϵ)
        end
    end
    Σa_sum, Σb_sum = 0.5Σa/length(ks),-0.5Σb/length(ks)
    Σ = diagm(vcat(Σa_sum + Σb_sum,Σa_sum + Σb_sum))
    return Σ
end

function interacting_greens_function(ωs, swt::SpinWaveTheory, qpts;ϵ=0.01,dq = 0.5)
    (; sys, data) = swt
    qpts = convert(Sunny.AbstractQPoints, qpts)
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = 1
    Nf = Na*(N-1)
    ks = Sunny.make_q_grid(sys, dq)
    G = zeros(ComplexF64,length(qpts.qs),length(ωs),2Nf,2Nf)
    for (iω,ω) ∈ enumerate(ωs)
        @Threads.threads for iq in 1:length(qpts.qs)
            q = qpts.qs[iq]
            G0 = non_interacting_greens_function(ω,swt,q;ϵ)
            Σ = self_energy(swt,q;dq,ϵ)
            println("self energy")
            println(diag(Σ))
            G[iq,iω,:,:] .= inv(inv(G0)-Σ)
        end
    end
    return G
end


function greens_function_V2(ωs, swt::SpinWaveTheory, qpts;ϵ=0.01)
    qpts = convert(Sunny.AbstractQPoints, qpts)
    G = zeros(ComplexF64,length(qpts.qs),length(ωs),2Nf,2Nf)
    for (iω,ω) ∈ enumerate(ωs)
        @Threads.threads for iq in 1:length(qpts.qs)
            q = qpts.qs[iq]
            G0 = non_interacting_greens_function(ω,swt,q;ϵ)
            G[iq,iω,:,:] .= G0
            self_energy(swt,q;dq,ϵ)
        end
    end
    return G
end



path = q_space_path(cryst,[[0,0,0],[0.5,0,0]],10)
ωs = range(3.,5.,10)
Ga = interacting_greens_function(ωs, swt, path;ϵ=0.1,dq = 0.5)
sqw = zeros(Float64,length(path.qs),length(ωs))
for iq ∈ 1:length(path.qs)
    for iω ∈ 1:length(ωs)
        sqw[iq,iω] = imag.(tr(Ga[iq,iω,:,:]))
    end
end
heatmap(range(0,1,40),ωs,sqw)

path = q_space_path(cryst,[[0,0,0],[1,0,0]],50)
ωs = range(0.,6,100)
G = greens_function_V2(ωs, swt::SpinWaveTheory, path;ϵ=0.1)
sqw = zeros(Float64,length(path.qs),length(ωs))
for iq ∈ 1:length(path.qs)
    for iω ∈ 1:length(ωs)
        sqw[iq,iω] = imag.(tr(G[iq,iω,:,:]))
    end
end

heatmap(sqw)
Sigsa = []
Sigsb = []

dqs = [0.5,0.25,0.2]
for dq in dqs 
    Siga, Sigb = self_energies(swt,rand(3);dq,ϵ=0.05)
    push!(Sigsa,Siga)
    push!(Sigsb,Sigb)
    println("dq = $dq done")
end

val = [sum(norm.(Sigsa[n] .+ Sigsb[n])) for n ∈ 1:3]
vs1,vs2, en = vertex_diag(swt,rand(3),rand(3),rand(3))

extrema(norm.(vs1))
extrema(norm.(vs2))

begin
    fig = Figure()
    ax = fig[1,1]
    plot()
    fig
end
######################################################################################################################################################
######################################################################################################################################################


function Vns_DEPRECATED(swt::SpinWaveTheory,q_reshaped1,q_reshaped2,q_reshaped3,perm)
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
        push!(Ṽ_list, V[(1,2,3)])
    end
    for α1 ∈ 1:Na, α2 ∈ 1:Na, α3 ∈ 1:Na 
        for σ1 ∈ 1:N-1, σ2 ∈ 1:N-1, σ3 ∈ 1:N-1
            for n1 ∈ 1:Nf, n2 ∈ 1:Nf, n3  ∈ 1:Nf
            V1out[n1,n2,n3] += Ṽ_list[1][α1, α2, α3,σ1, σ2, σ3] * W22s[1][ind1,n1] * W11s[2][ind2,n2] * W11s[3][ind3,n3]  # Fix the 3 indices
                 +  Ṽ_list[2][α1, α2, α3,σ1, σ2, σ3] * W21s[3][ind1,n3] * W11s[2][ind2,n2] * W12s[1][ind3,n1]
                 +  Ṽ_list[3][α1, α2, α3,σ1, σ2, σ3] * W21s[2][ind1,n2] * W12s[1][ind2,n1] * W11s[3][ind3,n3]
                
                 +  Ṽ_list[4][α1, α2, α3,σ1, σ2, σ3] * conj(W21s[4][ind1,n1]) * conj(W12s[5][ind2,n2]) * conj(W12s[6][ind3,n3])
                 +  Ṽ_list[5][α1, α2, α3,σ1, σ2, σ3] * conj(W22s[6][ind1,n3]) * conj(W12s[5][ind2,n2]) * conj(W11s[4][ind3,n1])
                 +  Ṽ_list[6][α1, α2, α3,σ1, σ2, σ3] * conj(W22s[6][ind1,n3]) * conj(W11s[4][ind2,n1]) * conj(W12s[5][ind3,n2])

            V2out[n1,n2,n3] =  Ṽ_list[1][α1, α2, α3,σ1, σ2, σ3] * W22s[1][ind1,n1] * W12s[2][ind2,n2] * W12s[3][ind3,n3]
                        +conj(Ṽ_list[6][α1, α2, α3,σ1, σ2, σ3]) * conj(W21s[6][ind1,n3]) * conj(W11s[5][ind2,n2]) * conj(W12s[4][ind3,n1])  
            end
        end
    end
    return V1out, V2out 
end


function Vtildes_DEPRECATED(swt::SpinWaveTheory,q_reshaped1,q_reshaped2,q_reshaped3)
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
                        for perm in PERMUTATIONS
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
                    for perm in PERMUTATIONS
                        add_to_Ṽ_onsite(Tuple(perm), α, σ, i, op,  Ṽ, M,N)
                    end
                end
            end
        end
    end
    return Ṽ
end


# Define a helper function to compute the contributions
function add_to_Ṽ_DEPRECATED(perm123::NTuple{3,Int}, α::NTuple{3,Int}, σ::NTuple{3,Int}, 
                   i, j, Ai, Bj, phase, Ṽ, M, N)

    α1, α2, α3 = α[perm123[1]], α[perm123[2]], α[perm123[3]]
    σ1, σ2, σ3 = σ[perm123[1]], σ[perm123[2]], σ[perm123[3]]

    Ṽ[perm123][α1, α2, α3, σ1, σ2, σ3] += √(M) * (
        -0.5*δ(α1,j)*δ(α2,j)*δ(α3,j)*
             δ(σ1,σ2)*Ai[N,N]*Bj[N,σ3]
        -0.5*δ(α1,i)*δ(α2,i)*δ(α3,i)*
             δ(σ1,σ2)*Ai[N,σ3]*Bj[N,N]
        +δ(α1,j)*δ(α2,j)*δ(α3,j)*
             conj(phase[3])*Ai[N,σ3]*Bj[σ1,σ2]
        +δ(α1,j)*δ(α2,j)*δ(α3,j)*
             phase[3]*Ai[N,σ3]*Bj[σ1,σ2]
    )
end

function add_to_Ṽ_onsite_DEPRECATED(perm123::NTuple{3,Int}, α::NTuple{3,Int}, σ::NTuple{3,Int}, 
                   i, op, Ṽ, M, N)
    α1, α2, α3 = α[perm123[1]], α[perm123[2]], α[perm123[3]]
    σ1, σ2, σ3 = σ[perm123[1]], σ[perm123[2]], σ[perm123[3]]

    Ṽ[perm123][α1, α2, α3, σ1, σ2, σ3] += 1/(2√(M))*op[N,σ3]* δ(α1,i)* δ(α2,i)* δ(α3,i)*δ(σ1,σ2)
end


function Vns_light_DEPRECATED(swt::SpinWaveTheory,q_reshaped1,q_reshaped2,q_reshaped3)
    (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = N
    Nf = Na*(N-1)
    V1out = zeros(ComplexF64,Nf,Nf,Nf)
    V2out = zeros(ComplexF64,Nf,Nf,Nf)
    qs = [q_reshaped1,q_reshaped2,q_reshaped3,-q_reshaped1,-q_reshaped2,-q_reshaped3]
    Ṽ_list = zeros(ComplexF64, Na, Na, Na, N-1, N-1, N-1,length(qs))
    # Ts = zeros(ComplexF64,2Nf,2Nf,length(qs))
    # Hs = zeros(ComplexF64,2Nf,2Nf,length(qs))
    W11s = zeros(ComplexF64,Nf,Nf,length(qs))
    W12s = zeros(ComplexF64,Nf,Nf,length(qs))
    W21s = zeros(ComplexF64,Nf,Nf,length(qs))
    W22s = zeros(ComplexF64,Nf,Nf,length(qs))
    for (qi,q) ∈ enumerate(qs)
        H, T = excitations(swt,q)
        W11s[:,:,qi] .= T[1:Nf,1:Nf]
        W21s[:,:,qi] .= T[Nf+1:end,1:Nf]
        W12s[:,:,qi] .= T[1:Nf,Nf+1:end]
        W22s[:,:,qi] .= T[Nf+1:end,Nf+1:end]
        # Hs[:,:,qi] .= H
        # Ts[:,:,qi] .= T
    end
    qtriplets = [(q_reshaped1,q_reshaped2,q_reshaped3),(q_reshaped3,q_reshaped2,q_reshaped1),(q_reshaped2,q_reshaped1,q_reshaped3),
    (-q_reshaped1,-q_reshaped2,-q_reshaped3),(-q_reshaped3,-q_reshaped2,-q_reshaped1),(-q_reshaped3,-q_reshaped1,-q_reshaped2)]
    for (qi,qt) in enumerate(qtriplets)
        V = Vtildes_light(swt,qt[1],qt[2],qt[3]) 
        Ṽ_list[:,:,:,:,:,:,qi]  = V
    end
    for α1 ∈ 1:Na, α2 ∈ 1:Na, α3 ∈ 1:Na 
        for σ1 ∈ 1:N-1, σ2 ∈ 1:N-1, σ3 ∈ 1:N-1
            for n1 ∈ 1:Nf, n2 ∈ 1:Nf, n3  ∈ 1:Nf
            V1out[n1,n2,n3] += Ṽ_list[:,:,:,:,:,:,1][α1, α2, α3,σ1, σ2, σ3] * W22s[ind1,n1,1] * W11s[ind2,n2,2] * W11s[ind3,n3,3]  # Fix the 3 indices
                 +  Ṽ_list[:,:,:,:,:,:,2][α1, α2, α3,σ1, σ2, σ3] * W21s[ind1,n3,3] * W11s[ind2,n2,2] * W12s[ind3,n1,1]
                 +  Ṽ_list[:,:,:,:,:,:,3][α1, α2, α3,σ1, σ2, σ3] * W21s[ind1,n2,2] * W12s[ind2,n1,1] * W11s[ind3,n3,3]
                
                 +  Ṽ_list[:,:,:,:,:,:,4][α1, α2, α3,σ1, σ2, σ3] * conj(W21s[ind1,n1,4]) * conj(W12s[ind2,n2,5]) * conj(W12s[ind3,n3,6])
                 +  Ṽ_list[:,:,:,:,:,:,5][α1, α2, α3,σ1, σ2, σ3] * conj(W22s[ind1,n3,6]) * conj(W12s[ind2,n2,5]) * conj(W11s[ind3,n1,4])
                 +  Ṽ_list[:,:,:,:,:,:,6][α1, α2, α3,σ1, σ2, σ3] * conj(W22s[ind1,n3,6]) * conj(W11s[ind2,n1,4]) * conj(W12s[ind3,n2,5])

            V2out[n1,n2,n3] +=  Ṽ_list[:,:,:,:,:,:,1][α1, α2, α3,σ1, σ2, σ3] * W22s[ind1,n1,1] * W12s[ind2,n2,2] * W12s[ind3,n3,3]
                        +conj(Ṽ_list[:,:,:,:,:,:,6][α1, α2, α3,σ1, σ2, σ3]) * conj(W21s[ind1,n3,6]) * conj(W11s[ind2,n2,5]) * conj(W12s[ind3,n1,4])  
            end
        end
    end
    return V1out, V2out 
end
