using Sunny, GLMakie, LinearAlgebra, Combinatorics, ColorSchemes

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

@inline function idx(α, σ, N)
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

function get_realspace_vertices!(swt::SpinWaveTheory,V31p,V31m,V32p,V32m,Von)
    (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = 1
    V31p .= 0
    V31m .= 0
    V32p .= 0
    V32m .= 0
    Von .= 0
    for (i, int) in enumerate(sys.interactions_union)
        for coupling in int.pair
            (; isculled, bond) = coupling
            isculled && break
            j = bond.j
            for (Ai, Bj) in coupling.general.data 
                for σ1 ∈ 1:N-1
                    V31p[σ1] += -0.5Ai[N,N] * Bj[N,σ1]
                    V31m[σ2] += -0.5Bj[N,N] * Ai[N,σ1]
                    for σ2 ∈ 1:N-1, σ3 ∈ 1:N-1
                        V32p[σ1,σ2,σ3] += Ai[N,σ1]* (Bj[σ2,σ3]-δ(σ2,σ3)*Bj[N,N]) 
                        V32m[σ1,σ2,σ3] += Bj[N,σ1]* (Ai[σ2,σ3]-δ(σ2,σ3)*Ai[N,N])    
                    end
                end
            end
            # op = int.onsite
            # for σ ∈ 1:N-1
            #     Von[σ] += op[N,σ]
            # end
        end
    end
end


function get_cubic_vertices!(swt,q1,q2,q3,W,Es,V31p,V31m,V32p,V32m,Von)
    qs = (q1,a2,q3,-q1,-q2,-q3)
    for (qi,q) ∈ enumerate(qs)
        excitations!(view(W[:,:,qi]),view(Es[:,qi]),swt, q)
    end
    for (i,int) in enumerate(sys.interactions_union)
        for coupling in int.pair
            (; isculled, bond) = coupling
            isculled && break
            j = bond.j
            phase3 = exp(2π*im * dot(q3, bond.n)) # Phase associated with periodic wrapping
            V[j,j,j,σ1,σ1,σ3] += V31_p[σ3]
            V[i,i,i,σ1,σ1,σ3] += V31_n[σ3]
            V[j,j,i,σ1,σ1,σ3] += V32_p[σ3,σ1,σ2]*conj(phase3)
            V[i,i,j,σ1,σ1,σ3] += V32_m[σ3,σ1,σ2]*phase3

        end
        op = int.onsite
        for σ ∈ 1:N-1
            V[i,i,i,σ1,σ2,σ3] += 0.5op[N,σ3] 
        end
    end

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
                @inbounds for α1 in 1:Na, α2 in 1:Na, α3 in 1:Na
                    @inbounds for σ1 in 1:N-1, σ2 in 1:N-1, σ3 in 1:N-1
                        α = (α1, α2, α3)
                        σ = (σ1, σ2, σ3)
                        add_Ṽ123(α, σ, i, j, Ai, Bj, phase, Ṽ, M,N)
                    end
                end
            end
            op = int.onsite
            @inbounds for α1 in 1:Na, α2 in 1:Na, α3 in 1:Na
                @inbounds for σ1 in 1:N-1, σ2 in 1:N-1, σ3 in 1:N-1
                    α = (α1, α2, α3)
                    σ = (σ1, σ2, σ3)
                    add_Ṽ123_onsite(α, σ, i, op,  Ṽ, M,N)
                end
            end
        end
    end
    return Ṽ
end

for j ∈ 1:38
    println(sys.interactions_union[1].pair[j].general.data)
end

function vertex_diag_precalc(swt::SpinWaveTheory,q1,q2,q3)
    (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = 1
    Nf = Na*(N-1)
    qs = (q1,q2,q3,-q1,-q2,-q3)
    Ṽ_list = zeros(ComplexF64, Na, Na, Na, N-1, N-1, N-1,6)
    # Ts = zeros(ComplexF64,2Nf,2Nf,length(qs))
    energies_list = zeros(ComplexF64,2Nf,6)
    W11s = zeros(ComplexF64,Nf,Nf,6)
    W12s = zeros(ComplexF64,Nf,Nf,6)
    W21s = zeros(ComplexF64,Nf,Nf,6)
    W22s = zeros(ComplexF64,Nf,Nf,6)
    for (qi,q) ∈ enumerate(qs)
        energies, T = excitations(swt,q)
        @views begin
        W11s[:,:,qi] .= T[1:Nf,1:Nf]
        W21s[:,:,qi] .= T[(Nf+1):end,1:Nf]
        W12s[:,:,qi] .= T[1:Nf,(Nf+1):end]
        W22s[:,:,qi] .= T[(Nf+1):end,(Nf+1):end] #should be trivial but it seems like this is a little weird
        energies_list[:,qi] .= energies
        # Ts[:,:,qi] .= T
        end
    end
    qtriplets = ((q1,q2,q3),(q3,q2,q1),(q2,q1,q3),
    (-q1,-q2,-q3),(-q3,-q2,-q1),(-q3,-q1,-q2))
    for (qi,qt) in enumerate(qtriplets) 
        @views Ṽ_list[:,:,:,:,:,:,qi]  = vertex_initial(swt,qt[1],qt[2],qt[3]) 
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
        @views v1 = Ṽ_list[:,:,:,:,:,:,1]
        @views v2 = Ṽ_list[:,:,:,:,:,:,2]
        @views v3 = Ṽ_list[:,:,:,:,:,:,3]
        @views v4 = Ṽ_list[:,:,:,:,:,:,4]
        @views v5 = Ṽ_list[:,:,:,:,:,:,5]
        @views v6 = Ṽ_list[:,:,:,:,:,:,6]
       @inbounds for α1 ∈ 1:Na, α2 ∈ 1:Na, α3 ∈ 1:Na 
        @inbounds for σ1 ∈ 1:N-1, σ2 ∈ 1:N-1, σ3 ∈ 1:N-1
            ind1 = idx(α1, σ1, N)
            ind2 = idx(α2, σ2, N)
            ind3 = idx(α3, σ3, N)
            @inbounds for n1 ∈ 1:Nf, n2 ∈ 1:Nf, n3  ∈ 1:Nf
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
        @views v1 = Ṽ_list[:,:,:,:,:,:,1]
        @views v5 =  Ṽ_list[:,:,:,:,:,:,5]
       @inbounds for α1 ∈ 1:Na, α2 ∈ 1:Na, α3 ∈ 1:Na 
        @inbounds for σ1 ∈ 1:N-1, σ2 ∈ 1:N-1, σ3 ∈ 1:N-1
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


function self_energy(swt::SpinWaveTheory,q;dq = 0.2,ϵ=0.3)
     (; sys, data) = swt
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = 1
    Nf = Na*(N-1)
    Σa = zeros(ComplexF64,Nf)
    Σb = zeros(ComplexF64,Nf) 
    ks = Sunny.make_q_grid(sys, dq)
    for (ki,k) ∈ enumerate(ks)
        VS1, VS2, energies = vertex_diag(swt,-q,k,q-k)
        @inbounds @simd for n1 ∈ 1:Nf
            e1 =  energies[n1,4]
            @inbounds for n2 ∈ 1:Nf
                e2 = energies[n2,2]
                @inbounds for n3 ∈ 1:Nf
                    e3 = energies[n3,3]
                    v1 = VS1[n1,n2,n3]
                    v2 = VS2[n1,n2,n3]
                    Σa[n1] +=  ((v1*conj(v1)))/(e1-e2-e3+im*ϵ) # 4 because -q is q1 so +q = 4
                    Σb[n1] +=  ((v2*conj(v2)))/(e1+e2+e3-im*ϵ)
                end
            end
        end
    end
    Σa_sum, Σb_sum = 0.5Σa/length(ks),-0.5Σb/length(ks)
    Σ = diagm(vcat(Σa_sum + Σb_sum,Σa_sum + Σb_sum))
    return Σ
end

function interacting_greens_function(ωs, swt::SpinWaveTheory, qpts;ϵ=0.3,dq = 0.5)
    (; sys, data) = swt
    qpts = convert(Sunny.AbstractQPoints, qpts)
    Na = Sunny.natoms(sys.crystal)
    N = sys.Ns[1]
    M = 1
    Nf = Na*(N-1)
    G = zeros(ComplexF64,length(qpts.qs),length(ωs),2Nf,2Nf)
    for (iω,ω) ∈ enumerate(ωs)
        @Threads.threads for iq in 1:length(qpts.qs)
            q = qpts.qs[iq]
            G0 = non_interacting_greens_function(ω,swt,q;ϵ)
            Σ = self_energy(swt,q;dq,ϵ)
            G[iq,iω,:,:] .= inv(inv(G0)-Σ)
        end
        println("finished energy: $iω")
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
        end
    end
    return G
end





#################################################
#################################################
Ṽ_list, W11s, W12s, W21s, W22s, energies_list = vertex_diag_precalc(swt,rand(3),rand(3),rand(3))

a=W21s[:,:,1]
b=conj(W12s[:,:,4])

W11s[:,:,1] 



qs = [[0,0,0], [1,-0.5,0]]
path = q_space_path(cryst, qs, 50)
ωs = range(1,4,50)
Ga = @time interacting_greens_function(ωs, swt, path;ϵ=0.05,dq = 0.5)


sqw = zeros(Float64,length(path.qs),length(ωs))
for iq ∈ 1:length(path.qs)
    for iω ∈ 1:length(ωs)
        sqw[iq,iω] = -imag.(tr(Ga[iq,iω,:,:]))
    end
end
begin
    fig = Figure()
    cr = (0,30)
    cm = :viridis
    ax = Axis(fig[1,1];title = "Tr[G(q,ω)] (coarse integration) NLSWT (3T)",xticks = path.xticks)
    heatmap!(ax,1:50,ωs,sqw,colormap=cm)
    Colorbar(fig[1,2],colorrange=cr,colormap=cm)
    fig
end


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
Sigs = []

dqs = [0.5,0.25,0.2,0.1,0.075]
for dq in dqs 
    Sig = self_energy(swt,rand(3);dq,ϵ=0.05)
    push!(Sigs,Sig)
    println("dq = $dq done")
end

val = [sum(norm.(Sigs[n] .+ Sigs[n])) for n ∈ 1:5]
vs1,vs2, en = vertex_diag(swt,rand(3),rand(3),rand(3))

extrema(norm.(vs1))
extrema(norm.(vs2))

begin
    fig = Figure()
    ax = fig[1,1]
    plot()
    fig
end

q1s = [[0.1*(x-1),0,0] for x ∈ 1:10]
varray = [zeros(ComplexF64,8,8,8) for i in 1:10]
for (q1i,q1) ∈ enumerate(q1s)
        vv1,vv2,en  = vertex_diag(swt,q1,[0,0,0.],[0,0.0,0.0])
        varray[q1i] .= vv1
end


data = norm.(varray[1].-varray[2])
data2 = norm.(varray[1])
volume(data2)