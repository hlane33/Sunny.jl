"""
     dipole_field(sys,qs,z)

Calculates the dipolar field at distance z from a 2d magnetic plane. qs is provided in the r.l.u of the crystal and converted to the global frame. The crystal is assumed
to be defined such that a,b ⟂ z. This function assumes that ω << c so that the momentum and frequency filter functions can be separated.
"""

function dipole_field(sys,qs,z)
    H = zeros(ComplexF64,3,3,length(qs))
    for (qi,q) ∈ enumerate(qs)
        cryst = Sunny.orig_crystal(sys)
        V2d = abs(det(cryst.latvecs[1:2,1:2]))
        q_global = cryst.recipvecs * q
        λ = norm(q_global) # assume ω << c
        H[:,:,qi]=(exp(-λ*z)/(2V2d))*[(q_global[1]^2)/λ  (q_global[1]*q_global[2])/λ   im*q_global[1]
        (q_global[1]*q_global[2])/λ   (q_global[2]^2)/λ im*q_global[2]
        im*q_global[1]  im*q_global[2]  -norm(q_global)]
    end
    return H
end

"""
     noise_spectral_function(sys,ωs,n,z)

Calculates the noise spectral function at distance z from a 2d magnetic plane. qs is provided in the r.l.u of the crystal. The crystal is assumed
to be defined such that a,b ⟂ z. The momentum filter function is included in this definition with the appropriate terms for an NV oriented along the
direction n = (nx,ny,nz). The dipolar field diverges at q=[0,0,0], currently we skip this point. In future we can pick some sensible value here. The 
true value will depend on the size/shape of the sample. The integral is performed on a grid in the 2d Brillouin zone. Currently a Nq × Nq grid is built
to cover the 1BZ. In future we will want to sample more finely at small q. We can probably restrict the integral to a small area around q=[0,0,0]. 
"""



function noise_spectral_function_LLD(sys,n,z;Nq = 10,kT=0,dt = 0.1,nsamples = 3, ndwell=100,ωmax=2.0,dω=0.1)
    dq = 1/Nq
    qs = -1/2 : dq : 1/2 - dq 
    qgrid =  [[qx,qy,0] for qx ∈ qs, qy ∈ qs]
    Nqs = length(qgrid)
    nhat = norm(n)
    measure = ssf_custom((q, ssf) -> ssf,sys;  apply_g=false)
    kT == 0 ? error("Please provide finite temperature for LLD.") : nothing
    langevin = Langevin(dt; damping=0.2, kT)
    pos_ωs = 0:dω:ωmax
    cryst = Sunny.orig_crystal(sys)
    V2d = abs(det(cryst.latvecs[1:2,1:2]))
    sc = SampledCorrelations(sys; dt, energies=pos_ωs, measure)
    for _ in 1:nsamples
        for _ in 1:ndwell
            step!(sys, langevin)
        end
        add_sample!(sc, sys)
    end
    res = intensities(sc, qgrid[:]; energies=:available_with_negative, kT)
    Dμα = Sunny.dipole_field(sys,qgrid[:],z)
    Dνβ = Sunny.dipole_field(sys,-qgrid[:],z)
    Sqw = res.data
    ωs = available_energies(sc; negative_energies=true)     
    Nαβ = zeros(ComplexF64,3,3,length(qgrid),length(ωs))
    for (wi,w) ∈ enumerate(ωs)
        for qi ∈ 1:length(qgrid)
            buff = zeros(ComplexF64,3,3)
            for (μ,nμ) ∈ enumerate(nhat)
                for (ν,nν) ∈ enumerate(nhat)
                    for α = 1:3
                        for β = 1:3
                            buff[α,β] += nν*nμ*Dμα[μ,α,qi]*Dνβ[ν,β,qi]*Sqw[wi,qi][α,β]
                        end
                    end
                end
            end
            if isnan(sum(buff))
                # bad_q = qgrid[qi]
                # println("NaN at $bad_q")
            else
                Nαβ[:,:,qi,wi] .= buff
            end
        end
    end
    out = V2d*(1/Nqs)*(2*0.6745817653)^2*sum(Nαβ;dims = 1:3)[1,1,1,:]
    return out, ωs
end
function noise_spectral_function_LSWT(sys,ωs,n,z;Nq = 10,kT=0,Γ=0.025)
    dq = 1/Nq
    qs = -1/2 : dq : 1/2 - dq 
    qgrid =  [[qx,qy,0] for qx ∈ qs, qy ∈ qs]
    Nqs = length(qgrid)
    nhat = norm(n)
    measure = ssf_custom((q, ssf) -> ssf,sys;  apply_g=false)
    kernel = lorentzian(fwhm=Γ)
    swt = SpinWaveTheory(sys;measure)
    res = intensities(swt, qgrid[:]; energies=ωs, kernel,kT)
    Dμα = Sunny.dipole_field(sys,qgrid[:],z)
    Dνβ = Sunny.dipole_field(sys,-qgrid[:],z)
    Sqw = res.data
    Nαβ = zeros(ComplexF64,3,3,length(qgrid),length(ωs))
    for (wi,w) ∈ enumerate(ωs)
        for qi ∈ 1:length(qgrid)
            buff = zeros(ComplexF64,3,3)
            for (μ,nμ) ∈ enumerate(nhat)
                for (ν,nν) ∈ enumerate(nhat)
                    for α = 1:3
                        for β = 1:3
                            buff[α,β] += nν*nμ*Dμα[μ,α,qi]*Dνβ[ν,β,qi]*Sqw[wi,qi][α,β]
                        end
                    end
                end
            end
            if isnan(sum(buff))
                # bad_q = qgrid[qi]
                # println("NaN at $bad_q")
            else
                Nαβ[:,:,qi,wi] .= buff
            end
        end
    end
    out = (1/Nqs)*(2*0.6745817653)^2*sum(Nαβ;dims = 1:3)[1,1,1,:]
    return out
end


"""
     CPMGFilter(ωs,τ,N)

Frequency filter function evaluated at frequencies ωs and time τ for CPMG pulse sequence of N π pulses.
"""

function CPMGFilter(ωs,τ;N=1)
    W = zeros(Float64,length(ωs))
    if isodd(N)
        for (ωi,ω) in enumerate(ωs)
            W[ωi] = (16/(ω^2))*((sin(ω*τ/(4*N)))^4/(cos(ω*τ/(2*N)))^2)*cos(ω*τ/2)^2
        end
    else
        for (ωi,ω) in enumerate(ωs)
            W[ωi] = (16/(ω^2))*((sin(ω*τ/(4*N)))^4/(cos(ω*τ/(2*N)))^2)*sin(ω*τ/2)^2
        end
    end
    return W 
end

"""
     RamseyFilter(ωs,τ)

Frequency filter function evaluated at frequencies ωs and time τ for Ramsey pulse sequence.
"""

function RamseyFilter(ωs,τ;N=nothing)
    W = zeros(Float64,length(ωs))
    for (ωi,ω) ∈ enumerate(ωs)
        W[ωi] = (4*sin(ω*τ/2)^2)/ω^2
    end
    return W 
end

"""
     phi_sq(sys,τ,z,n;dω=0.05, ωmax=1.0, f = RamseyFilter,τ=1,N=nothing)

<ϕ²> for a pulse sequence defined by the function f. The integral is performed over a grid in
frequencies with step dω
"""


function phi_sq(sys,τ,z,n;method="LSWT",Nq=10,kT = 0.0,nsamples = 2, ndwell = 100,ωmax=2.0,dω=0.1,f=RamseyFilter,N=nothing,Γ=0.05,dt =0.1  )
    if method == "LSWT"
        ωgrid = -ωmax:dω:ωmax #discontinuous at ω=0, so start at small offset
        δ = 1e-6 # small offset to get rid of discontinuity at [0,0,0]
        integral_grid = f(ωgrid.+δ,τ;N).*noise_spectral_function_LSWT(sys,ωgrid,n,z;Nq,kT,Γ)
        out = real((1/2π)*sum(integral_grid)*dω)
    elseif method == "LLD"
        δ = 1e-6 # small offset to get rid of discontinuity at [0,0,0]
        Nw,ωs = noise_spectral_function_LLD(sys,n,z;Nq,kT,dt,nsamples, ndwell,ωmax,dω)
        integral_grid = f(ωs.+δ,τ;N).*Nw
        out = real((1/2π)*sum(integral_grid)*(1/length(ωs)))
    else
        error("Provide valid method")
    end
    return out
end


#########################
# Test functions 

function momentum_filter_test(sys,q,z)
    Dμα = dipole_field(sys,q,z)
    Dνβ = dipole_field(sys,-q,z)
    cryst = Sunny.orig_crystal(sys)
    q_glob = cryst.recipvecs * q[1]
    te=-(1/2)*exp(-norm(q_glob)*z)*norm(q_glob)
    # println("Expected dipole zz term: $te")
    # dipoleval = Dμα[1][3,3]
    # println("Dipole zz term: $dipoleval" )
    W= Dμα[3,3,1]*Dνβ[3,3,1]
    out = ((2*0.6745817653/0.05788381806)^2)*W
    return out
end

function noise_test(sys,z,w)
    dq = 1/30
    qs = -1/2 : dq : 1/2 - dq 
    d = 0.001
    qgrid =  [[qx+d,qy,0] for qx ∈ qs, qy ∈ qs]
    sumlist = []
    for q in qgrid
        W = momentum_filter_test(sys,[q],z)
        push!(sumlist,W*Sqw_test(sys,q,w))
    end
    out=(1/length(qgrid))*sum(sumlist)
    return out
end

function Sqw_test(sys,q,w)
    cryst = Sunny.orig_crystal(sys)
    q_global = cryst.recipvecs * q
    A = 1.0
    B = 1.0
    return (2A*norm(q_global)^2)/(w^2 +(B*norm(q_global)^2)^2)
end

##########################################
# Old functions
function noise_spectral_function_old(sys,ω,n,z;Nq = 10,kT=0)
    dq = 1/Nq
    qs = -1/2 : dq : 1/2 - dq 
    qgrid =  [[qx,qy,0] for qx ∈ qs, qy ∈ qs]
    Nqs = length(qgrid)
    nhat = norm(n)
    measure = ssf_custom((q, ssf) -> ssf,sys;  apply_g=false)
    swt = SpinWaveTheory(sys;measure)
    kernel = lorentzian(fwhm=0.1)
    res = intensities(swt, qgrid[:]; energies=[ω], kernel,kT)
    Dμα = dipole_field(sys,qgrid[:],z)
    Dνβ = dipole_field(sys,-qgrid[:],z)
    Sqw = res.data
    Nαβ = zeros(ComplexF64,3,3,length(qgrid))
    for qi ∈ 1:length(qgrid)
        buff = zeros(ComplexF64,3,3)
        for (μ,nμ) ∈ enumerate(nhat)
            for (ν,nν) ∈ enumerate(nhat)
                for α = 1:3
                    for β = 1:3
                        buff[α,β] += nν*nμ*Dμα[μ,α,qi]*Dνβ[ν,β,qi]*Sqw[qi][α,β]
                    end
                end
            end
        end
        if isnan(sum(buff))
            # bad_q = qgrid[qi]
            # println("NaN at $bad_q")
        else
            Nαβ[:,:,qi] .= buff
        end
    end
    out = (1/Nqs)*(2*0.6745817653)^2*sum(Nαβ)
    return out
end

function phi_sq_old(sys,τ,z,n;dω=0.05, ωmax=1.0, f = RamseyFilter,N=nothing,kT = 0)
    ωgrid = -ωmax:dω:ωmax #discontinuous at ω=0, so start at small offset
    δ = 1e-6
    integral_grid = zeros(ComplexF64,length(ωgrid))
    for (ωi,ω) in enumerate(ωgrid)
        integral_grid[ωi] = f(ω+δ,τ;N)[1]*noise_spectral_function(sys,ω+δ,n,z;kT)
    end
    return real((1/2π)*sum(integral_grid)*dω)
end

function noise_spectral_function(sys,ωs,n,z;method="LSWT",Nq = 10,kT=0,dt = 0.1,nsamples = 3, ndwell=100,Γ=0.025)
    dq = 1/Nq
    qs = -1/2 : dq : 1/2 - dq 
    qgrid =  [[qx,qy,0] for qx ∈ qs, qy ∈ qs]
    Nqs = length(qgrid)
    nhat = norm(n)
    measure = ssf_custom((q, ssf) -> ssf,sys;  apply_g=false)
    kernel = lorentzian(fwhm=Γ)
    if method == "LSWT"
        swt = SpinWaveTheory(sys;measure)
        res = intensities(swt, qgrid[:]; energies=ωs, kernel,kT)
    elseif method == "LLD"
        kT == 0 ? error("Please provide finite temperature for LLD.") : nothing
        langevin = Langevin(dt; damping=0.2, kT)
        ωmax = round(ωs[end];digits=3)
        dω = round(ωs[end]-ωs[end-1];digits=3)
        pos_ωs = 0:dω:ωmax
        sc = SampledCorrelations(sys; dt, energies=pos_ωs, measure)
        for _ in 1:nsamples
            for _ in 1:ndwell
                step!(sys, langevin)
            end
            add_sample!(sc, sys)
        end
        res = intensities(sc, qgrid[:]; energies=:available_with_negative, kT)
    else
        error("Please provide a valid method: LLD or LSWT, with appropriate kwargs.")
    end
    Dμα = Sunny.dipole_field(sys,qgrid[:],z)
    Dνβ = Sunny.dipole_field(sys,-qgrid[:],z)
    Sqw = res.data
    if method=="LLD"
        ωs = available_energies(sc; negative_energies=true)
    end
    Nαβ = zeros(ComplexF64,3,3,length(qgrid),length(ωs))
    for (wi,w) ∈ enumerate(ωs)
        for qi ∈ 1:length(qgrid)
            buff = zeros(ComplexF64,3,3)
            for (μ,nμ) ∈ enumerate(nhat)
                for (ν,nν) ∈ enumerate(nhat)
                    for α = 1:3
                        for β = 1:3
                            buff[α,β] += nν*nμ*Dμα[μ,α,qi]*Dνβ[ν,β,qi]*Sqw[wi,qi][α,β]
                        end
                    end
                end
            end
            if isnan(sum(buff))
                # bad_q = qgrid[qi]
                # println("NaN at $bad_q")
            else
                Nαβ[:,:,qi,wi] .= buff
            end
        end
    end
    out = (1/Nqs)*(2*0.6745817653)^2*sum(Nαβ;dims = 1:3)[1,1,1,:]
    return out # This set of if statements is ugly. Only include until we test LSWT for detailed balance or work out a better way to deal with LLD for negative energy
end

function phi_sq_old2(sys,τ,z,n;dω=0.05, ωmax=1.0, f = RamseyFilter,N=nothing,kT = 0,method="LSWT",Γ=0.025)
    ωgrid = -ωmax:dω:ωmax #discontinuous at ω=0, so start at small offset
    δ = 1e-6 # small offset to get rid of discontinuity at [0,0,0]
    integral_grid = f(ωgrid.+δ,τ;N).*noise_spectral_function(sys,ωgrid,n,z;kT,method)
    return real((1/2π)*sum(integral_grid)*dω)
end