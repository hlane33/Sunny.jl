"""
     dipole_field(sys,qs,z)

Calculates the dipolar field at distance z from a 2d magnetic plane. qs is provided in the r.l.u of the crystal and converted to the global frame. The crystal is assumed
to be defined such that a,b ⟂ z. This function assumes that ω << c so that the momentum and frequency filter functions can be separated.
"""

function dipole_field(sys,qs,z)
    out = []
    for q ∈ qs
        cryst = Sunny.orig_crystal(sys)
        V2d = abs(det(cryst.latvecs[1:2,1:2]))
        q_global = cryst.recipvecs * q
        λ = norm(q_global) # assume ω << c
        H=(exp(-λ*z)/(2V2d))*[(q_global[1]^2)/λ  (q_global[1]*q_global[2])/λ   im*q_global[1]
        (q_global[1]*q_global[2])/λ   (q_global[2]^2)/λ im*q_global[2]
        im*q_global[1]  im*q_global[2]  -norm(q_global)]
        push!(out,H)
    end
    return out
end

"""
     noise_spectral_function(sys,ω,n,z)

Calculates the noise spectral function at distance z from a 2d magnetic plane. qs is provided in the r.l.u of the crystal. The crystal is assumed
to be defined such that a,b ⟂ z. The momentum filter function is included in this definition with the appropriate terms for an NV oriented along the
direction n = (nx,ny,nz). The dipolar field diverges at q=[0,0,0], currently we skip this point. In future we can pick some sensible value here. The 
true value will depend on the size/shape of the sample. The integral is performed on a grid in the 2d Brillouin zone. Currently a Nq × Nq grid is built
to cover the 1BZ. In future we will want to sample more finely at small q. We can probably restrict the integral to a small area around q=[0,0,0]. 
"""

function noise_spectral_function(sys,ω,n,z;Nq = 10,kT=0)
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
                        buff[α,β] += nν*nμ*Dμα[qi][μ,α]*Dνβ[qi][ν,β]*Sqw[qi][α,β]
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

function phi_sq(sys,τ,z,n;dω=0.05, ωmax=1.0, f = RamseyFilter,N=nothing,kT = 0)
    ωgrid = dω:dω:ωmax #discontinuous at ω=0, so start at dω
    integral_grid = zeros(ComplexF64,length(ωgrid))
    for (ωi,ω) in enumerate(ωgrid)
        integral_grid[ωi] = f(ω,τ;N)[1]*noise_spectral_function(sys,ω,n,z;kT)
    end
    return real((1/2π)*sum(integral_grid)*dω)
end

#########################
# Test functions 

function momentum_filter_test(sys,q,z)
    Dμα = dipole_field(sys,q,z)
    Dνβ = dipole_field(sys,-q,z)
    cryst = Sunny.orig_crystal(sys)
    # println(norm(cryst.recipvecs * q[1])*exp(-norm(cryst.recipvecs * q[1])*z)/2)
    q_glob = cryst.recipvecs * q[1]
    te=-(1/2)*exp(-norm(q_glob)*z)*norm(q_glob)
    # println("Expected dipole zz term: $te")
    # dipoleval = Dμα[1][3,3]
    # println("Dipole zz term: $dipoleval" )
    W= Dμα[1][3,3]*Dνβ[1][3,3]
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