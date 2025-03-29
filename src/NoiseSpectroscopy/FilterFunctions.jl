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
        H=(exp(-λ*z)/2V2d)*[(q_global[1]^2)/λ  q_global[1]*q_global[2]/λ   im*q_global[1];
        q_global[1]*q_global[2]/λ   q_global[2]^2/λ im*q_global[2];
        im*q_global[1]  im*q_global[2]  -norm(q)];
        push!(out,H)
    end
    return out
end
"""
     momentum_filter(sys,qs,z)

Calculates the momentum filter function at distance z from a 2d magnetic plane. qs is provided in the r.l.u of the crystal. The crystal is assumed
to be defined such that a,b ⟂ z. Deprecated function.
"""
function momentum_filter(sys,qs,z)
    out = []
    sys.dims[3] == 1 || error("System not two-dimensional in (a₁, a₂). Noise spectroscopy not available yet for 3d crystals.")
    V2d = abs(det(cryst.latvecs[1:2,1:2]))
    for q ∈ qs
        Hq = dipole_field(sys,q,z)
        Hmq = dipole_field(sys,-q,z)
        W = zeros(ComplexF64,3,3,3,3)
        for α ∈ 1:3 # DSSF component
            for β ∈ 1:3 # DSSF component
                for μ ∈ 1:3 # NV component
                    for ν ∈ 1:3 # NV component
                        W[μ,ν,α,β] = Hq[μ,α]*Hmq[ν,β] 
                    end
                end
            end
        end
        push!(out,W*V2d*(2*0.6745817653/0.05788381806)^2)
    end
    return out
end

function noise_spectral_function(sys,ω,n,z)
    dq = 1/10
    qs = -1/2 : dq : 1/2 - dq 
    qgrid =  [[qx,qy,0] for qx ∈ qs, qy ∈ qs]
    Nqs = length(qgrid)
    nhat = norm(n)
    measure = ssf_custom((q, ssf) -> ssf,sys;  apply_g=false)
    swt = SpinWaveTheory(sys;measure)
    kernel = lorentzian(fwhm=0.1)
    res = intensities(swt, qgrid[:]; energies=[ω], kernel)
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
            bad_q = qgrid[qi]
            println("NaN at $bad_q")
        else
            Nαβ[:,:,qi] .= buff
        end
    end
    out = (1/Nqs)*(2*0.6745817653)^2*sum(Nαβ)
    return out
end


begin CPMGFilter(ωs,τ,N)
    W = zeros(Float64,length(ωs))
    if isodd(N)
        f(ω,τ,N) = (16/(ω^2))*((sin(ω*τ/(4*N)))^4/(cos(ω*t/(2*N)))^2)*cos(ω*τ/2)^2
    else
        f(ω,τ,N) = (16/(ω^2))*((sin(ω*τ/(4*N)))^4/(cos(ω*t/(2*N)))^2)*sin(ω*τ/2)^2
    end
    for (ωi,ω) ∈ enumerate(ωs)
        W[ωi] = f(ω,τ,N)
    end
    return W 
end

begin RamseyFilter(ωs,τ,N)
    W = zeros(Float64,length(ωs))
    for (ωi,ω) ∈ enumerate(ωs)
        W[ωi] = (4*sin(ω*τ/2)^2)/ω^2
    end
    return W 
end

function phi_sq(sys,τ,z,n;dω=0.05, ωmax=1.0, f = RamseyFilter,τ=1,N=nothing)
    ωgrid = 0:dω:ωmax
    integral_grid = zeros(Float64,length(ωgrid))
    for (ωi,ω) in enumerate(ωgrid)
        integral_grid[ωi] = f(ω,τ,N)*noise_spectral_function(sys,ω,n,z)
    end
    return (1/2π)*sum(integral_grid)
end

res = noise_spectral_function(sys,0.1,[0,0,1.],0.02)