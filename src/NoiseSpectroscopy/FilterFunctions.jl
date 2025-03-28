function dipole_field(sys,q,z)
    cryst = Sunny.orig_crystal(sys)
    V2d = abs(det(cryst.latvecs[1:2,1:2]))
    # q_global = cryst.recipvecs * q
    q_global = q
    λ = norm(q_global) # assume ω << c
    H=(exp(-λ*z)/2V2d)*[(q_global[1]^2)/λ  q_global[1]*q_global[2]/λ   im*q_global[1];
    q_global[1]*q_global[2]/λ   q_global[2]^2/λ im*q_global[2];
    im*q_global[1]  im*q_global[2]  -norm(q)];
    return H
    # qpts = convert(AbstractQPoints, qpts)
    # Nq = length(qpts.qs)
    # for (iq, q) in enumerate(qpts.qs)
        # q_global = cryst.recipvecs * q
    # end
end

function momentum_filter(sys,q,z)
    sys.dims[3] == 1 || error("System not two-dimensional in (a₁, a₂). Noise spectroscopy not available yet for 3d crystals.")
    V2d = abs(det(cryst.latvecs[1:2,1:2]))
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
    return W*V2d*(2*0.6745817653/0.05788381806)^2 
end

function noise_spectral_function(sys,ω)
end
