######Code that attempts to do FT for 2D systems with single unit cell ###
### currently fails
import Sunny


function accum_sample_2d!(qc::QuantumCorrelations, FT_params, linear_predict_params; assume_real_S=false)
    (; data, samplebuf) = qc
    
    # Slice params (hardcoded for Sz) in 2D
    # TODO make this flexible so that it can be adjusted based on corr choice
    obs_idx = 3
    corr_idx = 1
    
    # Extract 2D spatial slice: G[spatial_x, spatial_y, spatial time]
    G = samplebuf[obs_idx, :, :, :, 1, :]  # Shape: (Nx, Ny, Nt)
    println("Shape of G (2D): ", size(G))
    
    if assume_real_S
        out = compute_S_2d(G, FT_params; linear_predict_params)
    else
        out = compute_S_2d_complex(G, FT_params; linear_predict_params)
    end
    
    println("size(out): ", size(out))

    data_slice= qc.data[corr_idx, 1, 1, :, :, 1, :]
    print(size(data_slice))
    
    # Store result: out has shape (Nqx, Nqy, Nω)
    data_slice .= out
end

function compute_S_2d(G::AbstractArray{ComplexF64,4}, ft_params::FTParams; linear_predict_params::LinearPredictParams)
    
    qxs = ft_params.allowed_qxs
    qys = ft_params.allowed_qys  
    ωs = ft_params.energies
    positions = ft_params.positions
    ts = ft_params.ts
    
    Nx, Ny, Nt = size(G)
    total_sites = Nx * Ny
    
    @assert size(positions, 2) == total_sites "Position matrix should have $(total_sites) sites, got $(size(positions, 2))"
    

    # Precompute trigonometric terms for time
    cos_ωt = [cos(ω*t) for ω in ωs, t in ts]
    sin_ωt = [sin(ω*t) for ω in ωs, t in ts]
    
    # Precompute spatial phase factors using real positions from Sunny
    cos_qr = zeros(Float64, length(qxs), length(qys), Nx, Ny)
    
    site_idx = 1
    for xi in 1:Nx, yi in 1:Ny
        # Get real-space position from Sunny system
        rx, ry, rz = positions[:, site_idx]  # x, y, z coordinates in real space
        
        for (qxi, qx) in enumerate(qxs), (qyi, qy) in enumerate(qys)
            # q⃗·r⃗ = qx*rx + qy*ry (ignoring z for 2D)
            cos_qr[qxi, qyi, xi, yi] = cos(qx*rx + qy*ry)
        end
        
        site_idx += 1
    end
    
    # Main computation
    out = zeros(Float64, length(qxs), length(qys), length(ωs))
    
    @inbounds for (qxi, qx) in enumerate(qxs), (qyi, qy) in enumerate(qys), (ωi, ω) in enumerate(ωs)
        sum_val = 0.0
        
        for xi in 1:Nx, yi in 1:Ny, ti in 1:Nt
            # 2D spatial phase factor (already computed)
            spatial_phase = cos_qr[qxi, qyi, xi, yi]
            
            # Temporal Fourier component
            temporal_component = cos_ωt[ωi, ti] * real(G[xi, yi, 1, ti]) - 
                               sin_ωt[ωi, ti] * imag(G[xi, yi, 1, ti])
            
            sum_val += spatial_phase * temporal_component
        end
        
        out[qxi, qyi, ωi] = sum_val
    end
    
    return out
end

# Function to extract positions from Sunny system
function extract_positions_from_sunny(sys)
    # Get all site positions from the Sunny system
    # sys.crystal.positions contains fractional coordinates
    # sys.dims gives the system dimensions
    
    positions = []
    for site in Sunny.eachsite(sys)
        # Get real-space position for this site
        r = Sunny.global_position(sys, site)  # Returns [x, y, z] in Angstroms
        push!(positions, r)
    end
    
    # Convert to matrix format [3 x Nsites] for easier indexing
    pos_matrix = hcat(positions...)
    return pos_matrix
end
