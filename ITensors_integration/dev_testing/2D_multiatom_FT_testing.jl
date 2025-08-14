##### I was beginning to try and get the FT working for the 2D case ###
### Became very complex in the case of a multi atom basis where you need to account ####
# For cross site correlations and feed that into intensities for it to do form factors #
#THE MULTI ATOM COMPUTE_S DOES NOT CURRENTLY WORK

function accum_sample_2d!(qc::QuantumCorrelations, FT_params, linear_predict_params; assume_real_S=false, atom_indices=:all)
    (; data, samplebuf) = qc
    
    # Slice params (hardcoded for Sz) in 2D
    # TODO make this flexible so that it can be adjusted based on corr choice
    obs_idx = 3
    corr_idx = 1
    
    # Extract 2D spatial slice including all atoms in unit cell
    # samplebuf shape: [obs, spatial_x, spatial_y, spatial_z, atom_in_unit_cell, time]
    if atom_indices == :all
        G = samplebuf[obs_idx, :, :, :, :, :]  # Shape: (Nx, Ny, Nz, Natoms, Nt)
        println("Shape of G (all atoms): ", size(G))
    else
        G = samplebuf[obs_idx, :, :, :, atom_indices, :]  # Shape: (Nx, Ny, Nz, length(atom_indices), Nt)
        println("Shape of G (selected atoms): ", size(G))
    end
    
    if assume_real_S
        out = compute_S_2d_multiAtom(G, FT_params; linear_predict_params)
    else
        out = compute_S_2d_multiAtom_complex(G, FT_params; linear_predict_params)
    end
    
    println("size(out): ", size(out))
    
    # Store result: out has shape (Nqx, Nqy, Nω)
    #Currently fixes atomic indices within unit cell as 1,1
    qc.data[corr_idx, 1, 1, :, :, :, :] .= out
end

function compute_S_2d_multiAtom(G::AbstractArray{ComplexF64,5}, ft_params::FTParams2D; linear_predict_params::LinearPredictParams)
    
    qxs = ft_params.allowed_qxs
    qys = ft_params.allowed_qys  
    ωs = ft_params.energies
    positions = ft_params.positions  # [3 x Nsites] matrix from Sunny
    atom_positions = ft_params.atom_positions  # [3 x Natoms] positions of atoms in unit cell
    ts = ft_params.ts
    
    Nx, Ny, Nz, Natoms, Nt = size(G)
    total_sites = Nx * Ny * Nz
    
    @assert size(positions, 2) == total_sites "Position matrix should have $(total_sites) sites, got $(size(positions, 2))"
    @assert size(atom_positions, 2) == Natoms "Atom position matrix should have $(Natoms) atoms, got $(size(atom_positions, 2))"
    
    # Time series extension (extend along time dimension)
    if linear_predict_params.n_predict > 0
        ts, G_extended = extend_time_series_multiAtom(ts, G; linear_predict_params)
        Nt_extended = length(ts)
    else
        G_extended = G
        Nt_extended = Nt
    end
    
    # Precompute trigonometric terms for time
    cos_ωt = [cos(ω*t) for ω in ωs, t in ts]
    sin_ωt = [sin(ω*t) for ω in ωs, t in ts]
    
    # Precompute spatial phase factors for each unit cell and atom
    cos_qr = zeros(Float64, length(qxs), length(qys), Nx, Ny, Nz, Natoms)
    
    site_idx = 1
    for xi in 1:Nx, yi in 1:Ny, zi in 1:Nz
        # Get unit cell position from Sunny system
        unit_cell_pos = positions[:, site_idx]  # [x, y, z] of this unit cell
        
        for atom_idx in 1:Natoms
            # Total position = unit cell position + atom position within unit cell
            atom_pos = unit_cell_pos + atom_positions[:, atom_idx]
            rx, ry, rz = atom_pos[1], atom_pos[2], atom_pos[3]
            
            for (qxi, qx) in enumerate(qxs), (qyi, qy) in enumerate(qys)
                # q⃗·r⃗ = qx*rx + qy*ry (can include qz*rz if needed)
                cos_qr[qxi, qyi, xi, yi, zi, atom_idx] = cos(qx*rx + qy*ry)
            end
        end
        
        site_idx += 1
    end
    
    # Main computation - sum over all atoms in all unit cells
    out = zeros(Float64, length(qxs), length(qys), length(ωs))
    
    @inbounds for (qxi, qx) in enumerate(qxs), (qyi, qy) in enumerate(qys), (ωi, ω) in enumerate(ωs)
        sum_val = 0.0
        
        for xi in 1:Nx, yi in 1:Ny, zi in 1:Nz, atom_idx in 1:Natoms, ti in 1:Nt_extended
            # Spatial phase factor for this unit cell and atom
            spatial_phase = cos_qr[qxi, qyi, xi, yi, zi, atom_idx]
            
            # Temporal Fourier component for this atom
            temporal_component = cos_ωt[ωi, ti] * real(G_extended[xi, yi, zi, atom_idx, ti]) - 
                               sin_ωt[ωi, ti] * imag(G_extended[xi, yi, zi, atom_idx, ti])
            
            sum_val += spatial_phase * temporal_component
        end
        
        out[qxi, qyi, ωi] = sum_val
    end
    
    return out
end

# Modified FTParams struct for multi-atom systems
struct FTParams2D
    allowed_qxs::Vector{Float64}
    allowed_qys::Vector{Float64}
    energies::Vector{Float64}
    positions::Matrix{Float64}  # Real-space positions of unit cells [3 x Nsites]
    atom_positions::Matrix{Float64}  # Positions of atoms within unit cell [3 x Natoms]
    ts::Vector{Float64}
end


# Alternative approach: Extract all atomic positions directly
function extract_all_atomic_positions_from_sunny(sys)
    all_positions = []
    
    for site in eachsite(sys)
        r = global_position(sys, site)  # Real-space position of this atom
        push!(all_positions, r)
    end
    
    pos_matrix = hcat(all_positions...)
    return pos_matrix
end

# Create FTParams2D for multi-atom systems
function create_ft_params_multiAtom_from_sunny(sys, qxs, qys, ωs, ts)
    unit_cell_positions, atom_positions = extract_positions_from_sunny_multiAtom(sys)
    return FTParams2D(qxs, qys, ωs, unit_cell_positions, atom_positions, ts)
end

# Helper function for extending time series in multi-atom case
function extend_time_series_multiAtom(ts, G::AbstractArray{ComplexF64,5}; linear_predict_params)
    Nx, Ny, Nz, Natoms, Nt = size(G)
    n_predict = linear_predict_params.n_predict
    
    # Extend time array
    dt = ts[2] - ts[1]
    ts_extended = vcat(ts, [ts[end] + i*dt for i in 1:n_predict])
    
    # Extend G array - predict for each spatial point and atom
    G_extended = zeros(ComplexF64, Nx, Ny, Nz, Natoms, Nt + n_predict)
    G_extended[:, :, :, :, 1:Nt] .= G
    
    # Apply linear prediction to each spatial point and atom independently
    for xi in 1:Nx, yi in 1:Ny, zi in 1:Nz, atom_idx in 1:Natoms
        G_slice = G[xi, yi, zi, atom_idx, :]
        _, G_predicted = extend_time_series(ts, reshape(G_slice, 1, :); linear_predict_params)
        G_extended[xi, yi, zi, atom_idx, :] = G_predicted[1, :]
    end
    
    return ts_extended, G_extended
end