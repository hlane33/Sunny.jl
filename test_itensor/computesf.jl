"""
Extract Dynamic Structure Factor S(q,ω) with custom resolution

compute_dsf_flexible(G, ts, η, sys_dims, positions; qs=nothing, ωs=nothing)

Extract S(q,ω) from correlation trajectories with custom q-points and frequencies.

# Arguments
- `G`: Correlation matrix [nsites × ntimes] from your compute_G function
- `ts`: Time array used in correlation computation  
- `η`: Damping parameter used
- `sys_dims`: System dimensions as (Lx, Ly, Lz) tuple
- `positions`: Array of position vectors [3 × nsites] in real space units
- `qs`: (Optional) Custom q-vectors [3 × nq] in units of 2π/L. If nothing, uses FFT grid
- `ωs`: (Optional) Custom frequency array [nω]. If nothing, uses FFT frequencies

# Returns
- `qs_out`: Array of q-vectors [3 × nq] used in computation
- `ωs_out`: Frequency array [nω] used in computation  
- `Sqw`: Dynamic structure factor S(q,ω) as [nq × nω] array
"""
function compute_dsf_flexible(G::Array{ComplexF64,2}, ts::Vector{Float64}, η::Float64,
                             sys_dims::Tuple{Int,Int,Int}, positions::Vector{Int64};
                             qs::Union{Vector{Float64}, Matrix{Float64}}=nothing,  # Accepts 1D or 2D, 
                             ωs::Union{Nothing,Vector{Float64}}=nothing)
    
    nsites, ntimes = size(G)
    Lx, Ly, Lz = sys_dims
    dt = ts[2] - ts[1]
    
    # Apply time windowing with damping
    window = exp.(-η .* ts)
    window ./= sum(window)  # Normalize
    
    # Apply windowing to correlations
    G_windowed = copy(G)
    for i in 1:nsites
        G_windowed[i, :] .*= window
    end
    
    # Set up q-points
    if qs === nothing
        # Use FFT grid (original behavior)
        qxs = fftfreq(Lx, Lx) .* 2π
        qys = fftfreq(Ly, Ly) .* 2π  
        qzs = fftfreq(Lz, Lz) .* 2π
        
        qs_out = zeros(3, prod(sys_dims))
        idx = 1
        for i in 1:Lx, j in 1:Ly, k in 1:Lz
            qs_out[:, idx] = [qxs[i], qys[j], qzs[k]]
            idx += 1
        end
    else
        qs_out = qs
    end
    
    # Set up frequencies  
    if ωs === nothing
        # Use FFT frequencies (original behavior)
        ωs_raw = fftfreq(ntimes, 1/dt) .* 2π
        ωs_out = fftshift(ωs_raw)
    else
        ωs_out = ωs
    end
    
    # Compute S(q,ω) using direct summation (like second function)
    nq = size(qs_out, 2)
    nω = length(ωs_out)
    Sqw = zeros(Float64, nq, nω)
    
    # Center of system for phase reference
    c = [Lx/2, Ly/2, Lz/2]
    
    for (qi, q_idx) in enumerate(1:nq)
        q = qs_out[:, q_idx]
        for (ωi, ω) in enumerate(ωs_out)
            sum_val = 0.0
            
            for xi in 1:nsites
                pos = positions[:, xi] 
                phase_factor = q ⋅ (pos - c)
                
                for ti in 1:ntimes
                    t = ts[ti]
                    # Fourier transform: Re[G(r,t) * exp(-i(q⋅r - ωt))]
                    val = cos(phase_factor - ω * t) * real(G_windowed[xi, ti]) + 
                          sin(phase_factor - ω * t) * imag(G_windowed[xi, ti])
                    sum_val += val
                end
            end
            
            Sqw[qi, ωi] = sum_val
        end
    end
    
    # Normalize by number of sites and time points
    Sqw ./= (nsites * ntimes)
    
    return qs_out, ωs_out, Sqw
end

"""
Helper function to create high-resolution q and ω grids
"""
function create_high_res_grids(sys_dims, ts; q_factor=2, ω_factor=2)
    Lx, Ly, Lz = sys_dims
    dt = ts[2] - ts[1]
    ntimes = length(ts)
    
    # High resolution q-grid
    nqx, nqy, nqz = q_factor .* sys_dims
    qxs = range(-π, π, length=nqx+1)[1:end-1] .* (Lx/π)
    qys = range(-π, π, length=nqy+1)[1:end-1] .* (Ly/π) 
    qzs = range(-π, π, length=nqz+1)[1:end-1] .* (Lz/π)
    
    qs_hires = zeros(3, nqx * nqy * nqz)
    idx = 1
    for qx in qxs, qy in qys, qz in qzs
        qs_hires[:, idx] = [qx, qy, qz]
        idx += 1
    end
    
    # High resolution frequency grid
    ω_max = π / dt  # Nyquist frequency
    nω_hires = ω_factor * ntimes
    ωs_hires = range(-ω_max, ω_max, length=nω_hires)
    
    return qs_hires, ωs_hires
end