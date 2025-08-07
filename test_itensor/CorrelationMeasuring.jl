#Like CorrelationSampling but for MeasuredCorrelations with QuantumCorrelations object
using LinearAlgebra


function get_observables_from_G!(buf, G, t_idx, observables, atom_idcs)
    """
    Load observables from precomputed TDVP trajectory G into buffer slice.
    Uses the exact same pattern as the original observable_values! function.
    """
    for idx in CartesianIndices(observables)
        
        obs_idx, la, lb, lc, pos = idx.I
        atom = atom_idcs[la, lb, lc, pos]
        site = la #true for simple 1D case
        

        if obs_idx == 3 && atom <= size(G, 1) #currently fixed because sz at 3 in observables to match TDVP
             #site should match site numbering done for TDVP, currently crude for 1D but should use code from Sunny to Itensor
             # G doesn't currently "know" what observables the code contains.
            buf[idx] = G[site, t_idx]

        else
            buf[idx] = 0.0
        end
        
    end
    
    return nothing
end

function get_trajectory_from_G!(buf, G, nsnaps, observables, atom_idcs)
    """
    Load full trajectory from G into buffer.
    Mirrors the structure of trajectory! but loads precomputed data.
    """
    @assert size(G, ndims(G)) >= nsnaps "G must have at least $nsnaps time steps"
    
    # Load each time snapshot
    for t_idx in 1:nsnaps
        get_observables_from_G!(@view(buf[:,:,:,:,:,t_idx]), G, t_idx, observables, atom_idcs)
    end
    
    return nothing
end

function new_sample!(qc::QuantumCorrelations, G::Array{ComplexF64})
    """
    Add a sample from precomputed TDVP trajectory G.
    """
    (; samplebuf, observables, atom_idcs) = qc

    N_time_steps = size(samplebuf, 6)

    # Load trajectory data
    get_trajectory_from_G!(samplebuf, G, N_time_steps, observables, atom_idcs)
    
    return nothing
end

function linear_predict(y::Union{Vector{Float64}, Vector{ComplexF64}}; n_predict::Int, n_coeff::Int)
    #n_coeff dictates how many previous values to use for prediction
    #n_predict is how many future values to predict
    n = length(y)
    n ≤ n_coeff && error("Time series too short for prediction.")
    
    # Construct the Toeplitz matrix for least-squares fitting
    Y = zeros(ComplexF64, n - n_coeff, n_coeff)
    for i in 1:(n - n_coeff)
        Y[i, :] = y[(n_coeff + i - 1):-1:i]
    end
    
    # Solve Y * d ≈ y[n_coeff+1:end] for coefficients d
    d = Y \ y[n_coeff+1:end]
    
    # Extrapolate using the AR model
    y_pred = copy(y)
    for i in (n+1):(n + n_predict)
        next_val = sum(d .* y_pred[(i-1):-1:(i - n_coeff)])
        push!(y_pred, next_val)
    end
    
    return y_pred
end

function compute_S(G, qs, ωs, positions, c, ts; linear_predict_params)
    
    out = zeros(Float64, length(qs), length(ωs))

    # Extend time axis via linear prediction
    if linear_predict_params.n_predict > 0
        ts = [ts; ts[end] .+ (1:linear_predict_params.n_predict) * (ts[2] - ts[1])]
        G = similar(G, (size(G,1), length(extended_ts)))
        # Extrapolate each spatial point's time series
        for xi in 1:size(G,1)
            G[xi, :] = linear_predict(G[xi, :]; linear_predict_params...)
        end
    end

    #cosine windowing
    window_func = cos.(range(0, π, length=length(ts))).^ 2
    G .*= window_func'  # Apply to all spatial sites
    out = zeros(Float64, length(qs), length(ωs))
    for (qi, q) ∈ enumerate(qs)
        for (ωi, ω) ∈ enumerate(ωs)
            sum_val = 0.0
            for xi ∈ 1:length(positions), ti ∈ 1:length(ts)
                val = cos(q * (positions[xi]-c)) * 
                      (cos(ω * ts[ti]) * real(G[xi, ti]) - 
                       sin(ω * ts[ti]) * imag(G[xi, ti]))
                sum_val += val
            end
            out[qi, ωi] = real(sum_val)
        end
    end
    return out
end

function compute_S_v2(G, qs, ωs, positions, c, ts; linear_predict_params)
    out = zeros(ComplexF64, length(qs), length(ωs))
    
    # Extend time axis via linear prediction
    if linear_predict_params.n_predict > 0
        ts = [ts; ts[end] .+ (1:linear_predict_params.n_predict) * (ts[2] - ts[1])]
        extended_G = similar(G, (size(G,1), length(ts)))
        # Extrapolate each spatial point's time series
        for xi in 1:size(G,1)
            extended_G[xi, :] = linear_predict(G[xi, :]; linear_predict_params...)
        end

        G = extended_G
    end

    #cosine windowing
    window_func = cos.(range(0, π, length=length(ts))).^ 2
    G .*= window_func'  # Apply to all spatial sites

    # Compute Fourier transform on extended data
    for (qi, q) in enumerate(qs)
        for (ωi, ω) in enumerate(ωs)
            sum_val = 0.0
            for xi in 1:length(positions), ti in 1:length(ts)
                pos_factor = exp(im * q * (positions[xi] - c))
                time_factor = exp(im * ω * ts[ti])
                sum_val += pos_factor * time_factor * G[xi, ti]
            end
            out[qi, ωi] = sum_val
        end
    end

    println("Applied linear prediction with n_predict=$(linear_predict_params.n_predict)")
    return out
end

function accum_sample_other!(qc::QuantumCorrelations, FT_params, linear_predict_params; window=:cosine)
    (; allowed_qs, energies, positions, c, ts) = FT_params
    (; data, samplebuf,) = qc
    
    # Slice params
    obs_idx = 3 # Assuming Sz is at index 3 in observables
    corr_idx = 1
    y_idx = 1    
    z_idx = 1  
    
    #compute S
    G = samplebuf[obs_idx, :, y_idx, z_idx, 1, :]
    println("Shape of G: ", size(G))
    out = compute_S(G, allowed_qs, energies, positions, c, ts; linear_predict_params)
    print("size(out): ", size(out))
    print("size data slice: ", size(data[corr_idx, 1, 1, :, y_idx, z_idx, :]))
    #data slice params 
    qc.data[corr_idx, 1, 1, :, y_idx, z_idx, :] .= out

end


function add_sample!(qc::QuantumCorrelations, G::Array{ComplexF64,2}, FT_params, linear_predict_params; window=:cosine)
    # Step 1: Replace new_sample! with quantum data injection
    new_sample!(qc, G)
    
    # Step 2: Use Sunny's existing accum_sample! 
    accum_sample_other!(qc, FT_params, linear_predict_params; window)

    println("Quantum TDVP data processed through Sunny's infrastructure")

    return nothing
end