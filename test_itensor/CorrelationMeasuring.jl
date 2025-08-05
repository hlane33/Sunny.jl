#Like CorrelationSampling but for MeasuredCorrelations with QuantumCorrelations object



function get_observables_from_G!(buf, G, t_idx, observables, atom_idcs)
    """
    Load observables from precomputed TDVP trajectory G into buffer slice.
    Uses the exact same pattern as the original observable_values! function.
    """

    print("Observable Index", CartesianIndices(observables))
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
    
    # Setup - same as original
    buf_size = size(samplebuf, 6)
    nsnaps = (buf_size÷2) + 1
    
    # Zero-pad second half for FFT
    samplebuf[:,:,:,:,:,(nsnaps+1):end] .= 0
    
    # Load trajectory data
    get_trajectory_from_G!(samplebuf, G, nsnaps, observables, atom_idcs)

    prefft_buf = copy(samplebuf)
    println("Original: ", pointer(samplebuf))
    println("Copy: ", pointer(prefft_buf))

    return prefft_buf
end

function compute_S_v2(qs, ωs, G, positions, c, ts)
    out = zeros(ComplexF64, length(qs), length(ωs))
    for (qi, q) ∈ enumerate(qs)
        for (ωi, ω) ∈ enumerate(ωs)
            sum_val = 0.0
            for xi ∈ 1:length(positions), ti ∈ 1:length(ts)
                 # Split into position and time exponentials
                pos_factor = exp(-im * q * (positions[xi] - c))
                time_factor = exp(im * ω * ts[ti])
                # Multiply with G and take the real part
                sum_val += real(pos_factor * time_factor * G[xi, ti])
            end
            out[qi, ωi] = sum_val
        end
    end
    print("V2")
    return real(out)
end

function accum_sample_other!(qc::QuantumCorrelations; window=:cosine)
    (; data, corr_pairs, samplebuf, corrbuf, space_fft!, time_fft!) = qc
    npos = size(samplebuf,5)
    
    
    # Get q-values for the first spatial dimension
    Lx = size(samplebuf, 2) 
    c = div(Lx, 2) # Centering for 1D chain
 
    # Get ω-values for temporal dimension
    Lt = size(samplebuf, 6)
    
    # Slice params
    obs_idx = 3 # Assuming Sz is at index 3 in observables
    corr_idx = 1
    y_idx = 1    
    z_idx = 1  
    
    #compute S params
    positions = 1:Lx
    energies = range(0, 5, 20)
    allowed_qs = 0:(1/Lx):2π
    new_allowed_qs = (2π/Lx) * (0:(Lx-1))
    ts = 0.0:(Lt-1) # Assuming uniform time steps, adjust as needed
    G = samplebuf[obs_idx, :, y_idx, z_idx, 1, :]
    out = compute_S_v2(new_allowed_qs, energies, G, positions, c, ts)

    #data slice params 
    q_max = min(size(out,1), size(data,4))
    ω_max = min(size(out,2), size(data,7))
    qc.data[corr_idx, 1, 1, 1:q_max, y_idx, z_idx, 1:ω_max] .= out[1:q_max, 1:ω_max]

end


function accum_sample!(qc::QuantumCorrelations; window)
    (; data, corr_pairs, samplebuf, corrbuf, space_fft!, time_fft!, corr_fft!, corr_ifft!) = qc
    npos = size(samplebuf)[5]
    num_time_offsets = size(samplebuf, 6)
    T = (num_time_offsets÷2) + 1 # Duration that each signal was recorded for

    # Time offsets (in samples) Δt = [0,1,...,(T-1),-(T-1),...,-1] produced by 
    # the cross-correlation between two length-T signals
    time_offsets = FFTW.fftfreq(num_time_offsets, num_time_offsets)

    # Transform A(q) = ∑ exp(iqr) A(r).
    # This is opposite to the FFTW convention, so we must conjugate
    # the fft by a complex conjugation to get the correct sign.
    samplebuf .= conj.(samplebuf)
    space_fft! * samplebuf
    samplebuf .= conj.(samplebuf)

    # Transform A(ω) = ∑ exp(-iωt) A(t)
    # In samplebuf, the original signal is from 1:T, and the rest
    # is zero-padding, from (T+1):num_time_offsets. This allows a
    # usual FFT in the time direction, even though the signal isn't periodic.
    time_fft! * samplebuf

    # Number of contributions to the DFT sum (non-constant due to zero-padding).
    # Equivalently, this is the number of estimates of the correlation with
    # each offset Δt that need to be averaged over.
    n_contrib = reshape(T .- abs.(time_offsets), 1, 1, 1, num_time_offsets)
    
    # As long as `num_time_offsets` is odd, there will be a non-zero number of
    # contributions, so we don't need this line
    #$ @assert isodd(num_time_offsets)
    n_contrib[n_contrib .== 0] .= Inf

    # count = sc.nsamples += 1

    for j in 1:npos, i in 1:npos, (c, (α, β)) in enumerate(corr_pairs)
        # α, β = ci.I

        # sample_α = @view samplebuf[α,:,:,:,i,:]
        # sample_β = @view samplebuf[β,:,:,:,j,:]
        databuf  = @view data[c,i,j,:,:,:,:]

        # # According to Sunny convention, the correlation is between
        # # α† and β. This conjugation implements both the dagger on the α
        # # as well as the appropriate spacetime offsets of the correlation.
        # @. corrbuf = sample_β
        # corr_ifft! * corrbuf
        # corrbuf ./= n_contrib

        # @assert window in (:cosine, :rectangular)
        # if window == :cosine
        #     # Multiply the real-time correlation data by a cosine window that
        #     # smoothly goes to zero at offsets approaching the trajectory
        #     # length, Δt → T. This smooth windowing mitigates ringing artifacts
        #     # that appear when imposing periodicity on the real-space
        #     # trajectory. Note, however, that windowing also broadens the signal
        #     # S(ω) on the characteristic scale of one frequency bin Δω = 2π/T.
        #     window_func = cos.(range(0, π, length=num_time_offsets+1)[1:end-1]).^2
        #     corrbuf .*= reshape(window_func, 1, 1, 1, num_time_offsets)
        # end

        # corr_fft! * corrbuf

        # databuf .= corrbuf
        databuf .= samplebuf[3,:,:,:,1,:]
    end

    return nothing
end



function add_sample!(qc::QuantumCorrelations, G::Array{ComplexF64,2}; window=:cosine)
    # Step 1: Replace new_sample! with quantum data injection
    prefft_buf = new_sample!(qc, G)
    
    # Step 2: Use Sunny's existing accum_sample! 
    accum_sample_other!(qc; window)
    
    println("Quantum TDVP data processed through Sunny's infrastructure")

    return prefft_buf # Return the updated sample buffer but before FFTS
end