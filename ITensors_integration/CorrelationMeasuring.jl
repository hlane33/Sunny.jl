#Like CorrelationSampling but for MeasuredCorrelations with QuantumCorrelations object
using LinearAlgebra
include("useful_functions.jl")


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
    out = compute_S_v2(G, allowed_qs, energies, positions, c, ts; linear_predict_params)
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