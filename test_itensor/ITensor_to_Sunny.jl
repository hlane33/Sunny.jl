using Sunny, FFTW

"""
G should have dimensions (N_sites, N_times) where N_times matches the 
expected trajectory length in samplebuf.

Like add_sample() in Sunny/CorrelationSampling but uses the quantum new sample
I imagine there is a way to merge quantum_new_sample with new_sample in the Sunny SRC code such that this function 
is redundant, but I didn't want to mess with it for now.
"""
function quantum_add_sample!(sc::SampledCorrelations, G::Array{ComplexF64,2}; window=:cosine)
    # Step 1: Replace new_sample! with quantum data injection
    quantum_new_sample!(sc, G)
    
    # Step 2: Use Sunny's existing accum_sample! 
    Sunny.accum_sample!(sc; window)
    
    println("Quantum TDVP data processed through Sunny's infrastructure")
end

function load_observables_from_G!(buf, G, t_idx, observables, atom_idcs)
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

function load_trajectory_from_G!(buf, G, nsnaps, observables, atom_idcs)
    """
    Load full trajectory from G into buffer.
    Mirrors the structure of trajectory! but loads precomputed data.
    """
    @assert size(G, ndims(G)) >= nsnaps "G must have at least $nsnaps time steps"
    
    # Load each time snapshot
    for t_idx in 1:nsnaps
        load_observables_from_G!(@view(buf[:,:,:,:,:,t_idx]), G, t_idx, observables, atom_idcs)
    end
    
    return nothing
end

function quantum_new_sample!(sc::SampledCorrelations, G::Array{ComplexF64})
    """
    Add a sample from precomputed TDVP trajectory G.
    """
    (; samplebuf, observables, atom_idcs) = sc
    
    # Setup - same as original
    buf_size = size(samplebuf, 6)
    nsnaps = (buf_size÷2) + 1
    
    # Zero-pad second half for FFT
    samplebuf[:,:,:,:,:,(nsnaps+1):end] .= 0
    
    # Load trajectory data
    load_trajectory_from_G!(samplebuf, G, nsnaps, observables, atom_idcs)
    
    return nothing
end


function get_quantum_correlations(energies, sys; dt)
    # Create minimal 1D system -- since we have already calculated correlations
    # this just ensures that dimensions of G match dimensions of samplebuffer
    
    
    # Create SampledCorrelations with matching parameters
    # dt is redundant here as correlations have already been calculated but must abide by what 
    #sampled correlations is expecting
    sc = SampledCorrelations(sys; 
                           measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false),
                           energies=energies, 
                           dt=dt, 
                           calculate_errors=false)

   
    
    return sc
end

"""
Complete workflow to process TDVP data through Sunny's infrastructure
"""
function Get_StructureFactor_with_Sunny(G::Array{ComplexF64,2}, energies, sys; window=:cosine)
    
    sc = get_quantum_correlations(energies, sys; dt = 0.5)
    # Inject quantum data and process
    quantum_add_sample!(sc, G; window)
    
    println("TDVP data successfully processed through Sunny")
    println("Sample count: $(sc.nsamples)")
    println("Energy range: $(energies[1]) to $(energies[end])")
    
    return sc
end

