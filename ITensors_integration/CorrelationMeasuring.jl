using LinearAlgebra

##########
# This file mimics Sunny.CorrelationSampling but for MeasuredCorrelations with QuantumCorrelations object
# The functions in here are responsible for adding the correlation data (G) to the sample buffer in the QuantumCorrelations object
#########

"""
    get_observables_from_G!(buf, G, t_idx, observables, atom_idcs)

Load observables from precomputed TDVP trajectory `G` into buffer slice `buf` at time index `t_idx`.

Uses the same pattern as `observable_values!` in Sunny.

# Arguments
- `buf`: Buffer to store observables (modified in-place).
- `G`: Precomputed trajectory data (complex array).
- `t_idx`: Time index to extract from `G`.
- `observables`: Indices defining observable locations.
- `atom_idcs`: Mapping from lattice positions to atom indices.

# Notes
- Assumes `sz` is at index 3 in `observables`.
- Currently simplified for 1D systems; should generalize for arbitrary lattices.
"""
function get_observables_from_G!(buf, G, t_idx, observables, atom_idcs)
    for idx in CartesianIndices(observables)
        obs_idx, la, lb, lc, pos = idx.I
        atom = atom_idcs[la, lb, lc, pos]
        site = la  # Simplified for 1D case

        if obs_idx == 3 && atom ≤ size(G, 1)  # `sz` assumed at index 3
            buf[idx] = G[site, t_idx]
        else
            buf[idx] = 0.0
        end
    end
    return nothing
end

"""
    get_trajectory_from_G!(buf, G, nsnaps, observables, atom_idcs)

Load a full trajectory from `G` into buffer `buf` for `nsnaps` time steps.

# Arguments
- `buf`: Output buffer (6D array, modified in-place).
- `G`: Input trajectory data (complex array).
- `nsnaps`: Number of time snapshots to load.
- `observables`: Observable indices.
- `atom_idcs`: Atom position mappings.

# Throws
- `AssertionError` if `G` has fewer than `nsnaps` time steps.
"""
function get_trajectory_from_G!(buf, G, nsnaps, observables, atom_idcs)
    @assert size(G, ndims(G)) ≥ nsnaps "G must have at least $nsnaps time steps"
    for t_idx in 1:nsnaps
        get_observables_from_G!(
            @view(buf[:, :, :, :, :, t_idx]), 
            G, t_idx, observables, atom_idcs
        )
    end
    return nothing
end

"""
    new_sample!(qc::QuantumCorrelations, G::Array{ComplexF64})

Add a sample from precomputed TDVP trajectory `G` to `QuantumCorrelations` object `qc`.

# Arguments
- `qc`: `QuantumCorrelations` container (modified in-place).
- `G`: Trajectory data (complex matrix).
"""
function new_sample!(qc::QuantumCorrelations, G::Array{ComplexF64})
    (; samplebuf, observables, atom_idcs) = qc
    N_time_steps = size(samplebuf, 6)
    get_trajectory_from_G!(samplebuf, G, N_time_steps, observables, atom_idcs)
    return nothing
end

"""
    accum_sample_other!(qc::QuantumCorrelations, FT_params, linear_predict_params; window=:cosine)

Accumulate a sample into `qc.data` after Fourier transform and linear prediction.

# Arguments
- `qc`: `QuantumCorrelations` object (modified in-place).
- `FT_params`: Named tuple with fields `(allowed_qs, energies, positions, c, ts)`.
- `linear_predict_params`: Parameters for linear prediction.

# Keywords
- `window`: Windowing function (`:cosine` by default).

# Notes
- Assumes `sz` is at index 3 in `qc.observables`.
- Debug prints retained for shape verification.
"""
function accum_sample_other!(qc::QuantumCorrelations, FT_params, linear_predict_params; window=:cosine)
    (; allowed_qs, energies, positions, c, ts) = FT_params
    (; data, samplebuf) = qc
    
    # Slice params (hardcoded for Sz)
    obs_idx = 3
    corr_idx = 1
    y_idx = 1    
    z_idx = 1  
    
    G = samplebuf[obs_idx, :, y_idx, z_idx, 1, :]
    println("Shape of G: ", size(G))
    out = compute_S_v2(G, allowed_qs, energies, positions, c, ts; linear_predict_params)
    println("size(out): ", size(out))
    qc.data[corr_idx, 1, 1, :, y_idx, z_idx, :] .= out
end

"""
    add_sample!(qc::QuantumCorrelations, G::Array{ComplexF64,2}, FT_params, linear_predict_params; window=:cosine)

Process a TDVP trajectory sample through Sunny's infrastructure.

# Arguments
- `qc`: `QuantumCorrelations` object (modified in-place).
- `G`: Trajectory data (complex matrix).
- `FT_params`: Fourier transform parameters.
- `linear_predict_params`: Linear prediction parameters.

# Keywords
- `window`: Windowing function (`:cosine` by default).

# Notes
1. Injects quantum data via `new_sample!`.
2. Processes with `accum_sample_other!`.
"""
function add_sample!(qc::QuantumCorrelations, G::Array{ComplexF64,2}, FT_params, linear_predict_params; window=:cosine)
    new_sample!(qc, G)
    accum_sample_other!(qc, FT_params, linear_predict_params; window)
    println("Quantum TDVP data processed through Sunny's infrastructure")
    return nothing
end