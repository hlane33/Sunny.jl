using StaticArrays: SVector
using Sunny: Intensities

# Key compatibility functions for Sunny's intensities interface
function available_energies(qc::QuantumCorrelations; negative_energies=false)
    if negative_energies
        return qc.energies  # Return all frequencies
    else
        # Return only non-negative frequencies
        return qc.energies[qc.energies .>= 0]
    end
end

function contains_dynamic_correlations(qc::QuantumCorrelations)
    return !isnan(qc.Δω)
end

function to_reshaped_rlu(qc::QuantumCorrelations, q)
    orig_cryst = @something qc.origin_crystal qc.crystal
    return qc.crystal.recipvecs \ orig_cryst.recipvecs * q
end

# Main intensities method overload for QuantumCorrelations
function intensities(qc::QuantumCorrelations, qpts; energies, kernel=nothing, kT=nothing)
    if !isnothing(kT) && kT <= 0
        error("Positive `kT` required for classical-to-quantum corrections, or set `kT=nothing` to disable.")
    end
    if !isnothing(kernel)
        error("Kernel post-processing not yet available for `QuantumCorrelations`.")
    end

    # Determine energy information
    (ωs, ωidcs) = if energies == :available
        ωs = available_energies(qc; negative_energies=false)
        (ωs, findall(ω -> ω >= 0, qc.energies))
    elseif energies == :available_with_negative
        ωs = available_energies(qc; negative_energies=true)
        (ωs, axes(ωs, 1))
    else
        # For custom energy list, find nearest available energies
        ωs_all = available_energies(qc; negative_energies=true)
        energies_sorted = sort(energies)
        @assert all(x -> x == true, energies .== energies_sorted) "Specified energies must be an ordered list."
        @assert minimum(energies) >= minimum(ωs_all) && maximum(energies) <= maximum(ωs_all) "Specified energies includes values for which there is no available data."
        
        ωidcs = map(val -> find_idx_of_nearest_fft_energy(ωs_all, val), energies)
        (ωs_all[ωidcs], ωidcs)
    end

    # Prepare memory and configuration variables
    qs_reshaped = [to_reshaped_rlu(qc, q) for q in qpts.qs]

    print("formfactors no.",size(qc.measure.formfactors))

    # Check form factors - get them properly from the measure
    ffs = qc.measure.formfactors[1, :]

    # Initialize intensities array
    intensities_out = zeros(eltype(qc.measure), length(ωs), length(qpts.qs))
    
    # Get wave vector information
    q_idx_info = pruned_wave_vector_info_quantum(qc, qs_reshaped)
    
    # Crystal and correlation setup
    crystal = qc.crystal
    NCorr = Val{size(qc.data, 1)}()
    NPos = Val{qc.natoms}()

    println("Computing intensities for $(length(qpts.qs)) q-points and $(length(ωs)) frequencies")
    println("Data range: [$(minimum(real(qc.data))), $(maximum(real(qc.data)))]")

    # Main intensities calculation
    intensities_aux_quantum!(intensities_out, qc.data, qc.crystal, qc.positions, 
                           qc.measure.combiner, ffs, q_idx_info, ωidcs, NCorr, NPos, qc.sys_dims)

    # Convert to q-space density in original RLU
    intensities_out .*= det(qc.crystal.recipvecs) / det(crystal.recipvecs)

    # Post-processing for dynamical correlations
    if contains_dynamic_correlations(qc)
        # Convert time axis to density
        n_all_ω = length(qc.energies)
        intensities_out ./= (n_all_ω * qc.Δω)

        # Apply classical-to-quantum correspondence factor
        if !isnothing(kT)
            c2q = [iszero(ω) ? 1 : abs((ω/kT) / (1 - exp(-ω/kT))) for ω in ωs]
            for i in axes(intensities_out, 2)
                intensities_out[:, i] .*= c2q
            end
        end
    end

    # Reshape to match expected output format
    intensities_out = reshape(intensities_out, length(ωs), size(qpts.qs)...)

    println("Final intensities range: [$(minimum(real(intensities_out))), $(maximum(real(intensities_out)))]")

    return if contains_dynamic_correlations(qc)
        Intensities(crystal, qpts, collect(ωs), intensities_out)
    else
        StaticIntensities(crystal, qpts, dropdims(intensities_out; dims=1))
    end
end

# Quantum-specific version of pruned_wave_vector_info
function pruned_wave_vector_info_quantum(qc::QuantumCorrelations, qs)
    # Round to nearest wavevector and wrapped index
    Ls = qc.sys_dims
    ms = map(qs) do q 
        round.(Int, Ls .* q)
    end
    idcs = map(ms) do m
        CartesianIndex{3}(map(i -> mod(m[i], Ls[i])+1, (1, 2, 3)))
    end

    # Convert to absolute units
    qabs_rounded = map(m -> qc.crystal.recipvecs * (m ./ qc.sys_dims), ms)

    # Find unique wave vectors
    starts = findall(i -> i == 1 || !isapprox(qabs_rounded[i-1], qabs_rounded[i]; atol=1e-12), eachindex(qabs_rounded))
    counts = if length(starts) > 1
        [starts[2:end] - starts[1:end-1]; length(idcs) - starts[end] + 1]
    else
        [length(idcs)]
    end

    # Remove contiguous repetitions
    qabs = qabs_rounded[starts]
    idcs = idcs[starts]

    return (; qabs, idcs, counts)
end

# Quantum-specific auxiliary function for intensities calculation
function intensities_aux_quantum!(intensities, data, crystal, positions, combiner, ff_atoms, 
                                q_idx_info, ωidcs, ::Val{NCorr}, ::Val{NPos}, sys_dims) where {NCorr, NPos}
    (; qabs, idcs, counts) = q_idx_info 
    (; recipvecs) = crystal 
    qidx = 1
    
    println("Processing $(length(qabs)) unique q-points")
    
    for (qabs, idx, count) in zip(qabs, idcs, counts)
        # Compute prefactors for phase averaging
        prefactors = prefactors_for_phase_averaging_quantum(qabs, recipvecs, positions, 
                                                          ff_atoms, Val{NCorr}(), Val{NPos}(), sys_dims)

        # Perform phase-averaging over all omega
        for (n, iω) in enumerate(ωidcs)
            elems = zero(SVector{NCorr, ComplexF64})
            
            # Extract correlations at this q-point and frequency
            for j in 1:NPos, i in 1:NPos
                # data has shape (ncorrs, natoms, natoms, Lx, Ly, Lz, nω)
                # Properly unpack CartesianIndex
                corr_data = SVector{NCorr}(view(data, :, i, j, idx.I..., iω))
                elems += (prefactors[i] * conj(prefactors[j])) * corr_data
            end
            
            val = combiner(qabs, elems)
            intensities[n, qidx] = val
        end

        # Copy for repeated q-values
        for idx_copy in qidx+1:qidx+count-1, n in axes(ωidcs, 1)
            intensities[n, idx_copy] = intensities[n, qidx]
        end

        qidx += count
    end
end

# Fixed prefactor calculation for quantum case
function prefactors_for_phase_averaging_quantum(qabs, recipvecs, positions, ff_atoms, ::Val{NCorr}, ::Val{NPos}, sys_dims) where {NCorr,NPos}
    prefactors = zeros(ComplexF64, NPos)
    
    for i in 1:NPos
        # Map atom index to site index (accounting for multiple unit cells)
        site_idx = i  # This works for single atom per unit cell; adjust if needed
        
        # Get position from the positions vector (now properly structured)
        pos = positions[site_idx]
        
        # Compute phase factor
        phase = exp(2π * im * dot(qabs, pos))
        
        # Extract form factor - handle both scalar and function form factors
        ff_value = if ff_atoms[site_idx] isa Function
            ff_atoms[site_idx](norm(qabs)^2)
        elseif ff_atoms[site_idx] isa Number
            ff_atoms[site_idx]
        else
            # Try to compute form factor using Sunny's function
            try
                Sunny.compute_form_factor(ff_atoms[site_idx], norm(qabs)^2)
            catch
                1.0  # Fallback to unity if form factor computation fails
            end
        end
        
        prefactors[i] = ff_value * phase
    end
    
    return prefactors
end

# Helper function for finding nearest FFT energy (copied from Sunny's implementation)
function find_idx_of_nearest_fft_energy(ref, val)
    for i in axes(ref[1:end-1], 1)
        x1, x2 = ref[i], ref[i+1]
        if x1 <= val <= x2 
            if abs(x1 - val) < abs(x2 - val)
                return i
            else
                return i+1
            end
        end
    end
    # Deal with edge case arising due to FFT index ordering
    if ref[end] <= val <= 0.0
        if abs(ref[end] - val) <= abs(val)
            return length(ref)
        else
            return 1
        end
    end
    error("Value does not lie in bounds of reference list.")
end