using StaticArrays: SVector
import Sunny: available_energies, contains_dynamic_correlations, pruned_wave_vector_info,
intensities, Intensities, AbstractQPoints, prefactors_for_phase_averaging

# Key compatibility functions for Sunny's intensities interface
function available_energies(qc::QuantumCorrelations; negative_energies=false)
    isnan(qc.Δω) && return NaN

    n_all_ω = size(qc.data, 7)
    n_non_neg_ω = div(n_all_ω, 2) + 1
    ωvals = collect(FFTW.fftfreq(n_all_ω, n_all_ω * qc.Δω))
    ωvals[n_non_neg_ω] *= -1  # Adjust for FFTW convention (which is largest frequency negative)
    return negative_energies ? ωvals : ωvals[1:n_non_neg_ω]
end


function contains_dynamic_correlations(qc::QuantumCorrelations)
    return !isnan(qc.Δω)
end

# Takes a list of q points, converts into SampledCorrelation.data indices and
# corresponding exact wave vectors, and eliminates repeated elements.
function pruned_wave_vector_info(qc::QuantumCorrelations, qs)
    L_hires = size(qc.data)[4:6]
    L_phys = size(qc.samplebuf)[2:4]  # Assuming this is correct for your case

    ms = map(qs) do q 
        round.(Int, L_hires .* q)  # Changed from q[1] to q since q is already a 3-vector
    end
    
    # Two different indices
    data_idcs = map(ms) do m
        modded = mod.(m, L_hires) .+ 1
        CartesianIndex(modded...)  # Creates 3D index from the 3-element tuple
    end
    
    pos_idcs = map(ms) do m  
        hires_idx = mod.(m, L_hires) .+ 1
        phys_idx = mod.(round.(Int, (hires_idx .- 1) .* L_phys ./ L_hires), L_phys) .+ 1
        CartesianIndex(phys_idx...)  # Creates 3D index from the 3-element tuple
    end
    
    qabs_rounded = map(m -> qc.crystal.recipvecs * (m ./ L_hires), ms)
    
    # NOW CALCULATE COUNTS (same logic as original)
    starts = findall(i -> i == 1 || !isapprox(qabs_rounded[i-1], qabs_rounded[i]), eachindex(qabs_rounded))
    counts = starts[2:end] - starts[1:end-1]
    append!(counts, length(data_idcs) - starts[end] + 1)
    
    # Remove contiguous repetitions (same as original)
    qabs = qabs_rounded[starts]
    data_idcs = data_idcs[starts] 
    pos_idcs = pos_idcs[starts]
    
    return (; qabs, data_idcs, pos_idcs, counts)
end

# Crude slow way to find the energy axis index closest to some given energy.
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


# If the user specifies an energy list, round to the nearest available energies
# and give the corresponding indices into the raw data. This is fairly
# inefficient, though the cost is likely trivial next to the rest of the
# computation. Since this is an edge case (user typically expected to choose
# :available or :available_with_negative), not spending time on optimization
# now.
function rounded_energy_information(qc, energies)
    ωvals = available_energies(qc; negative_energies = true)
    energies_sorted = sort(energies)
    @assert all(x -> x ==true, energies .== energies_sorted) "Specified energies must be an ordered list."
    @assert minimum(energies) >= minimum(ωvals) && maximum(energies) <= maximum(ωvals) "Specified energies includes values for which there is no available data."
    ωidcs = map(val -> find_idx_of_nearest_fft_energy(ωvals, val), energies)
    return ωvals[ωidcs], ωidcs
end


# Documented under intensities function for LSWT. TODO: As a hack, this function
# is also being used as the back-end to intensities_static.
function intensities(qc::QuantumCorrelations, qpts;kernel=nothing)
    # Determine energy information
    n_all_ω = size(qc.data, 7)

    ωs = collect(range(0, 5, length=n_all_ω))  # Custom energy range TODO: remove hard coded max energy
    ωidcs = 1:length(ωs)  # All frequency indices


    # Prepare memory and configuration variables for actual calculation
    qpts = Base.convert(AbstractQPoints, qpts)
    qs_reshaped = [to_reshaped_rlu(qc, q) for q in qpts.qs]

    # Check that form factors are uniform for each observable.
    for col in eachcol(qc.measure.formfactors)
        @assert allequal(col) "Observable-dependent form factors not yet supported."
    end
    ffs = qc.measure.formfactors[1, :]

    intensities = zeros(eltype(qc.measure), isnan(qc.Δω) ? 1 : length(ωs), length(qpts.qs)) # N.B.: Inefficient indexing order to mimic LSWT
    q_idx_info = pruned_wave_vector_info(qc, qs_reshaped)
    crystal = @something qc.origin_crystal qc.crystal
    NCorr  = Val{size(qc.data, 1)}()
    # NPos = Val{size(qc.data, 2)}()
    NPos = Val{length(qc.crystal.positions)}()

    # Intensities calculation
    intensities_aux!(intensities, qc.data, qc.crystal, qc.positions, qc.measure.combiner, ffs, q_idx_info, ωidcs, NCorr, NPos)

    # Convert to a q-space density in original (not reshaped) RLU.
    intensities .*= det(qc.crystal.recipvecs) / det(crystal.recipvecs)

    println("Computed intensities for $(length(qpts.qs)) q-points and $(length(ωs)) frequencies.")

    intensities = reshape(intensities, length(ωs), size(qpts.qs)...)


    return Intensities(crystal, qpts, collect(ωs), intensities)
end



function intensities_aux!(intensities, data, crystal, positions, combiner, ff_atoms, q_idx_info, ωidcs, ::Val{NCorr}, ::Val{NPos}) where {NCorr, NPos}
    (; recipvecs) = crystal 
    qidx = 1
    
    (; qabs, data_idcs, pos_idcs, counts) = q_idx_info 
    
    for (qabs_val, data_idx, pos_idx, count) in zip(qabs, data_idcs, pos_idcs, counts)
        # Use HIGH-RES q-vector for form factors, but PHYSICAL position for site locations
        prefactors = prefactors_for_phase_averaging(qabs_val, recipvecs, 
                                                    view(positions, pos_idx, :), 
                                                    ff_atoms, Val{NCorr}(), Val{NPos}())

        # Perform phase-averaging over all omega using HIGH-RES data index
        for (n, iω) in enumerate(ωidcs)
            elems = zero(SVector{NCorr, ComplexF64})
            for j in 1:NPos, i in 1:NPos
                elems += (prefactors[i] * conj(prefactors[j])) * 
                            SVector{NCorr}(view(data, :, i, j, data_idx, iω))
            end
            val = combiner(qabs_val, elems)
            intensities[n, qidx] = val
        end

        # Copy for repeated q-values (high-res q's that map to same physical site)
        for idx in qidx+1:qidx+count-1, n in axes(ωidcs, 1)
            intensities[n, idx] = intensities[n, qidx]
        end

        qidx += count
    end
end

