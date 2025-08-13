using StaticArrays: SVector
import Sunny: available_energies, pruned_wave_vector_info,
              intensities, Intensities, AbstractQPoints, prefactors_for_phase_averaging


#############
#The functions here overload the functions in Sunny.DataRetreival.jl
#Not all functions in DataRetreival are necessary in the case of the manual Fourier Transform, those have been removed
#############



"""
    pruned_wave_vector_info(qc::QuantumCorrelations, qs)

Mimics pruned_wave_vector_info(sc::SampledCorrelations) but adjusts for the fact
that we want to prune to the external energy/momenta grid the user chooses, not based off the sampling time steps/site positions

Process wave vectors to eliminate duplicates and map to data indices.
Returns named tuple with `(qabs, data_idcs, pos_idcs, counts)`.

"""
function pruned_wave_vector_info(qc::QuantumCorrelations, qs)
    L_hires = size(qc.data)[4:6]
    L_phys = size(qc.samplebuf)[2:4]  

    ms = map(qs) do q 
        round.(Int, L_hires .* q)  
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
    
    # NOW CALCULATE COUNTS -- Although maybe this is always 1?
    starts = findall(i -> i == 1 || !isapprox(qabs_rounded[i-1], qabs_rounded[i]), eachindex(qabs_rounded))
    counts = starts[2:end] - starts[1:end-1]
    append!(counts, length(data_idcs) - starts[end] + 1)
    
    # Remove contiguous repetitions (same as original)
    qabs = qabs_rounded[starts]
    data_idcs = data_idcs[starts] 
    pos_idcs = pos_idcs[starts]
    
    return (; qabs, data_idcs, pos_idcs, counts)
end

"""
    intensities(qc::QuantumCorrelations, energies, qpts)

Compute dynamic structure factor S(q,ω) from quantum correlations.

# Arguments
- `qc`: QuantumCorrelations container with precomputed data.
- `energies`: Energy range used on external grid
- `qpts`: Q-points to evaluate (in reciprocal lattice units).

# Returns
- `Intensities` object with crystal, q-points, energies, and intensity values.

# Notes
- Energy range currently hardcoded (TODO: fix).
- Form factors must be uniform across observables.
"""
function intensities(qc::QuantumCorrelations, energies, qpts;kernel=nothing)
    # Determine energy information
    n_all_ω = size(qc.data, 7)

    # Extract min and max from the input energies and create a range with n_all_ω points, to match original time sampling of G
    ω_min, ω_max = extrema(energies)
    ωs = collect(range(ω_min, ω_max, length=n_all_ω))  # Now uses the same range as 'energies'
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

"""
    intensities_aux!(intensities, data, crystal, positions, combiner, ff_atoms, q_idx_info, ωidcs, ::Val{NCorr}, ::Val{NPos})

Core intensity calculation kernel (modifies `intensities` in-place). 
Should be almost identical to intensities_aux(SampledCorrelations) save for application of form factors taking into account other grid

# Arguments
- `intensities`: Output array (ω × q).
- `data`: Correlation data from `QuantumCorrelations`.
- `crystal`: Crystal structure information.
- `positions`: Atomic positions in unit cell.
- `combiner`: Function to combine correlation components from Measure Combiner.
- `ff_atoms`: Form factors per atom.
- `q_idx_info`: Preprocessed q-vector info from `pruned_wave_vector_info`.
- `ωidcs`: Indices of energy bins to include.
- `Val{NCorr}`: Number of correlation components (type-level).
- `Val{NPos}`: Number of atomic positions (type-level).
"""

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
