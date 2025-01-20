function swt_hamiltonian_dipole_nlsw!(H::Matrix{ComplexF64}, ptt::PerturbativeTheory, q_reshaped::Vec3)
    (; swt, mean_field_values, real_space_quartic_vertices) = ptt
    (; sys, data) = swt
    (; stevens_coefs) = data

    L = nbands(swt)
    @assert size(H) == (2L, 2L)

    # Initialize Hamiltonian buffer 
    # Note that H11 for b†b, H22 for bb†, H12 for b†b†, and H21 for bb
    H .= 0.0 
    H11 = view(H, 1:L, 1:L)
    H12 = view(H, 1:L, L+1:2L)
    H21 = view(H, L+1:2L, 1:L)
    H22 = view(H, L+1:2L, L+1:2L)

    index = 0
    for (i, int) in enumerate(sys.interactions_union)

        # Single-ion anisotropy
        (; c2, c4, c6) = stevens_coefs[i]
        @assert iszero(c2) "Rank 2 Stevens operators not supported in :dipole non-perturbative calculations yet"
        @assert iszero(c4) "Rank 4 Stevens operators not supported in :dipole non-perturbative calculations yet"
        @assert iszero(c6) "Rank 6 Stevens operators not supported in :dipole non-perturbative calculations yet"

        # Pair interactions
        for coupling in int.pair
            (; isculled, bond) = coupling
            isculled && break

            index += 1
            (; Nii, Njj, Nij, Δii, Δjj, Δij) = mean_field_values[index]
            (; V41, V42, V43) = real_space_quartic_vertices[index]

            (; i, j) = bond
            phase = exp(2π*im * dot(q_reshaped, bond.n)) # Phase associated with periodic wrapping

            # Bilinear exchange
            if !iszero(coupling.bilin)

                Q = V41 * conj(Nij) + V42 * Δii + conj(V42) * conj(Δjj) + 2 * conj(V43) * (Nii + Njj)

                H11[i, j] += Q * phase
                H11[j, i] += conj(Q) * conj(phase)
                H22[i, j] += conj(Q) * phase
                H22[j, i] += Q  * conj(phase)

                P = V41 * conj(Δij) + 2 * V42 * (Nii + Njj) + V43 * conj(Δjj) + conj(V43) * conj(Δii)

                H21[i, j] += P * phase
                H21[j, i] += P * conj(phase)
                H12[i, j] += conj(P) * phase
                H12[j, i] += conj(P) * conj(phase)

                Qi = V41 * Njj + 2 * V42 * Δij + 2 * conj(V42) * conj(Δij) + 2 * V43 * conj(Nij) + 2 * conj(V43) * Nij
                Qj = V41 * Nii + 2 * V42 * Δij + 2 * conj(V42) * conj(Δij) + 2 * V43 * conj(Nij) + 2 * conj(V43) * Nij
                H11[i, i] += Qi
                H11[j, j] += Qj
                H22[i, i] += Qi
                H22[j, j] += Qj

                Pi = V42 * Nij + V43 * conj(Δij)
                Pj = V42 * conj(Nij) + conj(V43) * conj(Δij)
                H21[i, i] += Pi
                H21[j, j] += Pj
                H12[i, i] += conj(Pi)
                H12[j, j] += conj(Pj)
            end

            # Biquadratic exchange
            if !iszero(coupling.biquad)
                @error "Biquadratic exchange not supported in :dipole perturbative calculations yet"
            end
        end

    end
end