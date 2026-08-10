function mean_field_hamiltonians(sys::ElectronicSystem,q_reshaped::Vec3)
    L = 2*length(sys.mean_fields)
    H = zeros(ComplexF64, L, L)
    Hvu = zeros(ComplexF64,L,L)
    Hvd = zeros(ComplexF64,L,L)
    for (i, int) in enumerate(sys.interactions_union)
        # Pair interactions
        for coupling in int.pair
            (; isculled, bond) = coupling
            isculled && break
            (; i, j) = bond
            phase = exp(2π*im * dot(q_reshaped, bond.n)) # Phase associated with periodic wrapping
            if !iszero(coupling.hopping)
                t = coupling.hopping
                H[i, j] += t * phase
                H[j, i] += conj(t) * conj(phase)
            end
        end
    end

    return 
end

function dynamical_matrix_v2(sys::System,k,t,U,Vs,Δs)
    L = 2*length(sys.dipoles)
    H = zeros(ComplexF64,L,L)
    for (i, int) in enumerate(sys.interactions_union)
        for coupling in int.pair
            (; isculled, bond) = coupling
            isculled && break

            @assert i == bond.i
            j = bond.j

            phase = exp(2π*im * dot(k, bond.n)) # Phase associated with periodic wrapping

            # Bilinear exchange
            if !iszero(coupling.bilin)
                H[i, j] += -t * phase
                H[j, i] += -conj(t) * conj(phase)
            end
        end
    end
    return H
end