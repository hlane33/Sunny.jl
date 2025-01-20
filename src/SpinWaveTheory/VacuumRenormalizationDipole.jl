function vacuum_V41_dipole!(V41_buf::Array{ComplexF64, 2}, npt::NonPerturbativeTheory, bond::Bond, qs::NTuple{2, Vec3}, qs_indices::NTuple{2, CartesianIndex{3}}, φas::NTuple{4, Int}; opts...)
    V41_buf .= 0.0
    (; swt, Vps, clustersize, qs) = npt
    L = nbands(swt)

    q₁, q₂ = qs
    αs = (bond.i, bond.j)
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]

    Vq1 = view(Vps, :, :, qs_indices[1])
    Vq2 = view(Vps, :, :, qs_indices[2])

    for ci in CartesianIndices(clustersize)
        Vk = view(Vps, :, :, ci)
        k  = qs[ci]
        for n₁ in 1:L, n₂ in 1:L, m in 1:L
            V41_buf[n₁, n₂] += Vk[α₁+L, m] * conj(Vk[α₂, m]) * Vq1[α₃, n₁] * Vq2[α₄, n₂] * φ4((k, -k, q₁, q₂), φas, bond.n) +
            Vq1[α₁+L, n₁] * Vk[α₂+L, m] * conj(Vk[α₃+L, m]) * Vq2[α₄, n₂] * φ4((q₁, k, -k, q₂), φas, bond.n) +
            Vk[α₁+L, m] * Vq1[α₂+L, n₁] * conj(Vk[α₃+L, m]) * Vq2[α₄, n₂] * φ4((k, q₁, -k, q₂), φas, bond.n) +
            Vq1[α₁+L, n₁] * Vq2[α₂+L, n₂] * Vk[α₃, m] * conj(Vk[α₄+L, m]) * φ4((q₁, q₂, k, -k), φas, bond.n) +
            Vq1[α₁+L, n₁] * Vk[α₂+L, m] * Vq2[α₃, n₂] * conj(Vk[α₄+L, m]) * φ4((q₁, k, q₂, -k), φas, bond.n) +
            Vk[α₁+L, m] * Vq1[α₂+L, n₁] * Vq2[α₃, n₂] * conj(Vk[α₄+L, m]) * φ4((k, q₁, q₂, -k), φas, bond.n)
        end
    end

end

function vacuum_V42_dipole!(V42_buf::Array{ComplexF64, 2}, npt::NonPerturbativeTheory, bond::Bond, qs::NTuple{2, Vec3}, qs_indices::NTuple{2, CartesianIndex{3}}, φas::NTuple{4, Int}; opts...)
    V42_buf .= 0.0
    (; swt, Vps, clustersize, qs) = npt
    L = nbands(swt)

    q₁, q₂ = qs
    αs = (bond.i, bond.j)
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]

    Vq1 = view(Vps, :, :, qs_indices[1])
    Vq2 = view(Vps, :, :, qs_indices[2])

    for ci in CartesianIndices(clustersize)
        Vk = view(Vps, :, :, ci)
        k  = qs[ci]
        for n₁ in 1:L, n₂ in 1:L, m in 1:L
            V42_buf[n₁, n₂] += Vk[α₁+L, m] * conj(Vk[α₂+L, m]) * Vq1[α₃, n₁] * Vq2[α₄, n₂] * φ4((k, -k, q₁, q₂), φas, bond.n) +
            Vq1[α₁+L, n₁] * Vk[α₂, m] * conj(Vk[α₃+L, m]) * Vq2[α₄, n₂] * φ4((q₁, k, -k, q₂), φas, bond.n) +
            Vk[α₁+L, m] * Vq1[α₂, n₁] * conj(Vk[α₃+L, m]) * Vq2[α₄, n₂] * φ4((k, q₁, -k, q₂), φas, bond.n) +
            Vq1[α₁+L, n₁] * Vq2[α₂, n₂] * Vk[α₃, m] * conj(Vk[α₄+L, m]) * φ4((q₁, q₂, k, -k), φas, bond.n) +
            Vq1[α₁+L, n₁] * Vk[α₂, m] * Vq2[α₃, n₂] * conj(Vk[α₄+L, m]) * φ4((q₁, k, q₂, -k), φas, bond.n) +
            Vk[α₁+L, m] * Vq1[α₂, n₁] * Vq2[α₃, n₂] * conj(Vk[α₄+L, m]) * φ4((k, q₁, q₂, -k), φas, bond.n)
        end
    end

end

function vacuum_V43_dipole!(V43_buf::Array{ComplexF64, 2}, npt::NonPerturbativeTheory, bond::Bond, qs::NTuple{2, Vec3}, qs_indices::NTuple{2, CartesianIndex{3}}, φas::NTuple{4, Int}; opts...)
    V43_buf .= 0.0
    (; swt, Vps, clustersize, qs) = npt
    L = nbands(swt)

    q₁, q₂ = qs
    αs = (bond.i, bond.j)
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]

    Vq1 = view(Vps, :, :, qs_indices[1])
    Vq2 = view(Vps, :, :, qs_indices[2])

    for ci in CartesianIndices(clustersize)
        Vk = view(Vps, :, :, ci)
        k  = qs[ci]
        for n₁ in 1:L, n₂ in 1:L, m in 1:L
            V43_buf[n₁, n₂] += Vk[α₁+L, m] * conj(Vk[α₂, m]) * Vq1[α₃+L, n₁] * Vq2[α₄, n₂] * φ4((k, -k, q₁, q₂), φas, bond.n) +
            Vq1[α₁+L, n₁] * Vk[α₂+L, m] * conj(Vk[α₃, m]) * Vq2[α₄, n₂] * φ4((q₁, k, -k, q₂), φas, bond.n) +
            Vk[α₁+L, m] * Vq1[α₂+L, n₁] * conj(Vk[α₃, m]) * Vq2[α₄, n₂] * φ4((k, q₁, -k, q₂), φas, bond.n) +
            Vq1[α₁+L, n₁] * Vq2[α₂+L, n₂] * Vk[α₃+L, m] * conj(Vk[α₄+L, m]) * φ4((q₁, q₂, k, -k), φas, bond.n) +
            Vq1[α₁+L, n₁] * Vk[α₂+L, m] * Vq2[α₃+L, n₂] * conj(Vk[α₄+L, m]) * φ4((q₁, k, q₂, -k), φas, bond.n) +
            Vk[α₁+L, m] * Vq1[α₂+L, n₁] * Vq2[α₃+L, n₂] * conj(Vk[α₄+L, m]) * φ4((k, q₁, q₂, -k), φas, bond.n)
        end
    end

end

function vacuum_V4_dipole(npt::NonPerturbativeTheory, qs::NTuple{2, Vec3}, qs_indices::NTuple{2, CartesianIndex{3}})
    (; swt, real_space_quartic_vertices) = npt
    (; sys, data) = swt
    (; stevens_coefs) = data

    L = nbands(npt.swt)

    V4_tot = zeros(ComplexF64, L, L)
    V4 = zeros(ComplexF64, L, L)

    for i in 1:L
        (; c2, c4, c6) = stevens_coefs[i]
        @assert iszero(c2) "Rank 2 Stevens operators not supported in :dipole non-perturbative calculations yet"
        @assert iszero(c4) "Rank 4 Stevens operators not supported in :dipole non-perturbative calculations yet"
        @assert iszero(c6) "Rank 6 Stevens operators not supported in :dipole non-perturbative calculations yet"
    end

    i = 0
    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            bond = coupling.bond
            i += 1

            V41 = real_space_quartic_vertices[i].V41
            V42 = real_space_quartic_vertices[i].V42
            V43 = real_space_quartic_vertices[i].V43

            vacuum_V41_dipole!(V4, npt, bond, qs, qs_indices, (0, 1, 0, 1))
            @. V4_tot += V41 * V4

            vacuum_V42_dipole!(V4, npt, bond, qs, qs_indices, (1, 1, 1, 0))
            @. V4_tot += V42 * V4

            vacuum_V43_dipole!(V4, npt, bond, qs, qs_indices, (0, 1, 1, 1))
            @. V4_tot += conj(V42) * V4

            vacuum_V41_dipole!(V4, npt, bond, qs, qs_indices, (1, 1, 1, 0))
            @. V4_tot += V43 * V4

            vacuum_V41_dipole!(V4, npt, bond, qs, qs_indices, (0, 1, 1, 1))
            @. V4_tot += conj(V43) * V4

            vacuum_V42_dipole!(V4, npt, bond, qs, qs_indices, (0, 0, 0, 1))
            @. V4_tot += V42 * V4

            vacuum_V43_dipole!(V4, npt, bond, qs, qs_indices, (1, 0, 0, 0))
            @. V4_tot += conj(V42) * V4

            vacuum_V41_dipole!(V4, npt, bond, qs, qs_indices, (1, 0, 0, 0))
            @. V4_tot += V43 * V4

            vacuum_V41_dipole!(V4, npt, bond, qs, qs_indices, (0, 0, 0, 1))
            @. V4_tot += conj(V43) * V4
        end
    end

    return V4_tot
end