function vacuum_V41_SUN!(V41_buf::Array{ComplexF64, 6}, npt::NonPerturbativeTheory, bond::Bond, qs::NTuple{2, Vec3}, qs_indices::NTuple{2, CartesianIndex{3}}, φas::NTuple{4, Int}; opts...)
    V41_buf .= 0.0
    (; swt, Vps, clustersize, qs) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)

    q₁, q₂ = qs
    αs = (bond.i, bond.j)
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]

    Vq1 = view(Vps, :, :, qs_indices[1])
    Vq2 = view(Vps, :, :, qs_indices[2])

    for ci in CartesianIndices(clustersize)
        Vk = view(Vps, :, :, ci)
        k  = qs[ci]
        for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, σ₄ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, m in 1:L
            V41_buf[σ₁, σ₂, σ₃, σ₄, n₁, n₂] += Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vk[(α₂-1)*nflavors+σ₂, m]) * Vq1[(α₃-1)*nflavors+σ₃, n₁] * Vq2[(α₄-1)*nflavors+σ₄, n₂] * φ4((k, -k, q₁, q₂), φas, bond.n) +
            Vq1[(α₁-1)*nflavors+σ₁+L, n₁] * Vk[(α₂-1)*nflavors+σ₂+L, m] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * Vq2[(α₄-1)*nflavors+σ₄, n₂] * φ4((q₁, k, -k, q₂), φas, bond.n) +
            Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq1[(α₂-1)*nflavors+σ₂+L, n₁] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * Vq2[(α₄-1)*nflavors+σ₄, n₂] * φ4((k, q₁, -k, q₂), φas, bond.n) +
            Vq1[(α₁-1)*nflavors+σ₁+L, n₁] * Vq2[(α₂-1)*nflavors+σ₂+L, n₂] * Vk[(α₃-1)*nflavors+σ₃, m] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4((q₁, q₂, k, -k), φas, bond.n) +
            Vq1[(α₁-1)*nflavors+σ₁+L, n₁] * Vk[(α₂-1)*nflavors+σ₂+L, m] * Vq2[(α₃-1)*nflavors+σ₃, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4((q₁, k, q₂, -k), φas, bond.n) +
            Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq1[(α₂-1)*nflavors+σ₂+L, n₁] * Vq2[(α₃-1)*nflavors+σ₃, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4((k, q₁, q₂, -k), φas, bond.n)
        end
    end

end

function vacuum_V42_SUN!(V42_buf::Array{ComplexF64, 6}, npt::NonPerturbativeTheory, bond::Bond, qs::NTuple{2, Vec3}, qs_indices::NTuple{2, CartesianIndex{3}}, φas::NTuple{4, Int}; opts...)
    V42_buf .= 0.0
    (; swt, Vps, clustersize, qs) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)

    q₁, q₂ = qs
    αs = (bond.i, bond.j)
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]

    Vq1 = view(Vps, :, :, qs_indices[1])
    Vq2 = view(Vps, :, :, qs_indices[2])

    for ci in CartesianIndices(clustersize)
        Vk = view(Vps, :, :, ci)
        k  = qs[ci]
        for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, σ₄ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, m in 1:L
            V42_buf[σ₁, σ₂, σ₃, σ₄, n₁, n₂] += Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vk[(α₂-1)*nflavors+σ₂+L, m]) * Vq1[(α₃-1)*nflavors+σ₃, n₁] * Vq2[(α₄-1)*nflavors+σ₄, n₂] * φ4((k, -k, q₁, q₂), φas, bond.n) +
            Vq1[(α₁-1)*nflavors+σ₁+L, n₁] * Vk[(α₂-1)*nflavors+σ₂, m] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * Vq2[(α₄-1)*nflavors+σ₄, n₂] * φ4((q₁, k, -k, q₂), φas, bond.n) +
            Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq1[(α₂-1)*nflavors+σ₂, n₁] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * Vq2[(α₄-1)*nflavors+σ₄, n₂] * φ4((k, q₁, -k, q₂), φas, bond.n) +
            Vq1[(α₁-1)*nflavors+σ₁+L, n₁] * Vq2[(α₂-1)*nflavors+σ₂, n₂] * Vk[(α₃-1)*nflavors+σ₃, m] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4((q₁, q₂, k, -k), φas, bond.n) +
            Vq1[(α₁-1)*nflavors+σ₁+L, n₁] * Vk[(α₂-1)*nflavors+σ₂, m] * Vq2[(α₃-1)*nflavors+σ₃, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4((q₁, k, q₂, -k), φas, bond.n) +
            Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq1[(α₂-1)*nflavors+σ₂, n₁] * Vq2[(α₃-1)*nflavors+σ₃, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4((k, q₁, q₂, -k), φas, bond.n)
        end
    end

end

function vacuum_V43_SUN!(V43_buf::Array{ComplexF64, 6}, npt::NonPerturbativeTheory, bond::Bond, qs::NTuple{2, Vec3}, qs_indices::NTuple{2, CartesianIndex{3}}, φas::NTuple{4, Int}; opts...)
    V43_buf .= 0.0
    (; swt, Vps, clustersize, qs) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)

    q₁, q₂ = qs
    αs = (bond.i, bond.j)
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]

    Vq1 = view(Vps, :, :, qs_indices[1])
    Vq2 = view(Vps, :, :, qs_indices[2])

    for ci in CartesianIndices(clustersize)
        Vk = view(Vps, :, :, ci)
        k  = qs[ci]
        for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, σ₄ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, m in 1:L
            V43_buf[σ₁, σ₂, σ₃, σ₄, n₁, n₂] += Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vk[(α₂-1)*nflavors+σ₂, m]) * Vq1[(α₃-1)*nflavors+σ₃+L, n₁] * Vq2[(α₄-1)*nflavors+σ₄, n₂] * φ4((k, -k, q₁, q₂), φas, bond.n) +
            Vq1[(α₁-1)*nflavors+σ₁+L, n₁] * Vk[(α₂-1)*nflavors+σ₂+L, m] * conj(Vk[(α₃-1)*nflavors+σ₃, m]) * Vq2[(α₄-1)*nflavors+σ₄, n₂] * φ4((q₁, k, -k, q₂), φas, bond.n) +
            Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq1[(α₂-1)*nflavors+σ₂+L, n₁] * conj(Vk[(α₃-1)*nflavors+σ₃, m]) * Vq2[(α₄-1)*nflavors+σ₄, n₂] * φ4((k, q₁, -k, q₂), φas, bond.n) +
            Vq1[(α₁-1)*nflavors+σ₁+L, n₁] * Vq2[(α₂-1)*nflavors+σ₂+L, n₂] * Vk[(α₃-1)*nflavors+σ₃+L, m] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4((q₁, q₂, k, -k), φas, bond.n) +
            Vq1[(α₁-1)*nflavors+σ₁+L, n₁] * Vk[(α₂-1)*nflavors+σ₂+L, m] * Vq2[(α₃-1)*nflavors+σ₃+L, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4((q₁, k, q₂, -k), φas, bond.n) +
            Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq1[(α₂-1)*nflavors+σ₂+L, n₁] * Vq2[(α₃-1)*nflavors+σ₃+L, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4((k, q₁, q₂, -k), φas, bond.n)
        end
    end

end

function vacuum_V4_SUN(npt::NonPerturbativeTheory, qs::NTuple{2, Vec3}, qs_indices::NTuple{2, CartesianIndex{3}})
    (; swt, real_space_quartic_vertices, tensormode) = npt
    (; sys) = swt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(npt.swt)

    V4_tot = zeros(ComplexF64, L, L)
    V4_tot_buf = zeros(ComplexF64, L, L)
    V4 = zeros(ComplexF64, nflavors, nflavors, nflavors, nflavors, L, L)

    if tensormode == :loop
        i = 0
        for int in sys.interactions_union
            for coupling in int.pair
                coupling.isculled && break
                bond = coupling.bond
                i += 1

                V41 = real_space_quartic_vertices[i].V41
                V42 = real_space_quartic_vertices[i].V42
                V43 = real_space_quartic_vertices[i].V43

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (0, 1, 0, 1))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, σ₄ in 1:nflavors, n₁ in 1:L, n₂ in 1:L
                    V4_tot[n₁, n₂] += V41[σ₁, σ₂, σ₃, σ₄] * V4[σ₁, σ₃, σ₂, σ₄, n₁, n₂]
                end

                vacuum_V42_SUN!(V4, npt, bond, qs, qs_indices, (1, 1, 1, 0))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L
                    V4_tot[n₁, n₂] += V42[σ₁, σ₃] * V4[σ₂, σ₂, σ₃, σ₁, n₁, n₂]
                end

                vacuum_V43_SUN!(V4, npt, bond, qs, qs_indices, (0, 1, 1, 1))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L
                    V4_tot[n₁, n₂] += conj(V42[σ₁, σ₃]) * V4[σ₁, σ₃, σ₂, σ₂, n₁, n₂]
                end

                vacuum_V42_SUN!(V4, npt, bond, qs, qs_indices, (0, 0, 0, 1))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L
                    V4_tot[n₁, n₂] += V42[σ₁, σ₃] * V4[σ₂, σ₂, σ₁, σ₃, n₁, n₂]
                end

                vacuum_V43_SUN!(V4, npt, bond, qs, qs_indices, (1, 0, 0, 0))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L
                    V4_tot[n₁, n₂] += conj(V42[σ₁, σ₃]) * V4[σ₃, σ₁, σ₂, σ₂, n₁, n₂]
                end

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (0, 1, 1, 1))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L
                    V4_tot[n₁, n₂] += V43[σ₁, σ₃] * V4[σ₁, σ₂, σ₂, σ₃, n₁, n₂]
                end

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (1, 1, 1, 0))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L
                    V4_tot[n₁, n₂] += conj(V43[σ₁, σ₃]) * V4[σ₃, σ₂, σ₂, σ₁, n₁, n₂]
                end

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (0, 0, 0, 1))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L
                    V4_tot[n₁, n₂] += V43[σ₁, σ₃] * V4[σ₁, σ₂, σ₂, σ₃, n₁, n₂]
                end

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (1, 0, 0, 0))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L
                    V4_tot[n₁, n₂] += conj(V43[σ₁, σ₃]) * V4[σ₃, σ₂, σ₂, σ₁, n₁, n₂]
                end
            end
        end

    else
        i = 0
        for int in sys.interactions_union
            for coupling in int.pair
                coupling.isculled && break
                bond = coupling.bond
                i += 1

                V41 = real_space_quartic_vertices[i].V41
                V42 = real_space_quartic_vertices[i].V42
                V43 = real_space_quartic_vertices[i].V43

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (0, 1, 0, 1))
                @tensor V4_tot_buf[n₁, n₂] = V41[σ₁, σ₂, σ₃, σ₄] * V4[σ₁, σ₃, σ₂, σ₄, n₁, n₂]
                @. V4_tot += V4_tot_buf

                vacuum_V42_SUN!(V4, npt, bond, qs, qs_indices, (1, 1, 1, 0))
                @tensor V4_tot_buf[n₁, n₂] = V42[σ₁, σ₃] * V4[σ₂, σ₂, σ₃, σ₁, n₁, n₂]
                @. V4_tot += V4_tot_buf

                vacuum_V43_SUN!(V4, npt, bond, qs, qs_indices, (0, 1, 1, 1))
                @tensor V4_tot_buf[n₁, n₂] = conj(V42[σ₁, σ₃]) * V4[σ₁, σ₃, σ₂, σ₂, n₁, n₂]
                @. V4_tot += V4_tot_buf

                vacuum_V42_SUN!(V4, npt, bond, qs, qs_indices, (0, 0, 0, 1))
                @tensor V4_tot_buf[n₁, n₂] = V42[σ₁, σ₃] * V4[σ₂, σ₂, σ₁, σ₃, n₁, n₂]
                @. V4_tot += V4_tot_buf

                vacuum_V43_SUN!(V4, npt, bond, qs, qs_indices, (1, 0, 0, 0))
                @tensor V4_tot_buf[n₁, n₂] = conj(V42[σ₁, σ₃]) * V4[σ₃, σ₁, σ₂, σ₂, n₁, n₂]
                @. V4_tot += V4_tot_buf

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (0, 1, 1, 1))
                @tensor V4_tot_buf[n₁, n₂] = V43[σ₁, σ₃] * V4[σ₁, σ₂, σ₂, σ₃, n₁, n₂]
                @. V4_tot += V4_tot_buf

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (1, 1, 1, 0))
                @tensor V4_tot_buf[n₁, n₂] = conj(V43[σ₁, σ₃]) * V4[σ₃, σ₂, σ₂, σ₁, n₁, n₂]
                @. V4_tot += V4_tot_buf

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (0, 0, 0, 1))
                @tensor V4_tot_buf[n₁, n₂] = V43[σ₁, σ₃] * V4[σ₁, σ₂, σ₂, σ₃, n₁, n₂]
                @. V4_tot += V4_tot_buf

                vacuum_V41_SUN!(V4, npt, bond, qs, qs_indices, (1, 0, 0, 0))
                @tensor V4_tot_buf[n₁, n₂] = conj(V43[σ₁, σ₃]) * V4[σ₃, σ₂, σ₂, σ₁, n₁, n₂]
                @. V4_tot += V4_tot_buf
            end
        end
    end

    return V4_tot
end