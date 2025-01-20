function φ3(qs, φas::NTuple{3, Int}, n)
    ret = 1.0 + 0.0im
    for i in 1:3
        ret *= exp(2π*im * φas[i] * dot(qs[i], n))
    end
    return ret
end

function cubic_U31_SUN!(U31_buf::Array{ComplexF64, 6}, npt::NonPerturbativeTheory, bond::Bond, qs::NTuple{3, Vec3}, qs_indices::NTuple{3, CartesianIndex{3}}, φas::NTuple{3, Int})
    U31_buf .= 0.0
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)
    q₁, q₂, q₃ = qs

    αs = (bond.i, bond.j)
    α₁, α₂, α₃ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1]

    phase1 = φ3((q₁, q₂, q₃), φas, bond.n)
    phase2 = φ3((q₂, q₁, q₃), φas, bond.n)
    phase3 = φ3((q₃, q₂, q₁), φas, bond.n)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U31_buf[σ₁, σ₂, σ₃, n₁, n₂, n₃] += conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * Vp2[(α₂-1)*nflavors+σ₂, n₂] * Vp3[(α₃-1)*nflavors+σ₃, n₃] * phase1 +
        Vp2[(α₁-1)*nflavors+σ₁+L, n₂] * conj(Vp1[(α₂-1)*nflavors+σ₂+L, n₁]) * Vp3[(α₃-1)*nflavors+σ₃, n₃] * phase2 +
        Vp3[(α₁-1)*nflavors+σ₁+L, n₃] * Vp2[(α₂-1)*nflavors+σ₂, n₂] * conj(Vp1[(α₃-1)*nflavors+σ₃+L, n₁]) * phase3
    end
end

function cubic_U32_SUN!(U32_buf::Array{ComplexF64, 6}, npt::NonPerturbativeTheory, bond::Bond, qs::NTuple{3, Vec3}, qs_indices::NTuple{3, CartesianIndex{3}}, φas::NTuple{3, Int})
    U32_buf .= 0.0
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)
    q₁, q₂, q₃ = qs

    αs = (bond.i, bond.j)
    α₁, α₂, α₃ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1]

    phase1 = φ3((q₁, q₂, q₃), φas, bond.n)
    phase2 = φ3((q₂, q₁, q₃), φas, bond.n)
    phase3 = φ3((q₃, q₂, q₁), φas, bond.n)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U32_buf[σ₁, σ₂, σ₃, n₁, n₂, n₃] += conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * Vp2[(α₂-1)*nflavors+σ₂+L, n₂] * Vp3[(α₃-1)*nflavors+σ₃, n₃] * phase1 +
        Vp2[(α₁-1)*nflavors+σ₁+L, n₂] * conj(Vp1[(α₂-1)*nflavors+σ₂, n₁]) * Vp3[(α₃-1)*nflavors+σ₃, n₃] * phase2 +
        Vp3[(α₁-1)*nflavors+σ₁+L, n₃] * Vp2[(α₂-1)*nflavors+σ₂+L, n₂] * conj(Vp1[(α₃-1)*nflavors+σ₃+L, n₁]) * phase3
    end
end

function cubic_U3_symmetrized_SUN!(U3::Array{ComplexF64, 6}, U3_buf::Array{ComplexF64, 6}, U3_buf_perm::Array{ComplexF64, 6}, cubic_fun::Function, npt::NonPerturbativeTheory, bond::Bond, qs::NTuple{3, Vec3}, qs_indices::NTuple{3, CartesianIndex{3}}, φas::NTuple{3, Int})
    U3 .= 0.0
    U3_buf .= 0.0
    U3_buf_perm .= 0.0
    q₁, q₂, q₃ = qs
    iq₁, iq₂, iq₃ = qs_indices

    cubic_fun(U3_buf, npt, bond, qs, qs_indices, φas)
    U3 .+= U3_buf

    cubic_fun(U3_buf, npt, bond, (q₁, q₃, q₂), (iq₁, iq₃, iq₂), φas)
    permutedims!(U3_buf_perm, U3_buf, (1, 2, 3, 4, 6, 5))
    U3 .+= U3_buf_perm

    return U3
end

function cubic_U31′_SUN!(U31_buf::Array{ComplexF64, 6}, npt::NonPerturbativeTheory, qs_indices::NTuple{3, CartesianIndex{3}}, α::Int)
    U31_buf .= 0.0
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U31_buf[σ₁, σ₂, σ₃, n₁, n₂, n₃] += conj(Vp1[(α-1)*nflavors+σ₁, n₁]) * Vp2[(α-1)*nflavors+σ₂, n₂] * Vp3[(α-1)*nflavors+σ₃, n₃] +
        Vp2[(α-1)*nflavors+σ₁+L, n₂] * conj(Vp1[(α-1)*nflavors+σ₂+L, n₁]) * Vp3[(α-1)*nflavors+σ₃, n₃] +
        Vp3[(α-1)*nflavors+σ₁+L, n₃] * Vp2[(α-1)*nflavors+σ₂, n₂] * conj(Vp1[(α-1)*nflavors+σ₃+L, n₁])
    end
end

function cubic_U32′_SUN!(U32_buf::Array{ComplexF64, 6}, npt::NonPerturbativeTheory, qs_indices::NTuple{3, CartesianIndex{3}}, α::Int)
    U32_buf .= 0.0
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U32_buf[σ₁, σ₂, σ₃, n₁, n₂, n₃] += conj(Vp1[(α-1)*nflavors+σ₁, n₁]) * Vp2[(α-1)*nflavors+σ₂+L, n₂] * Vp3[(α-1)*nflavors+σ₃, n₃] +
        Vp2[(α-1)*nflavors+σ₁+L, n₂] * conj(Vp1[(α-1)*nflavors+σ₂, n₁]) * Vp3[(α-1)*nflavors+σ₃, n₃] +
        Vp3[(α-1)*nflavors+σ₁+L, n₃] * Vp2[(α-1)*nflavors+σ₂+L, n₂] * conj(Vp1[(α-1)*nflavors+σ₃+L, n₁])
    end
end

function cubic_U3′_symmetrized_SUN!(U3::Array{ComplexF64, 6}, U3_buf::Array{ComplexF64, 6}, U3_buf_perm::Array{ComplexF64, 6}, cubic_fun::Function, npt::NonPerturbativeTheory, qs_indices::NTuple{3, CartesianIndex{3}}, α::Int)
    U3 .= 0.0
    U3_buf .= 0.0
    U3_buf_perm .= 0.0

    iq₁, iq₂, iq₃ = qs_indices

    cubic_fun(U3_buf, npt, qs_indices, α)
    U3 .+= U3_buf

    cubic_fun(U3_buf, npt, (iq₁, iq₃, iq₂), α)
    permutedims!(U3_buf_perm, U3_buf, (1, 2, 3, 4, 6, 5))
    U3 .+= U3_buf_perm

    return U3
end

function cubic_vertex_SUN(npt::NonPerturbativeTheory, qs::NTuple{3, Vec3}, qs_indices::NTuple{3, CartesianIndex{3}})
    (; swt, real_space_cubic_vertices, tensormode) = npt
    (; sys) = swt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(npt.swt)

    U3_tot = zeros(ComplexF64, L, L, L)
    U3_tot_buf = zeros(ComplexF64, L, L, L)
    U3 = zeros(ComplexF64, nflavors, nflavors, nflavors, L, L, L)
    U3_buf = zeros(ComplexF64, nflavors, nflavors, nflavors, L, L, L)
    U3_buf_perm  = zeros(ComplexF64, nflavors, nflavors, nflavors, L, L, L)

    if tensormode == :loop
        i = 0
        for (atom, int) in enumerate(sys.interactions_union)
            cubic_U3′_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31′_SUN!, npt, qs_indices, atom)
            op = int.onsite[N, 1:N-1]
            for σ₁ in 1:nflavors, σ₂ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                U3_tot[n₁, n₂, n₃] = -0.5 * op[σ₁] * U3[σ₂, σ₂, σ₁, n₁, n₂, n₃]
            end

            cubic_U3′_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32′_SUN!, npt, qs_indices, atom)
            for σ₁ in 1:nflavors, σ₂ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                U3_tot[n₁, n₂, n₃] = -0.5 * conj(op[σ₁]) * U3[σ₁, σ₂, σ₂, n₁, n₂, n₃]
            end

            for coupling in int.pair
                coupling.isculled && break
                bond = coupling.bond
                i += 1

                V31_p = real_space_cubic_vertices[i].V31_p
                V31_m = real_space_cubic_vertices[i].V31_m
                V32_p = real_space_cubic_vertices[i].V32_p
                V32_m = real_space_cubic_vertices[i].V32_m

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31_SUN!, npt, bond, qs, qs_indices, (1, 1, 1))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                    U3_tot[n₁, n₂, n₃] += V31_p[σ₁] * U3[σ₂, σ₂, σ₁, n₁, n₂, n₃]
                end

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31_SUN!, npt, bond, qs, qs_indices, (0, 0, 0))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                    U3_tot[n₁, n₂, n₃] += V31_m[σ₁] * U3[σ₂, σ₂, σ₁, n₁, n₂, n₃]
                end

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32_SUN!, npt, bond, qs, qs_indices, (1, 1, 1))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                    U3_tot[n₁, n₂, n₃] += conj(V31_p[σ₁]) * U3[σ₁, σ₂, σ₂, n₁, n₂, n₃]
                end

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32_SUN!, npt, bond, qs, qs_indices, (0, 0, 0))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                    U3_tot[n₁, n₂, n₃] += conj(V31_m[σ₁]) * U3[σ₁, σ₂, σ₂, n₁, n₂, n₃]
                end

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31_SUN!, npt, bond, qs, qs_indices, (1, 1, 0))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                    U3_tot[n₁, n₂, n₃] += V32_p[σ₁, σ₂, σ₃] * U3[σ₂, σ₃, σ₁, n₁, n₂, n₃]
                end

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31_SUN!, npt, bond, qs, qs_indices, (0, 0, 1))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                    U3_tot[n₁, n₂, n₃] += V32_m[σ₁, σ₂, σ₃] * U3[σ₂, σ₃, σ₁, n₁, n₂, n₃]
                end

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32_SUN!, npt, bond, qs, qs_indices, (0, 1, 1))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                    U3_tot[n₁, n₂, n₃] += conj(V32_p[σ₁, σ₂, σ₃]) * U3[σ₁, σ₃, σ₂, n₁, n₂, n₃]
                end

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32_SUN!, npt, bond, qs, qs_indices, (1, 0, 0))
                for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
                    U3_tot[n₁, n₂, n₃] += conj(V32_m[σ₁, σ₂, σ₃]) * U3[σ₁, σ₃, σ₂, n₁, n₂, n₃]
                end
            end
        end

    else
        i = 0
        for (atom, int) in enumerate(sys.interactions_union)
            cubic_U3′_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31′_SUN!, npt, qs_indices, atom)
            op = int.onsite[N, 1:N-1]
            @tensor U3_tot_buf[n₁, n₂, n₃] = -0.5 * op[σ₁] * U3[σ₂, σ₂, σ₁, n₁, n₂, n₃]
            U3_tot .+= U3_tot_buf

            cubic_U3′_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32′_SUN!, npt, qs_indices, atom)
            @tensor U3_tot_buf[n₁, n₂, n₃] = -0.5 * conj(op[σ₁]) * U3[σ₁, σ₂, σ₂, n₁, n₂, n₃]
            U3_tot .+= U3_tot_buf

            for coupling in int.pair
                coupling.isculled && break
                bond = coupling.bond
                i += 1

                V31_p = real_space_cubic_vertices[i].V31_p
                V31_m = real_space_cubic_vertices[i].V31_m
                V32_p = real_space_cubic_vertices[i].V32_p
                V32_m = real_space_cubic_vertices[i].V32_m

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31_SUN!, npt, bond, qs, qs_indices, (1, 1, 1))
                @tensor U3_tot_buf[n₁, n₂, n₃] = V31_p[σ₁] * U3[σ₂, σ₂, σ₁, n₁, n₂, n₃]
                U3_tot .+= U3_tot_buf

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31_SUN!, npt, bond, qs, qs_indices, (0, 0, 0))
                @tensor U3_tot_buf[n₁, n₂, n₃] = V31_m[σ₁] * U3[σ₂, σ₂, σ₁, n₁, n₂, n₃]
                U3_tot .+= U3_tot_buf

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32_SUN!, npt, bond, qs, qs_indices, (1, 1, 1))
                @tensor U3_tot_buf[n₁, n₂, n₃] = conj(V31_p[σ₁]) * U3[σ₁, σ₂, σ₂, n₁, n₂, n₃]
                U3_tot .+= U3_tot_buf

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32_SUN!, npt, bond, qs, qs_indices, (0, 0, 0))
                @tensor U3_tot_buf[n₁, n₂, n₃] = conj(V31_m[σ₁]) * U3[σ₁, σ₂, σ₂, n₁, n₂, n₃]
                U3_tot .+= U3_tot_buf

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31_SUN!, npt, bond, qs, qs_indices, (1, 1, 0))
                @tensor U3_tot_buf[n₁, n₂, n₃] = V32_p[σ₁, σ₂, σ₃] * U3[σ₂, σ₃, σ₁, n₁, n₂, n₃]
                U3_tot .+= U3_tot_buf

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U31_SUN!, npt, bond, qs, qs_indices, (0, 0, 1))
                @tensor U3_tot_buf[n₁, n₂, n₃] = V32_m[σ₁, σ₂, σ₃] * U3[σ₂, σ₃, σ₁, n₁, n₂, n₃]
                U3_tot .+= U3_tot_buf

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32_SUN!, npt, bond, qs, qs_indices, (0, 1, 1))
                @tensor U3_tot_buf[n₁, n₂, n₃] = conj(V32_p[σ₁, σ₂, σ₃]) * U3[σ₁, σ₃, σ₂, n₁, n₂, n₃]
                U3_tot .+= U3_tot_buf

                cubic_U3_symmetrized_SUN!(U3, U3_buf, U3_buf_perm, cubic_U32_SUN!, npt, bond, qs, qs_indices, (1, 0, 0))
                @tensor U3_tot_buf[n₁, n₂, n₃] = conj(V32_m[σ₁, σ₂, σ₃]) * U3[σ₁, σ₃, σ₂, n₁, n₂, n₃]
                U3_tot .+= U3_tot_buf
            end
        end

    end

    return U3_tot
end