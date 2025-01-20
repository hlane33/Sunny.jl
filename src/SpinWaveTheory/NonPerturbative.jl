struct RealSpaceQuarticVerticesSUN
    V41 :: Array{ComplexF64, 4}
    V42 :: Array{ComplexF64, 2}
    V43 :: Array{ComplexF64, 2}
end

struct RealSpaceQuarticVerticesDipole
    V41 :: ComplexF64
    V42 :: ComplexF64
    V43 :: ComplexF64
end

struct RealSpaceCubicVerticesSUN
    V31_p :: Vector{ComplexF64}
    V31_m :: Vector{ComplexF64}
    V32_p :: Array{ComplexF64, 3}
    V32_m :: Array{ComplexF64, 3}
end

struct RealSpaceCubicVerticesDipole
    V31 :: ComplexF64
    V32 :: ComplexF64
end

struct NonPerturbativeTheory
    swt :: SpinWaveTheory
    clustersize :: NTuple{3, Int}   # Cluster size for number of magnetic unit cell
    qs :: Array{Vec3, 3}
    Es :: Array{Float64, 4}
    Vps :: Array{ComplexF64, 5}
    real_space_quartic_vertices :: Vector{Union{RealSpaceQuarticVerticesSUN, RealSpaceQuarticVerticesDipole}}
    real_space_cubic_vertices   :: Vector{Union{RealSpaceCubicVerticesSUN, RealSpaceCubicVerticesDipole}}
end

function calculate_real_space_quartic_vertices_sun(sys::System)
    N = sys.Ns[1]
    V41_buf = zeros(ComplexF64, N-1, N-1, N-1, N-1)
    V42_buf = zeros(ComplexF64, N-1, N-1)
    V43_buf = zeros(ComplexF64, N-1, N-1)

    real_space_quartic_vertices = RealSpaceQuarticVerticesSUN[]

    for int in sys.interactions_union
        for coupling in int.pair

            coupling.isculled && break
            V41_buf .= 0.0
            V42_buf .= 0.0
            V43_buf .= 0.0

            for (A, B) in coupling.general.data
                for σ1 in 1:N-1, σ3 in 1:N-1
                    V42_buf[σ1, σ3] += -0.5 * A[N, σ1] * B[N, σ3]
                    V43_buf[σ1, σ3] += -0.5 * A[σ1, N] * B[N, σ3]
                    for σ2 in 1:N-1, σ4 in 1:N-1
                        V41_buf[σ1, σ2, σ3, σ4] += (A[σ1, σ2] - δ(σ1, σ2)*A[N, N]) * (B[σ3, σ4] - δ(σ3, σ4)*B[N, N])
                    end
                end
            end

            quartic_vertices = RealSpaceQuarticVerticesSUN(copy(V41_buf), copy(V42_buf), copy(V43_buf))
            push!(real_space_quartic_vertices, quartic_vertices)
        end
    end

    return real_space_quartic_vertices
end

function calculate_real_space_quartic_vertices_dipole(sys::System)
    N = sys.Ns[1]
    S = (N-1)/2
    real_space_quartic_vertices = RealSpaceQuarticVerticesDipole[]

    for int in sys.interactions_union
        for coupling in int.pair
            (; isculled, bilin, biquad, general) = coupling

            isculled && break
            J = Mat3(bilin*I)
            V41 = J[3, 3]
            V42 = 1/8 * (-J[1, 1] + J[2, 2] + 1im*J[1, 2] + 1im*J[2, 1])
            V43 = 1/8 * (-J[1, 1] - J[2, 2] - 1im*J[1, 2] + 1im*J[2, 1])
            quartic_vertices = RealSpaceQuarticVerticesDipole(V41, V42, V43)
            push!(real_space_quartic_vertices, quartic_vertices)

            @assert iszero(biquad) "Biquadratic interactions not supported in :dipole_large_S for the non-perburbative calculation."
            @assert isempty(general.data)
        end
    end

    return real_space_quartic_vertices
end

function calculate_real_space_cubic_vertices_sun(sys::System)
    N = sys.Ns[1]
    V31_p_buf = zeros(ComplexF64, N-1)
    V31_m_buf = zeros(ComplexF64, N-1)
    V32_p_buf = zeros(ComplexF64, N-1, N-1, N-1)
    V32_m_buf = zeros(ComplexF64, N-1, N-1, N-1)

    real_space_cubic_vertices = RealSpaceCubicVerticesSUN[]

    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            V31_p_buf .= 0.0
            V31_m_buf .= 0.0
            V32_p_buf .= 0.0
            V32_m_buf .= 0.0

            for (A, B) in coupling.general.data
                for σ1 in 1:N-1
                    V31_p_buf[σ1] += -0.5 * A[N, N] * B[N, σ1]
                    V31_m_buf[σ1] += -0.5 * B[N, N] * A[N, σ1]
                    for σ2 in 1:N-1, σ3 in 1:N-1
                        V32_p_buf[σ1, σ2, σ3] += A[N, σ1] * (B[σ2, σ3] - δ(σ2, σ3)*B[N, N])
                        V32_m_buf[σ1, σ2, σ3] += B[N, σ1] * (A[σ2, σ3] - δ(σ2, σ3)*A[N, N])
                    end
                end
            end

            cubic_vertices = RealSpaceCubicVerticesSUN(copy(V31_p_buf), copy(V31_m_buf), copy(V32_p_buf), copy(V32_m_buf))
            push!(real_space_cubic_vertices, cubic_vertices)
        end
    end

    return real_space_cubic_vertices
end

function calculate_real_space_cubic_vertices_dipole(sys::System)
    N = sys.Ns[1]
    S = (N-1)/2
    real_space_cubic_vertices = RealSpaceCubicVerticesDipole[]

    for int in sys.interactions_union
        for coupling in int.pair
            (; isculled, bilin, biquad, general) = coupling

            isculled && break
            J = Mat3(bilin*I)
            V31 = √(S/2) * (-J[1, 3] + 1im*J[2, 3] )
            V32 = √(S/2) * (-J[3, 1] + 1im*J[3, 2] )

            cubic_vertices = RealSpaceCubicVerticesDipole(V31, V32)
            push!(real_space_cubic_vertices, cubic_vertices)

            @assert iszero(biquad) "Biquadratic interactions not supported in :dipole_large_S for the non-perburbative calculation."
            @assert isempty(general.data)
        end
    end

    return real_space_cubic_vertices
end

function NonPerturbativeTheory(swt::SpinWaveTheory, clustersize::NTuple{3, Int})
    (; sys) = swt
    @assert sys.mode in (:SUN, :dipole) "Non-perturbative calculation is only supported in :SUN or :dipole mode."
    Nu1, Nu2, Nu3 = clustersize
    @assert isodd(Nu1) && isodd(Nu2) && isodd(Nu3) "Each linear dimension of the non-perturbative cluster must be odd to guarantee an equal number of two particle states for all `qcom`s."

    L = nbands(swt)

    qs = [Vec3([i/Nu1, j/Nu2, k/Nu3]) for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]

    Es = zeros(L, Nu1, Nu2, Nu3)
    Vps = zeros(ComplexF64, 2L, 2L, Nu1, Nu2, Nu3)
    H_buf = zeros(ComplexF64, 2L, 2L)
    V_buf = zeros(ComplexF64, 2L, 2L)

    for iq in CartesianIndices(qs)
        q = qs[iq]
        dynamical_matrix!(H_buf, swt, q)
        E = bogoliubov!(V_buf, H_buf)
        Es[:, iq] = E[1:L]
        Vps[:, :, iq] = copy(V_buf)
    end

    if sys.mode == :SUN
        real_space_quartic_vertices = calculate_real_space_quartic_vertices_sun(sys)
        real_space_cubic_vertices   = calculate_real_space_cubic_vertices_sun(sys)
    else
        real_space_quartic_vertices = calculate_real_space_quartic_vertices_dipole(sys)
        real_space_cubic_vertices   = calculate_real_space_cubic_vertices_dipole(sys)
    end

    return NonPerturbativeTheory(swt, clustersize, qs, Es, Vps, real_space_quartic_vertices, real_space_cubic_vertices)

end

"""
    generate_two_particle_states(clustersize, q_index::CartesianIndex{3})

For a given `q_index::CartesianIndex{3}`, this block constructs a `Dict` that 
contains all possible two-particle states sharing the same center-of-mass momentum.
The dictionary keys consist of:
 - The Cartesian indices `q1` and `q2`, representing the momenta of the two particles.
 - The band indices of the two particles.
 
The dictionary values consist of a tuple with:
 1. The center-of-mass index of the state (an integer).
 2. The bosonic symmetry factor, which is either 1 (for distinguishable particles) or 1/√2 (for identical bosons in the same state).
 3. The final two components are the global indices of the two-particle state.
"""
function generate_two_particle_states(clustersize, L::Int, q_index::CartesianIndex{3})
    Nu1, Nu2, Nu3 = clustersize
    dict_states = Dict{Tuple{CartesianIndex{3}, CartesianIndex{3}, Int, Int}, Tuple{Int, Float64, Int, Int}}()
    cartes_indices = CartesianIndices((1:Nu1, 1:Nu2, 1:Nu3, 1:L))
    linear_indices = LinearIndices(cartes_indices)
    com_index = 0
    for k_index in CartesianIndices((1:Nu1, 1:Nu2, 1:Nu3))
        qmk_index = CartesianIndex(mod(q_index[1]-k_index[1], Nu1)+1, mod(q_index[2]-k_index[2], Nu2)+1, mod(q_index[3]-k_index[3], Nu3)+1)
        for band1 in 1:L
            ci = CartesianIndex(Tuple(k_index)..., band1)
            i  = linear_indices[ci]
            for band2 in 1:L
                cj = CartesianIndex(Tuple(qmk_index)..., band2)
                j  = linear_indices[cj]
                if i ≤ j
                    com_index += 1
                    dict_states[(k_index, qmk_index, band1, band2)] = (com_index, i == j ? 1/√2 : 1.0, i, j)
                end
            end
        end
    end
    return dict_states
end