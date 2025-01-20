struct MeanFieldValuesSUN
    Nii :: Matrix{ComplexF64}
    Njj :: Matrix{ComplexF64}
    Nij :: Matrix{ComplexF64}
    Δii :: Matrix{ComplexF64}
    Δjj :: Matrix{ComplexF64}
    Δij :: Matrix{ComplexF64}
end

struct MeanFieldValuesDipole
    Nii :: ComplexF64
    Njj :: ComplexF64
    Nij :: ComplexF64
    Δii :: ComplexF64
    Δjj :: ComplexF64
    Δij :: ComplexF64
end

struct PerturbativeTheory
    swt :: SpinWaveTheory
    real_space_quartic_vertices :: Vector{Union{RealSpaceQuarticVerticesSUN, RealSpaceQuarticVerticesDipole}}
    real_space_cubic_vertices   :: Vector{Union{RealSpaceCubicVerticesSUN, RealSpaceCubicVerticesDipole}}
    mean_field_values :: Vector{Union{MeanFieldValuesSUN, MeanFieldValuesDipole}}
end


function calculate_mean_field_values_dipole(swt::SpinWaveTheory; opts...)
    (; sys) = swt
    mean_field_values = MeanFieldValuesDipole[]
    L = nbands(swt)
    H = zeros(ComplexF64, 2L, 2L)
    V = zeros(ComplexF64, 2L, 2L)
    mean_field_buf = zeros(ComplexF64, 6)

    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            bond = coupling.bond
            di, dj = bond.i, bond.j

            ret = hcubature((0,0,0), (1,1,1); opts...) do k
                swt_hamiltonian_dipole!(H, swt, Vec3(k))
                bogoliubov!(V, H)

                mean_field_buf .= 0.0

                for band in 1:L
                    mean_field_buf[1] += V[di+L, band] * conj(V[di+L, band])
                    mean_field_buf[2] += V[dj+L, band] * conj(V[dj+L, band])
                    mean_field_buf[3] += V[di+L, band] * conj(V[dj+L, band]) * exp(-2π*im * dot(k, bond.n))
                    mean_field_buf[4] += V[di, band] * conj(V[di+L, band])
                    mean_field_buf[5] += V[dj, band] * conj(V[dj+L, band])
                    mean_field_buf[6] += V[di, band] * conj(V[dj+L, band]) * exp(-2π*im * dot(k, bond.n))
                end

                return SVector{6}(mean_field_buf)
            end

            push!(mean_field_values, MeanFieldValuesDipole(ret[1]...))
        end
    end

    return mean_field_values
end

function PerturbativeTheory(swt::SpinWaveTheory; opts...)
    (; sys) = swt

    if sys.mode == :SUN
        @error "SUN mode not implemented"
        real_space_quartic_vertices = calculate_real_space_quartic_vertices_sun(sys)
        real_space_cubic_vertices   = calculate_real_space_cubic_vertices_sun(sys)
    else
        real_space_quartic_vertices = calculate_real_space_quartic_vertices_dipole(sys)
        real_space_cubic_vertices   = calculate_real_space_cubic_vertices_dipole(sys)
        mean_field_values = calculate_mean_field_values_dipole(swt; opts...)
    end

    return PerturbativeTheory(swt, real_space_quartic_vertices, real_space_cubic_vertices, mean_field_values)
end