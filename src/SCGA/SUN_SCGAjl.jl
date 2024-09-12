# SCGA for SU(N)   

function fourier_transform_interaction_matrix_SUN(sys::System; k, ϵ=0)
    @assert sys.mode in (:SUN) "This function is for SU(N) mode"
    @assert sys.latsize == (1, 1, 1) "System must have only a single cell"
    @assert length(unique(sys.Ns)) == 1 "Systems containing sites with different Hilbert space sizes not currently supported"
    Nf = sys.Ns[1]^2 - 1
    Na = Sunny.natoms(sys.crystal)
    J_k = zeros(ComplexF64,Nf , Na, Nf, Na)

    for i in 1:Na
        for coupling in sys.interactions_union[i].pair
            (; isculled, bond, bilin) = coupling
            isculled && break

            (; j, n) = bond
            J = exp(2π * im * dot(k, n+sys.crystal.positions[j]-sys.crystal.positions[i])) * Sunny.Mat3(bilin*I)
            J_k[:, i, :, j] += J / 2
            J_k[:, j, :, i] += J' / 2
        end
    end

    if !isnothing(sys.ewald)
        A = Sunny.precompute_dipole_ewald_at_wavevector(sys.crystal, (1,1,1), k) * sys.ewald.μ0_μB²
        A = reshape(A, Na, Na)
        for i in 1:Na, j in 1:Na
            J_k[:, i, :, j] += gs[i]' * A[i, j] * gs[j] / 2
        end
    end
    J_k = 2J_k # I'm not so sure why we need this
    for i in 1:Na
        onsite_coupling = sys.interactions_union[i].onsite
        (; c2, c4, c6) = onsite_coupling
        anisotropy = [c2[1]-c2[3]   c2[5]   0.5c2[2];
                        c2[5]   -c2[1]-c2[3]  0.5c2[4];
                        0.5c2[2]    0.5c2[4]    2c2[3]]
        J_k[:, i, :, i] += anisotropy 
    end

    J_k = reshape(J_k, 3*Na, 3*Na)
    @assert Sunny.diffnorm2(J_k, J_k') < 1e-15
    J_k = hermitianpart(J_k)
    return J_k 
end