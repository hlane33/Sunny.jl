function fourier_transform_exchange(sys::System; k, ϵ=0)
    @assert sys.mode in (:dipole, :dipole_large_S) "SU(N) mode not supported"
    @assert sys.latsize == (1, 1, 1) "System must have only a single cell"

    Na = Sunny.natoms(sys.crystal)
    J_k = zeros(ComplexF64, 3, Na, 3, Na)

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

    J_k = reshape(J_k, 3*Na, 3*Na)
    @assert Sunny.diffnorm2(J_k, J_k') < 1e-15
    J_k = hermitianpart(J_k)
    return 2J_k
end

function fourier_transform_exchange_anisotropy(sys::System; k, ϵ=0)
    @assert sys.mode in (:dipole, :dipole_large_S) "SU(N) mode not supported"
    @assert sys.latsize == (1, 1, 1) "System must have only a single cell"

    Na = Sunny.natoms(sys.crystal)
    J_k = zeros(ComplexF64, 3, Na, 3, Na)

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
    J_k = 2J_k # Gao has this, I'm not so sure
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
    J_k = hermitianpart(J_k) # not sure about the 2 - Gao has this
    return J_k 
end


function find_lagrange_multiplier(sys::System,T; ϵ=0)
    Nq = 8
    dq = 1/Nq;
    Na = Sunny.natoms(sys.crystal)
    qarray = -0.5: dq : 0.5-dq
    q = [[qx, qy, qz] for qx in qarray, qy in qarray, qz in qarray]
    Jq_array = [fourier_transform_exchange(sys; k=q_in, ϵ=0) for q_in ∈ q] 
    S_sq =  sys.κs[1]^2 # will add option for both S^2 and S(S+1)
    # S_sq = sys.κs[1]*(sys.κs[1]+1)
    kb=0.0861733326
    function loss2(λ)
        sum_term = sum(1 ./ (λ .+ (1/(kb*T)).*eig_vals ))
        return ((1/(Na*length(q))) * sum_term - S_sq)^2 
    end
    lower = maximum(-eig_vals./((1/(kb*T))))
    upper = Inf
    p = [lower+20rand()]
    result = optimize(loss2, lower, upper, p,NelderMead(), Optim.Options(time_limit = 10.0))
    min = Optim.minimizer(result)
    println(min)
    return min
end

function Smn(sys,T,qs)
    Na = Sunny.natoms(sys.crystal)
    kb=0.0861733326
    λ = find_lagrange_multiplier(sys,T)
    Sout = zeros(ComplexF64, 3,3, length(qs))
    for q ∈ qs
        J_mat = fourier_transform_exchange(sys; k=q, ϵ=0)
        inverted_matrix = (inv(I(3*Na)*λ[1] + (1/(kb*T))*J_mat))
        output_Sab = zeros(ComplexF64, 3,3)
        index_list = 1:3:3Na 
        for i ∈ index_list
            for j ∈ index_list
                output_Sab += inverted_matrix[i:i+2,j:j+2] 
            end
        end
        Sout[:,:,qi] = output_Sab
    end
    return Sout
end

@inline function polarization_matrix(q)
    q /= norm(q) + 1e-12
    return (I(3) - q * q')
end

function neutron_intensity(sys,T,qs,form_factor)
    S = Smn(sys,T,qs)
    int_out = zeros(Float64,length(qs))
    Na = Sunny.natoms(sys.crystal)
    for (qi,q) ∈ enumerate(qs) 
        q_abs = sys.crystal.recipvecs * q
        # ff = Sunny.compute_form_factor(form_factor, norm(q_abs))
        # int = ff*real(sum(polarization_matrix(q_abs).*S[qi]))
        int = real(sum(polarization_matrix(q_abs).*S[qi]))/Na
        int_out[qi] = int
    end
     return reshape(int_out,size(qs))
end
