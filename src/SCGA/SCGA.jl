abstract type AbstractSCGA end

struct SCGA <: AbstractSCGA
    sys            :: System
    measure        :: MeasureSpec
    regularization :: Float64
end

function SCGA(sys::System; measure::Union{Nothing, MeasureSpec}, regularization=1e-8)

    measure = @something measure empty_measurespec(sys)
    if length(eachsite(sys)) != prod(size(measure.observables)[2:5])
        error("Size mismatch. Check that measure is built using consistent system.")
    end
    return SCGA(sys, regularization)
end


function fourier_transform_interaction_matrix(sys::System; k, ϵ=0)
    @assert sys.mode in (:dipole, :dipole_large_S) "SU(N) mode not supported"
    @assert sys.latsize == (1, 1, 1) "System must have only a single cell"

    Na = natoms(sys.crystal)
    J_k = zeros(ComplexF64, 3, Na, 3, Na)

    for i in 1:Na
        for coupling in sys.interactions_union[i].pair
            (; isculled, bond, bilin) = coupling
            isculled && break

            (; j, n) = bond
            J = exp(2π * im * dot(k, n+sys.crystal.positions[j]-sys.crystal.positions[i])) * Mat3(bilin*I)
            J_k[:, i, :, j] += J / 2
            J_k[:, j, :, i] += J' / 2
        end
    end

    if !isnothing(sys.ewald)
        A = precompute_dipole_ewald_at_wavevector(sys.crystal, (1,1,1), k) * sys.ewald.μ0_μB²
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
    @assert diffnorm2(J_k, J_k') < 1e-15
    J_k = hermitianpart(J_k)
    return J_k 
end


function find_lagrange_multiplier(sys::System,T; ϵ=0, SumRule = "Classical")
    Nq = 8
    dq = 1/Nq;
    Na = natoms(sys.crystal)
    qarray = -0.5: dq : 0.5-dq
    q = [[qx, qy, qz] for qx in qarray, qy in qarray, qz in qarray]
    Jq_array = [fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0) for q_in ∈ q] 
    if SumRule == "Classical"
        S_sq = sys.κs[1]^2
    elseif SumRule == "Quantum"
        S_sq = sys.κs[1] * (sys.κs[1] + 1)
    else
        error("Unsupported SumRule: $SumRule. Expected 'Classical' or 'Quantum'.")
    end
    println("Sum rule: $SumRule")
    kb=0.0861733326
    eig_vals = zeros(3Na,length(Jq_array))
    for j in 1:length(Jq_array)
         eig_vals[:,j] .= eigvals(Jq_array[j])
     end
    function loss(λ)
        sum_term = sum(1 ./ (λ .+ (1/(kb*T)).*eig_vals ))
        return ((1/(Na*length(q))) * sum_term - S_sq)^2 
    end
    lower = maximum(-eig_vals./((1/(kb*T))))
    upper = Inf
    p = [lower+20]
    result = optimize(loss, lower, upper, p,NelderMead(), Optim.Options(time_limit = 60.0))
    min = Optim.minimizer(result)[1]
    println("Lagrange multiplier: $min")
    return min
end

function StructureFactorSCGA(sys,T,qs;SumRule = "Classical")
    Na = natoms(sys.crystal)
    kb=0.0861733326
    λ = find_lagrange_multiplier(sys,T;SumRule)
    Sout = zeros(ComplexF64, 3,3, length(qs))
    for (qi,q) ∈ enumerate(qs)
        J_mat = fourier_transform_interaction_matrix(sys; k=q, ϵ=0)
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

@inline function orientation_matrix(q)
    q /= norm(q) + 1e-12
    return (I(3) - q * q')
end

function intensities_SCGA(sys,T,qs;form_factor=nothing,SumRule = "Classical")
    S = StructureFactorSCGA(sys,T,qs;SumRule)
    int_out = zeros(Float64,length(qs))
    Na = natoms(sys.crystal)
    if isnothing(form_factor)
        for (qi,q) ∈ enumerate(qs) 
            q_abs = sys.crystal.recipvecs * q
            int = real(sum(orientation_matrix(q_abs).*S[:,:,qi]))/Na
            int_out[qi] = int
        end
    else
        for (qi,q) ∈ enumerate(qs) 
            q_abs = sys.crystal.recipvecs * q
            ff = compute_form_factor(form_factor, norm(q_abs))
            int = ff*real(sum(orientation_matrix(q_abs).*S[:,:,qi]))
            int_out[qi] = int
        end
    end
    return reshape(int_out,size(qs))
end

function fourier_transform_deprecated(sys::System; k, ϵ=0)
    @assert sys.mode in (:dipole, :dipole_large_S) "SU(N) mode not supported"
    @assert sys.latsize == (1, 1, 1) "System must have only a single cell"

    Na = natoms(sys.crystal)
    J_k = zeros(ComplexF64, 3, Na, 3, Na)

    for i in 1:Na
        for coupling in sys.interactions_union[i].pair
            (; isculled, bond, bilin) = coupling
            isculled && break
            (; j, n) = bond
            J = exp(2π * im * dot(k, n+sys.crystal.positions[j]-sys.crystal.positions[i])) * Mat3(bilin*I)
            J_k[:, i, :, j] += J / 2
            J_k[:, j, :, i] += J' / 2
        end
    end

    if !isnothing(sys.ewald)
        A = precompute_dipole_ewald_at_wavevector(sys.crystal, (1,1,1), k) * sys.ewald.μ0_μB²
        A = reshape(A, Na, Na)
        for i in 1:Na, j in 1:Na
            J_k[:, i, :, j] += gs[i]' * A[i, j] * gs[j] / 2
        end
    end

    J_k = reshape(J_k, 3*Na, 3*Na)
    @assert diffnorm2(J_k, J_k') < 1e-15
    J_k = hermitianpart(J_k)
    return 2J_k
end
#=
function intensities(scga::SCGA, qpts; formfactors=nothing, kT=0.0)
    qpts = convert(AbstractQPoints, qpts)
    data = zeros(eltype(scga.measure), length(qpts.qs))
    return intensities!(data, scga, qpts; formfactors, kT)
end

function intensities!(data, scga::SCGA, qpts; formfactors=nothing, kT=0.0)
    qpts = convert(AbstractQPoints, qpts)
    (; sys, measure, regularization) = swt

    ff_atoms = propagate_form_factors_to_atoms(formfactors, sys.crystal)

    for (iq, q) in enumerate(qpts.qs)
        q_reshaped = to_reshaped_rlu(sys, q)
        q_global = cryst.recipvecs * q

        for i in 1:Na
            Avec_pref[i] = compute_form_factor(ff_atoms[i], norm2(q_global))
        end
    end
    return InstantIntensities(cryst, qpts, data)
end
=#