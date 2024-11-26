abstract type AbstractSCGA end

struct SCGA <: AbstractSCGA
    sys            :: System
    measure        :: MeasureSpec
    regularization :: Float64
end

"""
    SCGA(sys::System; measure, regularization=1e-8)

Constructs an object to perform the self consistent gaussian approximation. Enables 
the use of `intensities_static`](@ref) with an SCGA type to calculate the static 
structure factor in the paramagnetic phase. If the temperature is below the ordering
temperature, the intensities will be negative.
"""

function SCGA(sys::System; measure::Union{Nothing, MeasureSpec}, regularization=1e-8)
    measure = @something measure empty_measurespec(sys)
    if length(eachsite(sys)) != prod(size(measure.observables)[2:5])
        error("Size mismatch. Check that measure is built using consistent system.")
    end
    return SCGA(sys,measure, regularization)
end


function fourier_transform_interaction_matrix(sys::System; k, ϵ=0)
    @assert sys.mode in (:dipole, :dipole_large_S) "SU(N) mode not supported"
    @assert sys.dims == (1, 1, 1) "System must have only a single cell"
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
        J_k[:, i, :, i] += 2*anisotropy
    end

    J_k = reshape(J_k, 3*Na, 3*Na)
    @assert diffnorm2(J_k, J_k') < 1e-15
    J_k = hermitianpart(J_k)
    return J_k 
end


function find_lagrange_multiplier(sys::System,kT; ϵ=0, SumRule = "Classical",starting_offset = 0.2, maxiters=1_000,tol = 1e-10,method = "Nelder Mead")
    Nq = 8
    dq = 1/Nq;
    Na = Sunny.natoms(sys.crystal)
    qarray = -0.5: dq : 0.5-dq
    q = [[qx, qy, qz] for qx in qarray, qy in qarray, qz in qarray]
    Jq_array = [Sunny.fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0) for q_in ∈ q] 
    if SumRule == "Classical"
        S_sqs = sys.κs.^2
    elseif SumRule == "Quantum"
        S_sqs = sys.κs .* (sys.κs .+ ones(Float64,Na))
    else
        error("Unsupported SumRule: $SumRule. Expected 'Classical' or 'Quantum'.")
    end
    println("Sum rule: $SumRule")
    eig_vals = zeros(3Na,length(Jq_array))
    for j in 1:length(Jq_array)
         eig_vals[:,j] .= eigvals(Jq_array[j])
    end
    class_list = sys.crystal.classes
    unique_sites = unique(class_list)
    
    if method == "Nelder Mead"
        function loss(λs_unique)
            losses = zeros(Float64,Na)
            class_assigment = Dict(unique_sites .=> λs_unique)
            λs = [class_assigment[class] for class in class_list]
            λ_mat = diagm(repeat(λs,inner=3))
            for (ind,λ) ∈ enumerate(λs)
                invmat = zeros(3Na,3Na,length(Jq_array))
                for j in 1:length(Jq_array)
                    invmat[:,:,j] .= real(inv(λ_mat .+ (1/(kT)).*Jq_array[j]))
                end
                sum_term = sum(invmat[3ind-2:3ind,3ind-2:3ind,:])
                losses[ind] = ((1/(length(q))) * sum_term - S_sqs[ind])^2 
            end
            return sum(losses)
        end
        lower = -minimum(eig_vals)/kT
        upper = Inf
        p = [lower+starting_offset] # some offset from the lower bound.
        options = Optim.Options(; iterations=maxiters, show_trace=false,g_tol=tol)
        result = optimize(loss, lower, upper, p,NelderMead(), options)
        min = Optim.minimizer(result)[1]
    elseif method == "Newton's Method" #Newton's method 
        function f(λ)
            sum_term = sum(1 ./ (λ .+ (1/(kT)).*eig_vals ))
            return (1/(Na*length(q))) * sum_term  
        end
        function J(λ)
            sum_term = sum((1 ./ (λ .+ (1/(kT)).*eig_vals )).^2)
            return -(1/(Na*length(q))) * sum_term  
        end
        lower = -minimum(eig_vals)/kT
        λn = starting_offset*0.1+ lower
        for n ∈ 1:maxiters
          λ = λn + (1/J(λn))*(S_sqs[1]-f(λn))
          if abs(λ-λn) < tol 
              println("Newton's method converged to within tolerance, $tol, after $n steps.")
              min=λ
              break
          else
          λn = λ
          end
        end
        
    else
        throw("Please provide valid method for the λ optimization.")
    end
    println("Lagrange multiplier: $min")
    return min
end

function intensities_static(scga::SCGA, qpts; formfactors=nothing, kT=0.0, SumRule = "Quantum",starting_offset = 0.2, maxiters=1_000,method="Nelder Mead",tol=1e-7)
    kT == 0.0 && error("kT must be non-zero")
    qpts = convert(AbstractQPoints, qpts)
    (; sys, measure, regularization) = scga
    Na = natoms(sys.crystal)
    Nq = length(qpts.qs)
    Nobs = num_observables(measure)
    λ = find_lagrange_multiplier(sys,kT;SumRule,method,tol,maxiters,starting_offset)
    intensity = zeros(eltype(measure),Nq)
    Ncorr = length(measure.corr_pairs)
    for (iq, q) in enumerate(qpts.qs)
        pref = zeros(ComplexF64, Nobs, Na)
        corrbuf = zeros(ComplexF64, Ncorr)
        intensitybuf = zeros(eltype(measure),3,3)
        q_reshaped = to_reshaped_rlu(sys, q)
        q_global = sys.crystal.recipvecs * q
        for i in 1:Na
            for μ in 1:3
                ff = get_swt_formfactor(measure, μ, i)
                pref[μ, i] = compute_form_factor(ff, norm2(q_global))
            end
        end
        J_mat = fourier_transform_interaction_matrix(sys; k=q_reshaped, ϵ=regularization)
        inverted_matrix = (inv(I(3*Na)*λ[1] + (1/kT)*J_mat)) # this is [(Iλ+J(q))^-1]^αβ_μν
        for i ∈ 1:Na 
            for j ∈ 1:Na 
                intensitybuf += pref[1,i]*conj(pref[1,j])*inverted_matrix[1+3(i-1):3+3(i-1),1+3(j-1):3+3(j-1)] 
                # TODO allow different form factor for each observable 
            end
        end
        map!(corrbuf, measure.corr_pairs) do (α, β)
            intensitybuf[α,β]
        end
        intensity[iq] = measure.combiner(q_global, corrbuf)
    end
    if extrema(intensity)[1] < -1e-2
        @warn "Warning: negative intensities! kT is probably below the ordering temperature."
        # TODO Throw an error. This is for diagnostic purposes.
    end
    return StaticIntensities(sys.crystal, qpts, reshape(intensity,size(qpts.qs)))
end
