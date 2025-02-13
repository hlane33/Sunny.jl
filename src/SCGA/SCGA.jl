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

"""
    fourier_transform_interaction_matrix(sys::System; k, ϵ=0)

Fourier transforms the interaaction matrix for a System and evaluatues it at q=k. A 
small regularization, ϵ, can be added. This function is adapted from 
`luttinger_tisza_exchange`](@ref) to include the correct prefactors and local anisotropies.

"""

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
            J_k[:, i, :, j] += sys.gs[i]' * A[i, j] * sys.gs[j] / 2
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

"""
    find_lagrange_multiplier(sys::System,kT; ϵ=0, SumRule = "Classical",starting_offset = 0.2, maxiters=1_000,tol = 1e-10,method = "Nelder Mead", Nq = 20)

Computes the Lagrange multiplier for the standard SCGA approach with a common Lagrange multiplier 
for all sublattices. Two optimization methods can be chosen, "Nelder Mead" which takes advantage 
of the implementation in Optim.jl and "Newton's Method" which is a fast converging root finding
algorithm. SumRule specifices the sum rule that the Lagrange multipliers are chosen to satisfy, 
either the Classical or Quantum sum rules.

"""

function find_lagrange_multiplier(sys::System,kT; ϵ=0, SumRule = "Classical",starting_offset = 0.2, maxiters=1_000,tol = 1e-10,method = "Nelder Mead",Nq = 20)
    dq = 1/Nq;
    Na = natoms(sys.crystal)
    bond_counter = zeros(Float64,3)
    for i in 1:Na
        for coupling in sys.interactions_union[i].pair
            (; isculled, bond, bilin) = coupling
            isculled && break
            (; j, n) = bond
            bond_counter += abs.(n)
        end
    end
    qarray = -0.5: dq : 0.5-dq
    qarrays = []
    for i ∈ 1:3
        if bond_counter[i] == 0
            push!(qarrays,[0])
        else
            push!(qarrays,qarray)
        end
    end
    q = [[qx, qy, qz] for qx in qarrays[1], qy in qarrays[2], qz in qarrays[3]]
    Jq_array = [fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0) for q_in ∈ q] 
    av_S = sum(sys.κs)/Na
    if SumRule == "Classical"
        S_sq = av_S^2
    elseif SumRule == "Quantum"
        S_sq = av_S * (av_S + 1)
    else
        error("Unsupported SumRule: $SumRule. Expected 'Classical' or 'Quantum'.")
    end
    println("Sum rule: $SumRule")
    eig_vals = zeros(3Na,length(Jq_array))
    for j in 1:length(Jq_array)
         eig_vals[:,j] .= eigvals(Jq_array[j])
    end
    if method == "Nelder Mead"
        function loss(λ)
            sum_term = sum(1 ./ (λ .+ (1/(kT)).*eig_vals ))
            return ((1/(Na*length(q))) * sum_term - S_sq)^2 
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
          λ = λn + (1/J(λn))*(S_sq-f(λn))
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

"""
     intensities_static_single(scga::SCGA, qpts; kT=0.0, SumRule = "Quantum",starting_offset = 0.2, maxiters=1_000,method="Nelder Mead",tol=1e-7)

Computes the static structure factor in the standard SCGA approach, with a single
Lagrange multiplier, over the qpts specified. 

"""

function intensities_static_single(scga::SCGA, qpts; kT=0.0, SumRule = "Quantum",starting_offset = 0.2, maxiters=1_000,method="Nelder Mead",tol=1e-7,Nq=50)
    kT == 0.0 && error("kT must be non-zero")
    qpts = convert(AbstractQPoints, qpts)
    (; sys, measure, regularization) = scga
    Na = natoms(sys.crystal)
    Nobs = num_observables(measure)
    λ = find_lagrange_multiplier(sys,kT;SumRule,method,tol,maxiters,starting_offset,Nq)
    intensity = zeros(eltype(measure),length(qpts.qs))
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
        # inverted_matrix = diagm(1 ./ (λ[1] .+ (1/(kT)).*eigvals(J_mat)))
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

"""
     intensities_static_sublattice(scga::SCGA, qpts; kT=0.0, SumRule = "Quantum", maxiters=500,tol=1e-10,λs_init)

Computes the static structure factor in the sublattice resolved SCGA method, over the 
qpts specified.
"""

function intensities_static_sublattice(scga::SCGA, qpts; kT=0.0, SumRule = "Quantum", maxiters=500,tol=1e-10,λs_init,Nq = 20)
    kT == 0.0 && error("kT must be non-zero")
    println("Sublattice resolved SCGA currently only supports Conjugate Gradient optimization.")
    qpts = convert(AbstractQPoints, qpts)
    (; sys, measure, regularization) = scga
    Na = natoms(sys.crystal)
    Nobs = num_observables(measure)
    λs = find_lagrange_multiplier_opt_sublattice(sys,λs_init,kT;SumRule,maxiters,tol,Nq)
    intensity = zeros(eltype(measure),length(qpts.qs))
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
        Λ =  diagm(repeat(λs, inner=3))
        inverted_matrix = (inv((1/kT)*Λ + (1/kT)*J_mat)) # this is [(Iλ+J(q))^-1]^αβ_μν
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
    println("Optimized Lagrange multipliers: $λs")
    return StaticIntensities(sys.crystal, qpts, reshape(intensity,size(qpts.qs)))
end

"""
     intensities_static(scga::SCGA, qpts; kT=0.0, SumRule = "Quantum", maxiters=500,tol=1e-10,method="Newton's Method",λs_init=nothing,sublattice_resolved = false)

Computes the static structure factor. The sublattice resolved method can be chosen by selecting
sublattice_resolved = true. 
"""


function intensities_static(scga::SCGA, qpts; kT=0.0, SumRule = "Quantum", maxiters=500,tol=1e-10,method="Newton's Method",λs_init=nothing,sublattice_resolved = false,Nq = 20)
    if sublattice_resolved == true
        return intensities_static_sublattice(scga::SCGA, qpts; kT, SumRule , maxiters,tol,λs_init,Nq)
    else
        return intensities_static_single(scga::SCGA, qpts; kT, SumRule ,starting_offset = 0.2, maxiters,method,tol,Nq)
    end
end

"""
    find_lagrange_multiplier_opt_sublattice(sys,λs,kT;SumRule = "Quantum",maxiters=500,tol=1e-10)

Computes the Lagrange multiplier for the sublattice resolved SCGA method. The optimization is performed
using the Conjugate Gradient method as implemented in Optim.jl.

"""

function find_lagrange_multiplier_opt_sublattice(sys,λs,kT;SumRule = "Quantum",maxiters=500,tol=1e-10,Nq = 20)
    if SumRule == "Classical"
        S_sq = vec(sys.κs.^2)
    elseif SumRule == "Quantum"
        S_sq =vec( sys.κs .* (sys.κs .+ 1))
    else
        error("Unsupported SumRule: $SumRule. Expected 'Classical' or 'Quantum'.")
    end
    Na = natoms(sys.crystal)
    dq = 1/Nq;
    bond_counter = zeros(Float64,3)
    for i in 1:Na
        for coupling in sys.interactions_union[i].pair
            (; isculled, bond, bilin) = coupling
            isculled && break
            (; j, n) = bond
            bond_counter += abs.(n)
        end
    end
    qarray = -0.5: dq : 0.5-dq
    qarrays = []
    for i ∈ 1:3
        if bond_counter[i] == 0
            push!(qarrays,[0])
        else
            push!(qarrays,qarray)
        end
    end
    q = [[qx, qy, qz] for qx in qarrays[1], qy in qarray[2], qz in qarray[3]]
    N = length(q)
    function f(λs)
        Λ =  diagm(repeat(λs, inner=3))
        A_array = [(1/kT)*Sunny.fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0) .+  (1/kT)*Λ for q_in ∈ q] 
        eig_vals = zeros(3Na,length(A_array))
        Us = zeros(ComplexF64,3Na,3Na,length(A_array))
        for j in 1:length(A_array)
            T = eigen(A_array[j])
            eig_vals[:,j] .= T.values
            Us[:,:,j] .= T.vectors
        end
        if minimum(eig_vals) < 0 
            F =  -Inf
        else
            F =  0.5*kT*sum(log.(eig_vals))
        end
        G = F - 0.5*N*sum(λs.*S_sq)
        return -G
    end
    function fg!(fbuffer,gbuffer,λs)
        Λ =  diagm(repeat(λs, inner=3))
        A_array = [(1/kT)*fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0) .+  (1/kT)*Λ for q_in ∈ q] 
        eig_vals = zeros(3Na,length(A_array))
        Us = zeros(ComplexF64,3Na,3Na,length(A_array))
        for j in 1:length(A_array)
            T = eigen(A_array[j])
            eig_vals[:,j] .= T.values
            Us[:,:,j] .= T.vectors
        end
        if gbuffer !== nothing
            gradF = zeros(ComplexF64,Na)
            for i ∈ 1:Na
                gradλ =diagm(zeros(ComplexF64,3Na))
                gradλ[3i-2:3i,3i-2:3i] =diagm([1,1,1])
                gradF[i] =0.5sum([tr(diagm(1 ./eig_vals[:,j]) * Us[:,:,j]'*gradλ*Us[:,:,j]) for j ∈ 1:length(A_array)]) 
            end
            gradG = gradF -0.5*N*S_sq
            gbuffer .= -real(gradG)
        end
        if fbuffer !== nothing
            if minimum(eig_vals) < 0 
                F =  -Inf
            else
                F =  0.5*kT*sum(log.(eig_vals))
            end
            G = F - 0.5*N*sum(λs.*S_sq)
            fbuffer= -G
        end
    end
    if λs == nothing
        println("No user provided initial guess for the Lagrange multipliers. Determining a sensible starting point from the interaction matrix.")
        A_array = [fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0)  for q_in ∈ q] 
        eig_vals = zeros(3Na,length(A_array))
        Us = zeros(ComplexF64,3Na,3Na,length(A_array))
        for j in 1:length(A_array)
            T = eigen(A_array[j])
            eig_vals[:,j] .= T.values
            Us[:,:,j] .= T.vectors
        end
        extreme_eigvals = extrema(eig_vals)
        if extreme_eigvals[1] < 0
            pos_shift = -extreme_eigvals[1]
        else
            pos_shift = 0
        end
        λ_init = pos_shift + (extreme_eigvals[2]-extreme_eigvals[1])/2
        λs = λ_init*ones(Float64,Na)
    else
        println("Using user provided initial starting point for the optimization of the Lagrange multipliers.")
        if f(λs) > 1e7
            println("Matrix is not positive definite. Shifting Lagrange multipliers to find a better starting point.")
            A_array = [fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0)  for q_in ∈ q] 
            eig_vals = zeros(3Na,length(A_array))
            Us = zeros(ComplexF64,3Na,3Na,length(A_array))
            for j in 1:length(A_array)
                T = eigen(A_array[j])
                eig_vals[:,j] .= T.values
                Us[:,:,j] .= T.vectors
            end
            extreme_eigvals = extrema(eig_vals)
            shift = (extreme_eigvals[2] - extreme_eigvals[1])/2
            while f(λs) > 1e7
                λs += shift*ones(Float64,Na)
            end
        end
    end
    println(λs)
    upper = Inf
    lower = -Inf
    options = Optim.Options(; iterations=maxiters, show_trace=true,g_tol=tol)
    result = optimize(Optim.only_fg!(fg!), λs, ConjugateGradient(),options)
    min = Optim.minimizer(result)
    return real.(min)
end

function find_lagrange_multiplier_opt_WORKING(sys,λs,kT;SumRule = "Quantum",method = "ConjugateGradient",maxiters=500,tol=1e-10)
    if SumRule == "Classical"
        S_sq = vec(sys.κs.^2)
    elseif SumRule == "Quantum"
        S_sq =vec( sys.κs .* (sys.κs .+ 1))
    else
        error("Unsupported SumRule: $SumRule. Expected 'Classical' or 'Quantum'.")
    end
    Na = natoms(sys.crystal)
    Nq = 8
    dq = 1/Nq;
    qarray = -0.5: dq : 0.5-dq
    q = [[qx, qy, qz] for qx in qarray, qy in qarray, qz in qarray]
    N = length(q)
    if method == "ConjugateGradient"
        function f(λs)
            Λ =  diagm(repeat(λs, inner=3))
            A_array = [(1/kT)*fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0) .+  (1/kT)*Λ for q_in ∈ q] 
            eig_vals = zeros(3Na,length(A_array))
            Us = zeros(ComplexF64,3Na,3Na,length(A_array))
            for j in 1:length(A_array)
                T = eigen(A_array[j])
                eig_vals[:,j] .= T.values
                Us[:,:,j] .= T.vectors
            end
            if minimum(eig_vals) < 0 
                F =  -Inf
            else
                F =  0.5*kT*sum(log.(eig_vals))
            end
            G = F - 0.5*N*sum(λs.*S_sq)
            return -G
        end
        function g!(storage,λs)
        # gradient 
            Λ =  diagm(repeat(λs, inner=3))
            A_array = [(1/kT)*fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0) .+  (1/kT)*Λ for q_in ∈ q] 
            eig_vals = zeros(3Na,length(A_array))
            Us = zeros(ComplexF64,3Na,3Na,length(A_array))
            for j in 1:length(A_array)
                T = eigen(A_array[j])
                eig_vals[:,j] .= T.values
                Us[:,:,j] .= T.vectors
            end
            gradF = zeros(ComplexF64,Na)
            for i ∈ 1:Na
                gradλ =diagm(zeros(ComplexF64,3Na))
                gradλ[3i-2:3i,3i-2:3i] =diagm([1,1,1])
                # gradF[i] =0.5kT*sum([tr(inv(A) * gradλ) for A ∈ A_array])
                gradF[i] =0.5sum([tr(diagm(1 ./eig_vals[:,j]) * Us[:,:,j]'*gradλ*Us[:,:,j]) for j ∈ 1:length(A_array)]) 
                # replace this -> we want to use eigenvalues determined above and invariance of trace under change of basis Tr(A (dΛ/dλ) )= Tr(U'A⁻¹UU'(dΛ/dλ)U) = Tr(D⁻¹U'(dΛ/dλ)U)  
            end
            gradG = gradF -0.5*N*S_sq
            storage .= -real(gradG)
        end
        upper = Inf
        lower = -Inf
        options = Optim.Options(; iterations=maxiters, show_trace=true,g_tol=1e-8)
        result = optimize(f, g!,λs , ConjugateGradient(),options)
        min = Optim.minimizer(result)
    elseif method == "Newton's Method"
        function precalculate_matrices(λs)
            Λ =  diagm(repeat(λs, inner=3))
            A_array = [(1/kT)*fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0) +  (1/kT)*Λ for q_in ∈ q] 
            eig_vals = zeros(Float64,3Na,length(A_array))
            Us = zeros(ComplexF64,3Na,3Na,length(A_array))
            for j in 1:length(A_array)
                T = eigen(A_array[j])
                eig_vals[:,j] .= T.values
                Us[:,:,j] .= T.vectors
            end
            return eig_vals, Us, A_array
        end
        function G(λs,eig_vals)
            if minimum(eig_vals) < 0 
                F =  -10^9
            else
                F =  0.5*kT*sum(log.(eig_vals))
            end
            Gout = F - 0.5*N*sum(λs.*S_sq)
            return -real.(Gout)
        end
        function Gp(eig_vals, Us) 
            gradF = zeros(ComplexF64,Na)
            for i ∈ 1:Na
                gradλ =diagm(zeros(ComplexF64,3Na))
                gradλ[3i-2:3i,3i-2:3i] =diagm([1,1,1])
                gradF[i] =0.5sum([tr(diagm(1 ./eig_vals[:,j]) * Us[:,:,j]'*gradλ*Us[:,:,j]) for j ∈ 1:size(eig_vals,2)]) 
            end
            gradG = gradF -0.5*N*S_sq
            return -real.(gradG)
        end
        function Gpp(eig_vals, Us, A_array)
            Gppmat = zeros(ComplexF64,Na,Na)
            for i ∈ 1:Na
                for j ∈ 1:Na
                    P1 =diagm(zeros(ComplexF64,3Na))
                    P1[3i-2:3i,3i-2:3i] =diagm([1,1,1])
                    P2 =diagm(zeros(ComplexF64,3Na))
                    P2[3j-2:3j,3j-2:3j] =diagm([1,1,1])
                    # Gppmat[i,j] = -0.5*sum([tr(diagm(1 ./eig_vals[:,q]) * Us[:,:,q]' * P2 * Us[:,:,q] * diagm(1 ./eig_vals[:,q]) *  Us[:,:,q]' * P1 * Us[:,:,q] ) for q ∈ 1:size(eig_vals,2)])
                    Gppmat[i,j] = -0.5*sum([tr(inv(A_array[qi])  * P2  * inv(A_array[qi]) * P1  ) for qi ∈ 1:size(eig_vals,2)])
                end
            end
            return -real.(Gppmat)
        end
        λn = λs
        lout = []
        for n ∈ 1:maxiters
            eig_vals, Us, A_array =  precalculate_matrices(λn) 
            λ = λn -inv(Gpp(eig_vals, Us,A_array)) * Gp(eig_vals,Us) 
            if sum(abs.(real.(λ-λn))) < tol 
                println("Newton's method converged to within tolerance, $tol, after $n steps.")
                min=λ
                break
            else
            λn = λ
            end
        end
    else
        throw("Please provide a valid method.")
    end
    return real.(min)
end

function free_energy_and_gradient(sys,λs,kT;SumRule = "Quantum")
    if SumRule == "Classical"
        S_sq = vec(sys.κs.^2)
    elseif SumRule == "Quantum"
        S_sq =vec( sys.κs .* (sys.κs .+ 1))
    else
        error("Unsupported SumRule: $SumRule. Expected 'Classical' or 'Quantum'.")
    end
    Na = natoms(sys.crystal)
    Nq = 8
    dq = 1/Nq;
    qarray = -0.5: dq : 0.5-dq
    N = length(qarray)
    q = [[qx, qy, qz] for qx in qarray, qy in qarray, qz in qarray]
    Λ =  diagm(repeat(λs, inner=3))
    A_array = [kT*fourier_transform_interaction_matrix(sys; k=q_in, ϵ=0) .+  kT*Λ for q_in ∈ q] 
    eig_vals = zeros(3Na,length(A_array))
    for j in 1:length(A_array)
         eig_vals[:,j] .= eigvals(A_array[j])
    end
    if minimum(eig_vals) < 0 
        F =  -Inf
    else
        F =  0.5*kT*sum(log.(eig_vals))
    end
    G = F - 0.5*N*sum(λs.*S_sq)
    # gradient 
    gradF = zeros(ComplexF64,Na)
    for i ∈ 1:Na
        gradλ =diagm(zeros(ComplexF64,3Na))
        gradλ[3i-2:3i,3i-2:3i] =diagm([1,1,1])
        # gradF[i] =0.5kT*sum([tr(inv(A) * gradλ) for A ∈ A_array])
        gradF[i] =0.5sum([tr(diagm(1 ./eig_vals[:,j]) * Us[:,:,j]'*gradλ*Us[:,:,j]) for j ∈ 1:length(A_array)]) 
        # replace this -> we want to use eigenvalues determined above and invariance of trace under change of basis Tr(A (dΛ/dλ) )= Tr(U'A⁻¹UU'(dΛ/dλ)U) = Tr(D⁻¹U'(dΛ/dλ)U)  
    end
    gradG = gradF -0.5*N*S_sq
    return G, gradG
end

