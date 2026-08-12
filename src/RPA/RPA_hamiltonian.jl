# function mean_field_hamiltonians(rpa::RPA,q_reshaped::Vec3)
#     (; sys) = rpa
#     # (; extfield) = sys

#     L = length(sys.mean_fields)
#     H = zeros(ComplexF64, L, L)
#     Hu = zeros(ComplexF64,L,L)
#     Hd = zeros(ComplexF64,L,L)
#     Hv = zeros(ComplexF64,L,L)


#     for (i, int) in enumerate(sys.interactions_union)

#         # Pair interactions
#         for coupling in int.pair
#             (; isculled, bond) = coupling
#             isculled && break

#             @assert i == bond.i
#             j = bond.j

#             phase = exp(2π*im * dot(q_reshaped, bond.n)) # Phase associated with periodic wrapping

#             # Bilinear exchange
#             if !iszero(coupling.hopping)
#                 t = coupling.hopping 
#                 H[i, j] += -t * phase
#                 H[j, i] += conj(-t) * conj(phase)
#             end
#         end
#         for hub in int.hubbard
#             Hu[i,i] += hub*sys.mean_fields[i][2]
#             Hd[i,i] += hub*sys.mean_fields[i][1]
#         end
#         for pot in int.onsite 
#             Hv[i,i] += pot
#         end
#     end
#     Hup = H + Hu +Hv
#     Hdown = H + Hd +Hv
#     # H must be hermitian up to round-off errors
#     @assert diffnorm2(Hup, Hup') < 1e-12
#     @assert diffnorm2(Hdown, Hdown') < 1e-12

#     # Make H exactly hermitian
#     hermitianpart!(Hup)
#     hermitianpart!(Hdown)

#     # Add small constant shift for positive-definiteness
#     for i in 1:L
#         Hup[i, i] += rpa.regularization
#         Hdown[i, i] += rpa.regularization
#     end
#     return Hup, Hdown
# end

function lor(x, Γ)
    return (Γ/2) / (π*(x^2+(Γ/2)^2))
end

function get_bound_from_tol(tol,Γ)
    return sqrt(2Γ -π*(Γ^2)*tol)/(2*sqrt(tol*π))
end

function density_of_states(energies,rpa::RPA,dq::Float64;Γ=0.1)
    # careful here about the q range in SCGA we use [0,1) because supercell is (1,1,1), but here we need [0,1/N)
    # 0 < dq < 1 || error("Select q-space resolution 0 < dq < 1/N")
    (; sys) = rpa
    qs = make_q_grid(sys, dq)  
    Nqs = length(qs)
    ρs = zeros(Float64,length(energies))
    for (Ei,E) in enumerate(energies)
        for qi in 1:Nqs
            q= qs[qi]
            Hup, Hdown = mean_field_hamiltonians(rpa,q)
            Eus, Uus = eigen(Hup)    
            Eds, Uds = eigen(Hdown) 
            for m in 1:length(Eus)
                ρs[Ei] += lor(E-Eus[m],Γ)
            end
            for m in 1:length(Eds)
                ρs[Ei] += lor(E-Eds[m],Γ)
            end  
        end
    end
    ρs /= Nqs
    return ρs
end

function fermi(E,μ,kT)
    return 1/(exp((E-μ)/kT)+1)
end

function find_chemical_potential(rpa::RPA,dq::Float64,ne,nEs;Γ=0.1, tol=1e-2,mu_bounds =nothing,rel_tol,kT)
    # careful here about the q range in SCGA we use [0,1) because supercell is (1,1,1), but here we need [0,1/N) 
    (; sys) = rpa
     ne ≤ 2*length(sys.mean_fields) || @warn "Electron density is more than 2. Consider a different value of ne "
    qs = make_q_grid(sys, dq)  
    Nqs = length(qs)
    ρs = zeros(Float64,nEs)
    nmodes = length(sys.mean_fields)
    Evals_up = zeros(Float64,nmodes,Nqs)
    Evals_down = zeros(Float64,nmodes,Nqs)
    (emin,emax) = extrema(hcat(Evals_up,Evals_down))
    E_ϵ =  get_bound_from_tol(tol,Γ) 
    for qi in 1:Nqs
        q= qs[qi]
        Hup, Hdown = mean_field_hamiltonians(rpa,q)
        Eus, _ = eigen(Hup)    
        Eds, _ = eigen(Hdown) 
        Evals_up[:,qi] .= Eus
        Evals_down[:,qi] .= Eds
    end 
    Elims = extrema(hcat(Evals_up,Evals_down)) .+ (-E_ϵ,E_ϵ)
    energies = range(Elims[1],Elims[2];length=nEs)
    dE = step(energies)
    ρs = zeros(Float64,length(energies))
    for (Ei,E) in enumerate(energies)
        for qi in 1:Nqs
            q= qs[qi]
            for m in 1:nmodes
                ρs[Ei] += lor(E-Evals_up[m,qi],Γ)
            end
            for m in 1:nmodes
                ρs[Ei] += lor(E-Evals_down[m,qi],Γ)
            end  
        end
    end
    ρs /= Nqs
    function density(μ)
        return (sum(ρs .* fermi.(energies, μ, kT)) * dE)
    end
    loss(μ) = (density(μ) - ne)^2

    if isnothing(mu_bounds)
        mu_bounds = Elims
    end

    res = Optim.optimize(loss, mu_bounds[1], mu_bounds[2], Optim.Brent();rel_tol)
        μ_solution = Optim.minimizer(res)
    return μ_solution
end

function self_consistency(rpa::RPA,dq::Float64,ne,nEs;Γ=0.1, tol=1e-2,mu_bounds =nothing,rel_tol,kT,tol_sc,α,niters=200)
    (; sys) = rpa
    qs = Sunny.make_q_grid(sys, dq)  
    Nq = length(qs)
    Na = length(sys.mean_fields)
    for ii in 1:niters
        μ =  Sunny.find_chemical_potential(rpa,dq,ne,nEs;Γ, tol,mu_bounds ,rel_tol,kT)
        mean_fields_new = fill([0.0, 0.0], size(sys.mean_fields)...)
        for qi in 1:Nq
            q = qs[qi]
            Hup, Hdown = Sunny.mean_field_hamiltonians(rpa,q)
            Eus, Uus = eigen(Hup)    
            Eds, Uds = eigen(Hdown)    
            for m ∈ 1:Na
                for i ∈ 1:Na
                    mean_fields_new[m][2] +=real(abs2(Uds[m,i])*fermi(Eds[i],μ,kT))
                    mean_fields_new[m][1] += real(abs2(Uus[m,i])*fermi(Eus[i],μ,kT))
                end
            end
        end
        mean_fields_new /= Nq
        mean_fields_new /= Nq
        diff = norm(mean_fields_new - sys.mean_fields) /norm(mean_fields_new)
        
        if diff < tol_sc
            println("converged after $ii iterations, diff: $diff") 
            println("mean fields: $mean_fields_new")
            return mean_fields_new #; break
        end
        sys.mean_fields[1] = (1-α)*mean_fields_new[1]+α*mean_fields_new[2]
        sys.mean_fields[2] = (1-α)*mean_fields_new[2]+α*mean_fields_new[1]
        println("Not converged: iteration $ii, diff: $diff")
    end
end

function optimize_mean_fields!(sys::ElectronicSystem,dq,ne,nEs;Γ=0.1,tol=1e-2,mu_bounds =nothing, rel_tol,tol_sc,kT, α,niters=200)
    dummy_measure = ssf_perp(sys) #stopgap
    dummy_rpa = RPA(sys, regularization=1e-8;measure = dummy_measure)
    mean_fields_new = self_consistency(dummy_rpa,dq,ne,nEs;Γ, tol,mu_bounds,rel_tol,kT,tol_sc,α,niters)
    for m in 1:length(mean_fields_new)
        sys.mean_fields[m] = mean_fields_new[m]
    end
end

###########################################
###########################################
function mean_field_hamiltonians(mft::MeanFieldTheory,q_reshaped::Vec3)
    (; sys, mean_fields) = mft
    L = length(sys.mean_fields)
    H = zeros(ComplexF64, L, L)
    Hu = zeros(ComplexF64,L,L)
    Hd = zeros(ComplexF64,L,L)
    Hv = zeros(ComplexF64,L,L)

    for (i, int) in enumerate(sys.interactions_union)

        # Pair interactions
        for coupling in int.pair
            (; isculled, bond) = coupling
            isculled && break

            @assert i == bond.i
            j = bond.j

            phase = exp(2π*im * dot(q_reshaped, bond.n)) # Phase associated with periodic wrapping

            # Bilinear exchange
            if !iszero(coupling.hopping)
                t = coupling.hopping 
                H[i, j] += -t * phase
                H[j, i] += conj(-t) * conj(phase)
            end
        end
        for hub in int.hubbard
            Hu[i,i] += hub*mean_fields[i][2]
            Hd[i,i] += hub*mean_fields[i][1]
        end
        for pot in int.onsite 
            Hv[i,i] += pot
        end
    end
    Hup = H + Hu +Hv
    Hdown = H + Hd +Hv
    # H must be hermitian up to round-off errors
    @assert diffnorm2(Hup, Hup') < 1e-12
    @assert diffnorm2(Hdown, Hdown') < 1e-12

    # Make H exactly hermitian
    hermitianpart!(Hup)
    hermitianpart!(Hdown)

    # Add small constant shift for positive-definiteness
    for i in 1:L
        Hup[i, i] += mft.regularization
        Hdown[i, i] += mft.regularization
    end
    return Hup, Hdown
end


function find_chemical_potential(mft::MeanFieldTheory,dq::Float64,ne,nEs;Γ=0.1, tol=1e-2,mu_bounds =nothing,rel_tol,kT)
    # careful here about the q range in SCGA we use [0,1) because supercell is (1,1,1), but here we need [0,1/N) 
    (; sys, mean_fields) = mft
    ne ≤ 2*length(mean_fields) || @warn "Electron density is more than 2. Consider a different value of ne "
    qs = make_q_grid(sys, dq)  
    Nqs = length(qs)
    ρs = zeros(Float64,nEs)
    nmodes = length(mean_fields)
    Evals_up = zeros(Float64,nmodes,Nqs)
    Evals_down = zeros(Float64,nmodes,Nqs)
    E_ϵ =  get_bound_from_tol(tol,Γ) 
    for qi in 1:Nqs
        q= qs[qi]
        Hup, Hdown = mean_field_hamiltonians(mft,q)
        Eus, _ = eigen(Hup)    
        Eds, _ = eigen(Hdown) 
        Evals_up[:,qi] .= Eus
        Evals_down[:,qi] .= Eds
    end 
    Elims = extrema(hcat(Evals_up,Evals_down)) .+ (-E_ϵ,E_ϵ)
    energies = range(Elims[1],Elims[2];length=nEs)
    dE = step(energies)
    ρs = zeros(Float64,length(energies))
    for (Ei,E) in enumerate(energies)
        for qi in 1:Nqs
            q= qs[qi]
            for m in 1:nmodes
                ρs[Ei] += lor(E-Evals_up[m,qi],Γ)
            end
            for m in 1:nmodes
                ρs[Ei] += lor(E-Evals_down[m,qi],Γ)
            end  
        end
    end
    ρs /= Nqs
    function density(μ)
        return (sum(ρs .* fermi.(energies, μ, kT)) * dE)
    end
    loss(μ) = (density(μ) - ne)^2

    if isnothing(mu_bounds)
        mu_bounds = Elims
    end

    res = Optim.optimize(loss, mu_bounds[1], mu_bounds[2], Optim.Brent();rel_tol)
        μ_solution = Optim.minimizer(res)
    return μ_solution
end


function self_consistency(mft::MeanFieldTheory,dq::Float64,nEs;Γ=0.1, tol=1e-2,mu_bounds =nothing,rel_tol,kT,tol_sc,α,niters=200)
    (; sys,ne,μ, mean_fields) = mft
    qs = Sunny.make_q_grid(sys, dq)  
    Nq = length(qs)
    Na = length(mean_fields)
    for ii in 1:niters
        println("Beginning $ii")
        μ =  find_chemical_potential(mft,dq,ne,nEs;Γ, tol,mu_bounds ,rel_tol,kT)
        mft.μ = μ # maybe not necessary - it only updates fermi
        println(mft.μ)
        mean_fields_new = fill([0.0, 0.0], size(mean_fields)...)
        for qi in 1:Nq
            q = qs[qi]
            Hup, Hdown = Sunny.mean_field_hamiltonians(mft,q)
            Eus, Uus = eigen(Hup)    
            Eds, Uds = eigen(Hdown)    
            for m ∈ 1:Na
                for i ∈ 1:Na
                    mean_fields_new[m][2] +=real(abs2(Uds[m,i])*fermi(Eds[i],μ,kT))
                    mean_fields_new[m][1] += real(abs2(Uus[m,i])*fermi(Eus[i],μ,kT))
                end
            end
        end
        mean_fields_new /= Nq
        mean_fields_new /= Nq
        diff = norm(mean_fields_new - sys.mean_fields) /norm(mean_fields_new)
        
        if diff < tol_sc
            println("converged after $ii iterations, diff: $diff") 
            println("mean fields: $mean_fields_new")
            return mean_fields_new #; break
        end
        mft.mean_fields[1] = (1-α)*mean_fields_new[1]+α*mean_fields_new[2]
        mft.mean_fields[2] = (1-α)*mean_fields_new[2]+α*mean_fields_new[1]
        println("Not converged: iteration $ii, diff: $diff")
    end
end