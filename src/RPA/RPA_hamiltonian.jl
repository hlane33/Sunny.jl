
function lor(x, Γ)
    return (Γ/2) / (π*(x^2+(Γ/2)^2))
end

function get_bound_from_tol(tol,Γ)
    return sqrt(2Γ -π*(Γ^2)*tol)/(2*sqrt(tol*π))
end


# create wrapper that will become self consistency function but just to test
function self_consistent_aux(sys,ne;regularization=1e-12, nEs = 100, tol=1e-2, mu_bounds = nothing,dq = 0.1,Γ=0.1,kT,Niters=100,α=0.1)
    
    ########## THIS IS NEEDED TO SET UP INTERACTIONS
    # Create a single enlarged chemical cell that matches the full system size
    # while preserving the system's linear site order.
    new_cryst = resize_and_flatten_crystal(sys.crystal, sys.dims)
    # Create a new system with dims (1,1,1). A clone happens in all cases.
    sys = reshape_supercell_aux(sys, new_cryst, (1,1,1))

    # set up a q-grid for integration
    qs = make_q_grid(sys, dq)
    Nqs = length(qs)

    # get initial values for mean_fields as a starting point as well as interactions
    ints = sys.interactions_union
    mean_fields = map(_ -> SA[rand(), rand()], sys.mean_fields)

    # initialize Hamiltonian arrays
    Na = length(mean_fields)
    Hup = zeros(ComplexF64,Na,Na)
    Hdown = zeros(ComplexF64,Na,Na)
    
    # initialize arrays for chemical potential loop
    dos_array = zeros(Float64,nEs)
    Evals_up = zeros(Float64,Na,Nqs)
    Evals_down = zeros(Float64,Na,Nqs)
    Evecs_up = zeros(ComplexF64,Na,Na,Nqs)
    Evecs_down = zeros(ComplexF64,Na,Na,Nqs)
    E_ϵ =  get_bound_from_tol(tol,Γ) #some way to get bounds on the energy arrays (improve so that tol is meaningful everywhere)
    
    for ii in 1:Niters
        ###########################################
        mean_fields_new = zero(mean_fields)
        Evals_up .= 0.0
        Evals_down .= 0.0
        # create the eigenvalue arrays
        for qi in 1:Nqs
            q= qs[qi]
            mean_field_hamiltonians!(Hup,Hdown,ints,mean_fields,q;regularization)
            Eus, Euvecs = eigen(Hup)    
            Eds, Edvecs = eigen(Hdown) 
            Evals_up[:,qi] .= Eus
            Evals_down[:,qi] .= Eds
            Evecs_up[:,:,qi] .= Euvecs
            Evecs_down[:,:,qi] .= Edvecs
        end 

        #build the energy grid
        Elims = extrema(hcat(Evals_up,Evals_down)) .+ (-E_ϵ,E_ϵ)
        energies = range(Elims[1],Elims[2];length=nEs)
        dE = step(energies)

        dos_array .= 0.0
        for (Ei,E) in enumerate(energies)
            for qi in 1:Nqs
                for m in 1:Na
                    dos_array[Ei] += lor(E-Evals_up[m,qi],Γ)
                end
                for m in 1:Na
                    dos_array[Ei] += lor(E-Evals_down[m,qi],Γ)
                end  
            end
        end
        dos_array /= Nqs
        function density(μ)
            return (sum(dos_array .* fermi.(energies, μ, kT)) * dE)
        end
        loss(μ) = (density(μ) - ne)^2

        new_mu_bounds = isnothing(mu_bounds) ? Elims : mu_bounds
        res = Optim.optimize(loss, new_mu_bounds[1], new_mu_bounds[2], Optim.Brent();rel_tol = tol)
        μ = Optim.minimizer(res)
        ########################################
        for qi in 1:Nqs
            for m ∈ 1:Na
                    for i ∈ 1:Na
                        mean_fields_new[m] +=SA[real(abs2(Evecs_up[m,i,qi])*fermi(Evals_up[i,qi],μ,kT))
                        , real(abs2(Evecs_down[m,i,qi])*fermi(Evals_down[i,qi],μ,kT))]
                    end
            end
        end
        mean_fields_new /= Nqs
        diff = norm(mean_fields_new - mean_fields) /norm(mean_fields_new)
        
        if diff < tol
            println("converged after $ii iterations, diff: $diff") 
            println("mean fields: $mean_fields_new")
            return mean_fields_new, μ #; break
        end
        println("step $ii/$Niters: μ = $μ, diff: $diff")
        mean_fields = (1-α).*mean_fields .+ α.*mean_fields_new
    end
    println("not converged after $Niters iterations!")
end

function mean_field_self_consistency!(sys::ElectronicSystem,ne;regularization=1e-12, nEs = 100, tol=1e-2, mu_bounds = nothing,dq = 0.1,Γ=0.1,kT,Niters=100,α=0.1)
    mean_fields_new, μ = self_consistent_aux(sys,ne;regularization=1e-12, nEs = 100, tol=1e-2, mu_bounds = nothing,dq = 0.1,Γ=0.1,kT,Niters=100,α=0.1)
    for m in 1:length(mean_fields_new)
        sys.mean_fields[m] = mean_fields_new[m]
    end
    println("Updated mean fields!")
end

function fermi(E,μ,kT)
    return 1/(exp((E-μ)/kT)+1)
end


function mean_field_hamiltonians_test(sys,q_reshaped;regularization=1e-12)
    mean_fields = sys.mean_fields
    Na = length(mean_fields)
    Hup = zeros(ComplexF64,Na,Na)
    Hdown = zeros(ComplexF64,Na,Na)
    mean_field_hamiltonians!(Hup,Hdown,sys.interactions_union,sys.mean_fields,q_reshaped;regularization)
    return Hup, Hdown 
end

function mean_field_hamiltonians!(Hup,Hdown,ints,mean_fields,q_reshaped::Vec3;regularization)
    Hup .= 0.0
    Hdown .= 0.0
    Na = length(mean_fields)
    for (i, int) in enumerate(ints)

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
                Hup[i, j] += -t * phase
                Hdown[i, j] += -t * phase
                Hup[j, i] += conj(-t) * conj(phase)
                Hdown[j, i] += conj(-t) * conj(phase)
            end
        end
        for hub in int.hubbard
            Hup[i,i] += hub*mean_fields[i][2]
            Hdown[i,i] += hub*mean_fields[i][1]
        end
        for pot in int.onsite 
            Hup[i,i] += pot
            Hdown[i,i] += pot
        end
    end
    # H must be hermitian up to round-off errors
    @assert diffnorm2(Hup, Hup') < 1e-12
    @assert diffnorm2(Hdown, Hdown') < 1e-12

    # Make H exactly hermitian
    hermitianpart!(Hup)
    hermitianpart!(Hdown)

    # Add small constant shift for positive-definiteness
    for i in 1:Na
        Hup[i, i] += regularization
        Hdown[i, i] += regularization
    end
    return Hup, Hdown
end
