function intensities_rpa(rpa::RPA, qpts; kT=0)
    (; sys, measure, regularization) = rpa
    @warn "Overriding whatever measure you chose and constructing the Sperp"
    qpts = convert(AbstractQPoints, qpts)
    cryst = orig_crystal(sys)
    rs_global = global_positions(sys)

    # Number of (magnetic) atoms in magnetic cell
    @assert sys.dims == (1,1,1)
    Na = nsites(sys)
    # Number of chemical cells in magnetic cell
    Ncells = Na / natoms(cryst)
    # Number of quasiparticle modes
    L = 2*length(sys.mean_fields)
    # Number of wavevectors
    Nq = length(qpts.qs)


    for (iq, q) in enumerate(qpts.qs)
        q_global = cryst.recipvecs * q
        for i in 1:Na
            r_global = rs_global[i] # + offsets[μ, i]
            # ff = get_swt_formfactor(measure, μ, i)
            Avec_pref[i] = exp(- im * dot(q_global, r_global))
            # Avec_pref[i] *= compute_form_factor(ff, norm2(q_global))
        end

        Hup = zeros(ComplexF64, L, L)
        Hdown = zeros(ComplexF64, L, L)
        mean_field_hamiltonians!(Hup,Hdown,sys.interactions_union,sys.mean_fields,q;regularization)
        E_up, V_up = eigen(Hup)
        E_down, V_down = eigen(Hdown)
         Avec = zeros(ComplexF64, Nobs)

        # Fill `intensity` array
        for μ in 1:L
            
        
        end


    end

    disp = reshape(disp, L, size(qpts.qs)...)
    intensity = reshape(intensity, L, size(qpts.qs)...)
    return BandIntensities(cryst, qpts, disp, intensity)
end

"""
    intensities!(data, swt::SpinWaveTheory, qpts; energies, kernel, kT=0)
    intensities!(data, sc::SampledCorrelations, qpts; energies, kernel=nothing, kT=0)

Like [`intensities`](@ref), but makes use of storage space `data` to avoid
allocation costs.
"""
function intensities!(data, swt::AbstractSpinWaveTheory, qpts; energies, kernel::AbstractBroadening, kT=0)
    qpts = convert(AbstractQPoints, qpts)
    @assert size(data) == (length(energies), size(qpts.qs)...)
    bands = intensities_bands(swt, qpts; kT)
    @assert eltype(bands) == eltype(data)
    broaden!(data, bands; energies, kernel)
    return Intensities(bands.crystal, bands.qpts, collect(Float64, energies), data)
end

