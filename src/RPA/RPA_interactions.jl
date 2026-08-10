function set_onsite_potential!(sys::ElectronicSystem, V::Vector{Float64}, i::Int)
    is_homogeneous(sys) || error("Use `set_onsite_potential_at!` for an inhomogeneous system.")
    ints = interactions_homog(sys)

    # If `sys` has been reshaped, then operate first on `sys.origin`, which
    # contains full symmetry information.
    if !isnothing(sys.origin)
        set_onsite_potential!(sys.origin, V, i)
        transfer_interactions!(sys, sys.origin)
        return
    end

    @assert isnothing(sys.origin)
    (1 <= i <= natoms(sys.crystal)) || error("Atom index $i is out of range.")

    if !iszero(ints[i].onsite)
        println("Overriding potential for atom $i.")
    end

    onsite = V


    cryst = sys.crystal
    for j in all_symmetry_related_atoms(cryst, i)
        ints[j].onsite = onsite
    end
end

function set_onsite_potential_at!(sys::ElectronicSystem, V, site::Site)
    is_homogeneous(sys) && error("Use `to_inhomogeneous` first.")

    ints = interactions_inhomog(sys)
    site = to_cartesian(site)
    ints[site].onsite = V
end

function set_hopping_at_aux!(sys::ElectronicSystem, hopping::Float64, site1::Site, site2::Site, offset)
    is_homogeneous(sys) && error("Use `to_inhomogeneous` first.")
    # (is_vacant(sys, site1) || is_vacant(sys, site2)) && error("Cannot couple vacant site") # Haven't added vacancies yet
    ints = interactions_inhomog(sys)


    site1 = to_cartesian(site1)
    site2 = to_cartesian(site2)
    bond = sites_to_internal_bond(sys, site1, site2, offset)

    replace_coupling!(ints[site1].pair, ElectronicPairCoupling(bond, hopping))
    replace_coupling!(ints[site2].pair, ElectronicPairCoupling(reverse(bond), hopping))
end

function set_hopping_at!(sys::ElectronicSystem, hopping::Float64, site1::Site, site2::Site; offset=nothing) 
    set_hopping_at_aux!(sys, hopping, site1, site2, offset)
    return
end

function set_hopping_aux!(sys::ElectronicSystem, hopping::Float64, bond::Bond)
    # If `sys` has been reshaped, then operate first on `sys.origin`, which
    # contains full symmetry information.
    if !isnothing(sys.origin)
        set_hopping_aux!(sys.origin, hopping, bond)
        transfer_interactions!(sys, sys.origin)
        return
    end

    # Simple checks on bond indices
    validate_bond(sys.crystal, bond)

    # Print a warning if an interaction already exists for bond
    ints = interactions_homog(sys)
    if any(x -> x.bond == bond, ints[bond.i].pair)
        println("Overriding coupling for $bond.")
    end

  

    # Propagate all couplings by symmetry
    for i in 1:natoms(sys.crystal)
        for bond′ in all_symmetry_related_bonds_for_atom(sys.crystal, i, bond)
            replace_coupling!(ints[i].pair, ElectronicPairCoupling(bond′, hopping))
        end
    end
end

# Internal function only
function replace_coupling!(list, coupling::ElectronicPairCoupling; accum=false)
    (; bond) = coupling

    # Find and remove existing couplings for this bond
    idxs = findall(c -> c.bond == bond, list)
    existing = list[idxs]
    deleteat!(list, idxs)

    # If the new coupling is exactly zero, and we're not accumulating, then
    # return early
    iszero(coupling.hopping) && !accum && return

    # Optionally accumulate to an existing PairCoupling
    if accum && !isempty(existing)
        coupling += only(existing)
    end

    # Add to the list and sort by isculled. Sorting after each insertion will
    # introduce quadratic scaling in length of `couplings`. If this becomes
    # slow, we could swap two PairCouplings instead of performing a full sort.
    push!(list, coupling)
    sort!(list, by=c->c.isculled)

    return
end

function transfer_interactions!(sys::ElectronicSystem, src::ElectronicSystem)
    new_ints = interactions_homog(sys)

    for new_i in 1:natoms(sys.crystal)
        # Find `src` interaction either through an atom index or a site index
        if is_homogeneous(src)
            i = map_atom_to_other_crystal(sys.crystal, new_i, src.crystal)
        else
            i = map_atom_to_other_system(sys.crystal, new_i, src)
        end
        src_int = src.interactions_union[i]

        # Copy onsite couplings
        new_ints[new_i].onsite = src_int.onsite

        # Copy hubbard 
        new_ints[new_i].hubbard = src_int.hubbard

        # Copy pair couplings
        new_pc = ElectronicPairCoupling[]
        for pc in src_int.pair
            new_bond = map_bond_to_other_crystal(src.crystal, pc.bond, sys.crystal, new_i)
            push!(new_pc, ElectronicPairCoupling(new_bond, pc.hopping))
        end
        new_pc = sort!(new_pc, by=c->c.isculled)
        new_ints[new_i].pair = new_pc
    end
end


# Creates a copy of the Vector of PairCouplings. This is useful when cloning a
# system; mutable updates to one clone should not affect the other.
function clone_interactions(int::ElectronicInteractions)
    (; onsite,hubbard , pair) = int
    return ElectronicInteractions(onsite, hubbard, copy(pair))
end

function set_hopping!(sys::ElectronicSystem, hopping, bond; ) 
    is_homogeneous(sys) || error("Use `set_hopping_at!` for an inhomogeneous system.")
    set_hopping_aux!(sys, hopping, bond)
    return
end

#stopgap
function empty_electronic_anisotropy(;)
        return zeros(Float64, 2)
end

function set_hubbard!(sys::ElectronicSystem, U::Float64, i::Int)
    is_homogeneous(sys) || error("Use `set_hubbard_at!` for an inhomogeneous system.")
    ints = interactions_homog(sys)

    # If `sys` has been reshaped, then operate first on `sys.origin`, which
    # contains full symmetry information.
    if !isnothing(sys.origin)
        set_hubbard!(sys.origin, U, i)
        transfer_interactions!(sys, sys.origin)
        return
    end

    @assert isnothing(sys.origin)
    (1 <= i <= natoms(sys.crystal)) || error("Atom index $i is out of range.")

    if !iszero(ints[i].hubbard)
        println("Overriding hubbard for atom $i.")
    end

    hubbard = U


    cryst = sys.crystal
    for j in all_symmetry_related_atoms(cryst, i)
        ints[j].hubbard = hubbard
    end
end

function set_hubbard_at!(sys::ElectronicSystem, U, site::Site)
    is_homogeneous(sys) && error("Use `to_inhomogeneous` first.")

    ints = interactions_inhomog(sys)
    site = to_cartesian(site)
    ints[site].hubbard = U
end