function clone_system(sys::ElectronicSystem)
    (; origin, crystal, dims, extfield, interactions_union,
       params, active_labels, mean_fields, rng) = sys

    origin_clone = isnothing(origin) ? nothing : clone_system(origin)

    # Copy element-wise because each param has a mutable val
    params_clone = copy.(params)

    # Dynamically dispatch to the correct `map` function for either homogeneous
    # (Vector) or inhomogeneous interactions (4D Array)
    interactions_clone = map(clone_interactions, interactions_union)

    # Empty buffers are required for thread safety.
    empty_mean_field_buffers = Array{Vec2, 4}[]

    ret = ElectronicSystem(origin_clone, crystal, dims,
                 params_clone, active_labels, interactions_clone, copy(extfield),
                 copy(mean_fields), empty_mean_field_buffers
                 , copy(rng))

    return ret
end

@inline eachsite(sys::ElectronicSystem) = CartesianIndices(sys.mean_fields)
@inline eachsite_sublattice(sys::ElectronicSystem, i) = CartesianIndices((sys.dims..., i:i))
nsites(sys::ElectronicSystem) = length(sys.mean_fields)

# Number of (original) crystal cells in the system
ncells(sys::ElectronicSystem) = nsites(sys) / natoms(orig_crystal(sys))

function global_positions(sys::ElectronicSystem)
    mappedarray(site -> global_position_at(sys, site), eachsite(sys))
end


# Total volume of system
volume(sys::ElectronicSystem) = cell_volume(sys.crystal) * prod(sys.dims)


# Position of a site in global Cartesian coordinates
function global_position_at(sys::ElectronicSystem, site)
    site = to_cartesian(site)
    r = sys.crystal.positions[site[4]] + Vec3(site[1]-1, site[2]-1, site[3]-1)
    return sys.crystal.latvecs * r
end

# Position of a site in units of lattice vectors for the original crystal.
function position_at(sys::ElectronicSystem, site)
    return orig_crystal(sys).latvecs \ global_position_at(sys, site)
end


function position_to_site(sys::ElectronicSystem, r; tol=1e-12)
    # convert to fractional coordinates of possibly reshaped crystal
    r = Vec3(r)
    new_r = sys.crystal.latvecs \ orig_crystal(sys).latvecs * r
    i, offset = position_to_atom_and_offset(sys.crystal, new_r; tol)
    cell = @. mod1(offset+1, sys.dims) # 1-based indexing with periodicity
    return to_cartesian((cell..., i))
end

function symmetry_equivalent_bonds(sys::ElectronicSystem, bond::Bond)
    ret = Tuple{Site, Site, SVector{3, Int}}[]

    for new_i in 1:natoms(sys.crystal)
        # atom index in original crystal
        i = map_atom_to_other_crystal(sys.crystal, new_i, orig_crystal(sys))

        # loop over symmetry equivalent bonds in original crystal
        for bond′ in all_symmetry_related_bonds_for_atom(orig_crystal(sys), i, bond)

            # map to a bond with indexing for new crystal
            new_bond = map_bond_to_other_crystal(orig_crystal(sys), bond′, sys.crystal, new_i)

            # loop over all new crystal cells and push site pairs
            for site_i in eachsite_sublattice(sys, new_bond.i)
                site_j = bonded_site(site_i, new_bond, sys.dims)
                site_i < site_j && push!(ret, (site_i, site_j, new_bond.n))
            end
        end
    end

    return ret
end

function copy_mean_fields!(dst::ElectronicSystem, src::ElectronicSystem)
    size(dst.mean_fields) == size(src.mean_fields) || error("Mismatched system sizes")
    copy!(dst.mean_fields, src.mean_fields)
    return dst
end

function get_mean_field_buffers(sys::ElectronicSystem, numrequested) 
    numexisting = length(sys.mean_field_buffers)
    if numexisting < numrequested
        for _ in 1:(numrequested-numexisting)
            push!(sys.mean_field_buffers, zero(sys.mean_fields))
        end
    end
    return view(sys.mean_field_buffers, 1:numrequested)
end


function repopulate_couplings_from_params!(sys::ElectronicSystem)
    @assert is_homogeneous(sys)
    ints = interactions_homog(sys)

    # If `sys` has been reshaped, then also repopulate `sys.origin` (useful for
    # view_crystal(sys)).
    if !isnothing(sys.origin)
        repopulate_couplings_from_params!(sys.origin)
    end

    # Clear current interactions
    for i in eachindex(ints)
        ints[i].onsite = ints[i].onsite * 0.0
        empty!(ints[i].pair)
    end

    # Accumulate from params
    for param in sys.params
        for (i, oc) in param.onsites
            ints[i].onsite += oc * param.val
        end

        for pc in param.pairs
            b = pc.bond
            scaled_pc = pc * param.val
            ints_pairs = ints[b.i].pair

            # Find existing entry for this bond and accumulate
            idx = findfirst(pc′ -> pc′.bond == b, ints_pairs)
            if isnothing(idx)
                push!(ints_pairs, scaled_pc)
            else
                ints_pairs[idx] += scaled_pc
            end
        end
    end

    # Non-culled couplings must come first to enable early `break`
    for (; pair) in ints
        sort!(pair, by = pc -> pc.isculled)
    end
end

# TODO Figure this out - how are we doing anisotropies
function empty_interactions(Na)
    # Cannot use `fill` because the PairCoupling arrays must be
    # allocated separately for later mutation.
    return map(1:Na) do _
        Interactions(empty_anisotropy(), PairCoupling[])
    end
end

function interactions_homog(sys::ElectronicSystem)
    return sys.interactions_union :: Vector{Interactions}
end

function interactions_inhomog(sys::ElectronicSystem) 
    return sys.interactions_union :: Array{Interactions, 4}
end

function is_homogeneous(sys::ElectronicSystem)
    return sys.interactions_union isa Vector{Interactions}
end

function to_inhomogeneous(sys::ElectronicSystem) 
    is_homogeneous(sys) || error("System is already inhomogeneous.")
    ints = interactions_homog(sys)

    ret = clone_system(sys)

    # TODO: Zero out params and interactions of ret.origin?

    # Params unsupported for inhomogeneous system
    empty!(ret.params)

    # Population interactions_union as 4D array
    na = natoms(ret.crystal)
    ret.interactions_union = Array{Interactions}(undef, ret.dims..., na)
    for site in eachsite(ret)
        ret.interactions_union[site] = clone_interactions(ints[to_atom(site)])
    end

    return ret
end

function repeat_periodically(sys::ElectronicSystem, counts::NTuple{3,Int})
    is_homogeneous(sys) || error("Cannot repeat inhomogeneous system.")
    all(>=(1), counts) || error("Require at least one count in each direction.")
    return reshape_supercell_aux(sys, sys.crystal, counts .* sys.dims)
end


function reshape_supercell_aux(sys::ElectronicSystem, new_cryst::Crystal, new_dims::NTuple{3, Int}) 
    # Allocate data for new system, but with an empty list of interactions
    new_na               = natoms(new_cryst)
    new_ints             = empty_interactions(:dipole, new_na, 0)
    new_params           = ModelParam[]
    new_ewald            = nothing
    new_extfield         = zeros(Vec3, new_dims..., new_na)
    new_mean_fields          = zeros(Vec2, new_dims..., new_na)
    new_mean_field_buffers   = Array{Vec2, 4}[]

    # The `origin` system always has dims=(1, 1, 1) and uses the original
    # crystal. Perform a clone because mutable updates to interactions in the
    # reshaped system will also update interactions in its `origin` system.
    orig_sys = clone_system(@something sys.origin sys)

    new_sys = ElectronicSystem(orig_sys, new_cryst, new_dims,
                     new_params, sys.active_labels, new_ints, new_extfield,
                     new_mean_fields, new_mean_field_buffers, 
                     copy(sys.rng))

    if is_homogeneous(sys)
        # Transfer params from `new_sys.origin`, which will then be used to fill
        # interactions.
        transfer_params_from_origin!(new_sys)
    else
        # Inhomogeneous interactions must be transferred directly. This path
        # only exists to support SpinWaveTheory reshaping.
        @assert new_sys.dims == (1, 1, 1)
        @assert nsites(new_sys) == nsites(sys)
        transfer_interactions_from_inhomogeneous!(new_sys, sys)
    end

    # Copy per-site quantities
    for new_site in eachsite(new_sys)
        site = position_to_site(sys, position_at(new_sys, new_site))
        new_sys.extfield[new_site]  = sys.extfield[site]
        new_sys.mean_fields[new_site]   = sys.mean_fields[site]
    end

    return new_sys
end


# TODO Fix this - just commented out whilst interactions not set up
# Transfer interactions from `sys.origin` to reshaped `sys`.
function transfer_params_from_origin!(sys::ElectronicSystem)
    #=
    @assert is_homogeneous(sys)
    (; origin) = sys

    # Map atom in origin crystal to vector of atoms in new crystal
    origin_to_new = Dict(i => Int[] for i in 1:natoms(origin.crystal))
    for new_i in 1:natoms(sys.crystal)
        i = map_atom_to_other_crystal(sys.crystal, new_i, origin.crystal)
        # Append `new_i` to the vector at `origin_to_new[i]`
        push!(origin_to_new[i], new_i)
    end

    empty!(sys.params)

    for param in origin.params
        new_onsites = empty(param.onsites)
        for (i, oc) in param.onsites
            for new_i in origin_to_new[i]
                push!(new_onsites, (new_i, oc))
            end
        end

        new_pairs = PairCoupling[]
        for pc in param.pairs
            i_old = pc.bond.i
            for new_i in origin_to_new[i_old]
                new_bond = map_bond_to_other_crystal(origin.crystal, pc.bond, sys.crystal, new_i)
                push!(new_pairs, PairCoupling(new_bond, pc.scalar, pc.bilin, pc.biquad, pc.general))
            end
        end

        push!(sys.params, ModelParam(param.label, param.val, new_onsites, new_pairs))
    end

    repopulate_couplings_from_params!(sys)
    return
    =#
    # DO NOTHING FOR NOW
end