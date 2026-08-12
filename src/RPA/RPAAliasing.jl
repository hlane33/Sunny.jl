function clone_system(sys::ElectronicSystem)
    (; origin, crystal, dims, extfield, interactions_union,
        mean_fields,ne, chemical_potential, rng) = sys

    origin_clone = isnothing(origin) ? nothing : clone_system(origin)


    # Dynamically dispatch to the correct `map` function for either homogeneous
    # (Vector) or inhomogeneous interactions (4D Array)
    interactions_clone = map(clone_interactions, interactions_union)


    ret = ElectronicSystem(origin_clone, crystal, dims,
                  interactions_clone, copy(ne), copy(chemical_potential), copy(extfield),
                 copy(mean_fields)
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


function interactions_homog(sys::ElectronicSystem)
    return sys.interactions_union :: Vector{ElectronicInteractions}
end

function interactions_inhomog(sys::ElectronicSystem) 
    return sys.interactions_union :: Array{ElectronicInteractions, 4}
end

function is_homogeneous(sys::ElectronicSystem)
    return sys.interactions_union isa Vector{ElectronicInteractions}
end

function to_inhomogeneous(sys::ElectronicSystem) 
    is_homogeneous(sys) || error("System is already inhomogeneous.")
    ints = interactions_homog(sys)

    ret = clone_system(sys)

    # TODO: Zero out params and interactions of ret.origin?


    # Population interactions_union as 4D array
    na = natoms(ret.crystal)
    ret.interactions_union = Array{ElectronicInteractions}(undef, ret.dims..., na)
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
    new_ints             = empty_interactions(new_na)
    new_extfield         = zeros(Vec3, new_dims..., new_na)
    new_mean_fields          = zeros(Vec2, new_dims..., new_na)
    new_mean_field_buffers   = Array{Vec2, 4}[]

    # The `origin` system always has dims=(1, 1, 1) and uses the original
    # crystal. Perform a clone because mutable updates to interactions in the
    # reshaped system will also update interactions in its `origin` system.
    orig_sys = clone_system(@something sys.origin sys)

    new_sys = ElectronicSystem(orig_sys, new_cryst,ne, chemical_potential, new_dims,
                    new_ints, new_extfield,
                     new_mean_fields, 
                     copy(sys.rng))

    transfer_interactions!(new_sys, sys)

    # Copy per-site quantities
    for new_site in eachsite(new_sys)
        site = position_to_site(sys, position_at(new_sys, new_site))
        new_sys.extfield[new_site]  = sys.extfield[site]
        new_sys.mean_fields[new_site]   = sys.mean_fields[site]
    end

    return new_sys
end


function resize_supercell(sys::ElectronicSystem, latsize::NTuple{3,Int}) 
    return reshape_supercell(sys, diagm(collect(latsize)))
end

function reshape_supercell(sys::ElectronicSystem, shape)
    is_homogeneous(sys) || error("Cannot reshape inhomogeneous system.")

    orig = orig_crystal(sys)
    check_shape_commensurate(orig, shape)
    prim_cell = primitive_cell(orig)
    shape_in_prim = prim_cell \ shape
    @assert all_integer(shape_in_prim; tol=1e-12)
    shape_in_prim = round.(Int, shape_in_prim)

    # Unit cell for new system, in units of original unit cell.
    new_dims = NTuple{3, Int}(gcd.(eachcol(shape_in_prim)))
    new_shape = Mat3(shape * diagm(collect(inv.(new_dims))))
    new_cryst = reshape_crystal(orig_crystal(sys), new_shape)

    return reshape_supercell_aux(sys, new_cryst, new_dims)
end



function sites_to_internal_bond(sys::ElectronicSystem, site1::CartesianIndex{4}, site2::CartesianIndex{4}, n_ref) 
    (; crystal, dims) = sys

    n0 = to_cell(site2) .- to_cell(site1)

    # Try to build a bond with the provided offset n_ref
    if !isnothing(n_ref)
        if all(iszero, mod.(n_ref .- n0, dims))
            return Bond(to_atom(site1), to_atom(site2), n_ref)
        else
            cell1 = to_cell(site1)
            cell2 = to_cell(site2)
            error("""Cells $cell1 and $cell2 are not compatible with the offset
                     $n_ref for a system with dimensions $dims.""")
        end
    end

    # Otherwise, search over all possible wrappings of the bond
    ns = view([n0 .+ dims .* (i,j,k) for i in -1:1, j in -1:1, k in -1:1], :)
    bonds = map(ns) do n
        Bond(to_atom(site1), to_atom(site2), n)
    end
    distances = global_distance.(Ref(crystal), bonds)

    # Indices of bonds, from smallest to largest
    perm = sortperm(distances)

    # If one of the bonds is much shorter than all others by some arbitrary
    # `safety` factor, then return it
    safety = 4
    if safety * distances[perm[1]] < distances[perm[2]] - 1e-12
        return bonds[perm[1]]
    else
        n1 = bonds[perm[1]].n
        n2 = bonds[perm[2]].n
        error("""Ambiguous offset vector. Possibilities include $n1 and $n2.
                 Try using a bigger system size, or pass an explicit offset.""")
    end
end


# Maps atom `i` in `cryst` to the corresponding site in `other_sys`
function map_atom_to_other_system(cryst::Crystal, i, other_sys::ElectronicSystem)
    global_r = cryst.latvecs * cryst.positions[i]
    other_r = orig_crystal(other_sys).latvecs \ global_r
    return position_to_site(other_sys, other_r)
end

function to_standard_rlu(sys::ElectronicSystem, q_reshaped)
    return orig_crystal(sys).recipvecs \ (sys.crystal.recipvecs * q_reshaped)
end