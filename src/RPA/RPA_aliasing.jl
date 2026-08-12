function clone_system(sys::ElectronicSystem)
    (; origin, crystal, dims, extfield, interactions_union,
        mean_fields, rng) = sys

    origin_clone = isnothing(origin) ? nothing : clone_system(origin)


    # Dynamically dispatch to the correct `map` function for either homogeneous
    # (Vector) or inhomogeneous interactions (4D Array)
    interactions_clone = map(clone_interactions, interactions_union)

    ret = ElectronicSystem(origin_clone, crystal, dims,
                  interactions_clone, copy(extfield),
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

function get_mean_field_buffers(sys::ElectronicSystem, numrequested) 
    numexisting = length(sys.mean_field_buffers)
    if numexisting < numrequested
        for _ in 1:(numrequested-numexisting)
            push!(sys.mean_field_buffers, zero(sys.mean_fields))
        end
    end
    return view(sys.mean_field_buffers, 1:numrequested)
end

# TODO Figure this out - how are we doing anisotropies
function empty_interactions(Na)
    # Cannot use `fill` because the PairCoupling arrays must be
    # allocated separately for later mutation.
    return map(1:Na) do _
        ElectronicInteractions(empty_electronic_anisotropy(),0.0, PairCoupling[])
    end
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

    # The `origin` system always has dims=(1, 1, 1) and uses the original
    # crystal. Perform a clone because mutable updates to interactions in the
    # reshaped system will also update interactions in its `origin` system.
    orig_sys = clone_system(@something sys.origin sys)

    new_sys = ElectronicSystem(orig_sys, new_cryst, new_dims,
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

function allapproxequal(a; kwargs...)
    mean = sum(a; init=0.0) / length(a)
    all(x -> isapprox(mean, x), a)
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

struct RPA 
    sys            :: ElectronicSystem
    # data           :: Union{SWTDataDipole, SWTDataSUN}
    measure        :: MeasureSpec
    regularization :: Float64
end

function RPA(sys::ElectronicSystem; measure::Union{Nothing, MeasureSpec}, regularization=1e-8, energy_ϵ=nothing)
    if !isnothing(energy_ϵ)
        @warn "Keyword argument energy_ϵ is deprecated! Use `regularization` instead."
        regularization = energy_ϵ
    end

    measure = @something measure empty_measurespec(sys)
    if size(eachsite(sys)) != size(measure.observables)[2:5]
        error("Size mismatch. Check that measure is built using consistent system.")
    end

    # Create a single enlarged chemical cell that matches the full system size
    # while preserving the system's linear site order.
    new_cryst = resize_and_flatten_crystal(sys.crystal, sys.dims)

    # Create a new system with dims (1,1,1). A clone happens in all cases.
    sys = reshape_supercell_aux(sys, new_cryst, (1,1,1))

    # Rotate local operators to quantization axis
    # data = swt_data(sys, measure)
    # TODO figure out data for this struct
    # return SpinWaveTheory(sys, data, measure, regularization)
    return RPA(sys, measure, regularization)
end


function all_dipole_observables(sys::ElectronicSystem; )
    observables = zeros(Vec3, 3, size(eachsite(sys))...)
    for site in eachsite(sys)
        # Component α of observable is op⋅S = g[α,β] S[β]. Minus sign would
        # cancel because observables come in pairs.
        op =  Mat3(I)
        for α in 1:3
            observables[α, site] = op[α, :]
        end
    end
    return observables
end

function ssf_custom(f, sys::ElectronicSystem;  formfactors=nothing)
    observables = all_dipole_observables(sys; )
    corr_pairs = [(3,3), (2,3), (1,3), (2,2), (1,2), (1,1)]
    combiner(q, corr) = f(q, SA[
        corr[6]       corr[5]       corr[3]
        conj(corr[5]) corr[4]       corr[2]
        conj(corr[3]) conj(corr[2]) corr[1]
    ])
    formfactors = if isnothing(formfactors)
        fill(one(FormFactor), natoms(sys.crystal))
    else
        formfactors isa Vector{Pair{Int, FormFactor}} || error("Pass formfactors as [i1 => FormFactor(...), i2 => ...]")
        propagate_atom_data(orig_crystal(sys), sys.crystal, formfactors)
    end
    return MeasureSpec(observables, corr_pairs, combiner, formfactors)
end


function ssf_custom_bm(f, sys::ElectronicSystem; u, v, formfactors=nothing)
    u = orig_crystal(sys).recipvecs * u
    v = orig_crystal(sys).recipvecs * v
    e3 = normalize(u × v)

    return ssf_custom(sys::ElectronicSystem;  formfactors) do q, ssf
        if iszero(q)
            error("Blume-Maleev axis system not defined at zero q")
        end
        if abs(q ⋅ e3) > 1e-12
            error("Momentum transfer q not in scattering plane")
        end
        e1 = normalize(q)      # parallel to q
        e2 = normalize(e3 × q) # perpendicular to q, in the scattering plane
        bm = hcat(e1, e2, e3)  # Blume-Maleev axis system
        f(Vec3(norm(q), 0, 0), bm' * ssf * bm)
    end
end

"""
    ssf_perp(sys::System; apply_g=true, formfactors=nothing)

Specify measurement of the spin structure factor with contraction by
``(I-𝐪⊗𝐪/q^2)``. The contracted value provides an estimate of unpolarized
scattering intensity. In the singular limit ``𝐪 → 0``, the contraction matrix
is replaced by its rotational average, ``(2/3) I``.

This function is a special case of [`ssf_custom`](@ref).

# Example

```julia
# Select Co²⁺ form factor for atom 1 and its symmetry equivalents
formfactors = [1 => FormFactor("Co2")]
ssf_perp(sys; formfactors)
```
"""
function ssf_perp(sys::ElectronicSystem;  formfactors=nothing)
    return ssf_custom(sys;  formfactors) do q, ssf
        q2 = norm2(q)
        # Imaginary part vanishes in symmetric contraction
        ssf = real(ssf)
        # "S-perp" contraction matrix (1 - q⊗q/q²) appropriate to unpolarized
        # neutrons. In the limit q → 0, use (1 - q⊗q/q²) → 2/3, which
        # corresponds to a spherical average over uncorrelated data:
        # https://github.com/SunnySuite/Sunny.jl/pull/131
        (iszero(q2) ? (2/3)*tr(ssf) : tr(ssf) - dot(q, ssf, q) / q2)
    end
end

"""
    ssf_trace(sys::System; apply_g=true, formfactors=nothing)

Specify measurement of the spin structure factor, with trace over spin
components. This quantity can be useful for checking quantum sum rules.

This function is a special case of [`ssf_custom`](@ref).
"""
function ssf_trace(sys::ElectronicSystem; formfactors=nothing)
    return ssf_custom(sys; formfactors) do q, ssf
        tr(real(ssf))
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


mutable struct MeanFieldTheory 
    sys            :: ElectronicSystem
    ne             :: Float64
    μ              :: Float64
    mean_fields    :: Array{Vec2, 4}
    regularization :: Float64
end

function MeanFieldTheory(sys::ElectronicSystem, ne::Float64, μ::Float64;  regularization=1e-8)

    # Create a single enlarged chemical cell that matches the full system size
    # while preserving the system's linear site order.
    new_cryst = resize_and_flatten_crystal(sys.crystal, sys.dims)

    # Create a new system with dims (1,1,1). A clone happens in all cases.
    sys = reshape_supercell_aux(sys, new_cryst, (1,1,1))
    na = natoms(sys.crystal)
    mean_fields = fill(zero(Vec2), 1, 1, 1, na)

    # TODO Add options for the self consistency
    return MeanFieldTheory(sys, ne, μ, mean_fields, regularization)
end
