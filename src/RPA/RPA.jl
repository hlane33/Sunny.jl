# TODO make Abstract system, interactions and couplings to share the overhead
#=
mutable struct ElectronicSystem
    const origin           :: Union{Nothing, ElectronicSystem} # System for the original chemical cell

    const crystal          :: Crystal
    const dims             :: NTuple{3, Int}            # Dimensions of lattice in unit cells


    # Interactions may be homogeneous (defined for one unit cell), or
    # inhomogeneous (defined for every cell in the system).
    interactions_union     :: Union{Vector{ElectronicInteractions}, Array{ElectronicInteractions, 4}}


    # Dynamical variables and buffers (dims × natoms)
    const extfield         :: Array{Vec3, 4}            # External B field
    const mean_fields          :: Array{Vec2, 4}            # Expected dipoles
    const mean_field_buffers   :: Vector{Array{Vec2, 4}}    # Buffers for dynamics routines

    # Global data
    const rng              :: Random.Xoshiro
end

function ElectronicSystem(crystal::Crystal;
                dims::NTuple{3,Int}=(1, 1, 1), seed=nothing, units=nothing)
    if !isnothing(units)
        @warn "units argument to System is deprecated and will be ignored!"
    end

    # Symops must be non-empty
    validate_symops(crystal)

    # Crystal lattice vectors must be standard (crystal not reshaped)
    @assert isnothing(crystal.root) || crystal.latvecs == crystal.root.latvecs

    na = natoms(crystal)
    interactions = empty_interactions(mode, na, N)

    mean_fields = fill(zero(Vec2), 1, 1, 1, na)
    mean_field_buffers = Array{Vec2, 4}[]

    rng = isnothing(seed) ? Random.Xoshiro(rand(UInt64, 4)...) : Random.Xoshiro(seed)

    ret = System(nothing, crystal, (1, 1, 1),
                 interactions, extfield, mean_fields, mean_field_buffers, rng)
    return dims == (1, 1, 1) ? ret : repeat_periodically(ret, dims)
end

# Pair couplings are counted only once per bond
struct ElectronicPairCoupling
    isculled :: Bool # Bond directionality is used to avoid double counting
    bond     :: Bond

    hopping   :: Float64            

    function ElectronicPairCoupling(bond, hopping)
        return new(bond_parity(bond), bond, hopping)
    end
end

function dos(E,sys::System,t,U,Vs,Δus,Δds;Γ=0.1,dq= 0.05)
    ρ = 0.0
    qs = Sunny.make_q_grid(sys,dq)  
    Nqs = length(qs)
    for qi in 1:Nqs
        q= qs[qi]
        Eus, Uus = eigen(dynamical_matrix(sys,q,t,U,Vs,Δds))    
        Eds, Uds = eigen(dynamical_matrix(sys,q,t,U,Vs,Δus)) 
        for m in 1:length(Eus)
            ρ += lor(E-Eus[m],Γ)
        end
        for m in 1:length(Eds)
            ρ += lor(E-Eds[m],Γ)
        end  
    end
    ρ /= Nqs
    return ρ
end


function set_hopping_aux!(sys::ElectronicSystem, hopping::Float64, bond::Bond)
    @assert is_homogeneous(sys)

    # If `sys` has been reshaped, then operate first on `sys.origin`, which
    # contains full symmetry information.
    if !isnothing(sys.origin)
        set_hopping_aux!(sys.origin, hopping,  bond)
        return
    end

    # Simple checks on bond indices
    validate_bond(sys.crystal, bond)

    # Propagate pair couplings by symmetry
    pairs = ElectronicPairCoupling[]
    for i in 1:natoms(sys.crystal)
        for bond′ in all_symmetry_related_bonds_for_atom(sys.crystal, i, bond)
            # TODO in general case we willl need to transform_coupling_for_bonds, but since t is scalar for now, we do not
            push!(pairs, ElectronicPairCoupling(bond′, t))
        end
    end
    return
end

mutable struct ElectronicInteractions
    # Onsite coupling is either an N×N Hermitian matrix or possibly renormalized
    # Stevens coefficients, depending on the mode :SUN or :dipole.
    onsite :: ElectronicOnsiteCoupling
    # Pair couplings for every bond that starts at the given atom
    pair :: Vector{ElectronicPairCoupling}
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
    ret.interactions_union = Array{ElectronicInteractions}(undef, ret.dims..., na)
    for site in eachsite(ret)
        ret.interactions_union[site] = clone_interactions(ints[to_atom(site)])
    end

    return ret
end

function empty_electronic_interactions(Na::Int)
    # Cannot use `fill` because the PairCoupling arrays must be
    # allocated separately for later mutation.
    return map(1:Na) do _
        # TODO decide what to do with onsite terms 
        # ElectronicInteractions(empty_anisotropy(mode, N), ElectronicPairCoupling[])
    end
end

function interactions_inhomog(sys::ElectronicSystem) 
    return sys.interactions_union :: Array{Interactions, 4}
end

function clone_interactions(int::ElectronicInteractions)
    (; onsite, pair) = int
    return ElectronicInteractions(onsite, copy(pair))
end

function interactions_homog(sys::ElectronicSystem)
    return sys.interactions_union :: Vector{ElectronicInteractions}
end

function is_homogeneous(sys::ElectronicSystem)
    return sys.interactions_union isa Vector{ElectronicInteractions}
end

function clone_electronic_system(sys::ElectronicSystem) 
    (; origin, crystal, dims, interactions_union, extfield,
      mean_fields, rng) = sys

    origin_clone = isnothing(origin) ? nothing : clone_electronic_system(origin)

    # Dynamically dispatch to the correct `map` function for either homogeneous
    # (Vector) or inhomogeneous interactions (4D Array)
    interactions_clone = map(clone_electronic_interactions, interactions_union)

    # Empty buffers are required for thread safety.
    empty_mean_field_buffers = Array{Vec2, 4}[]

    ret = System(origin_clone,  crystal, dims, interactions_clone, copy(extfield),
                 copy(mean_fields),  empty_mean_field_buffers, copy(rng))


    return ret
end

 =#