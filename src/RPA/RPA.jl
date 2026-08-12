function ElectronicSystem(crystal::Crystal, ne::Float64, chemical_potential::Float64;
                dims::NTuple{3,Int}=(1, 1, 1), seed=nothing)

    # Symops must be non-empty
    validate_symops(crystal)

    # Crystal lattice vectors must be standard (crystal not reshaped)
    @assert isnothing(crystal.root) || crystal.latvecs == crystal.root.latvecs

    na = natoms(crystal)

    interactions = empty_interactions(mode, na, N) ### FIX


    extfield = zeros(Vec3, 1, 1, 1, na)
    mean_fields = fill(zero(Vec2), 1, 1, 1, na)

    rng = isnothing(seed) ? Random.Xoshiro(rand(UInt64, 4)...) : Random.Xoshiro(seed)

    ret = ElectronicSystem(nothing, crystal, (1, 1, 1),
                 interactions, ne, chemical_potential, mean_fields, rng)
    return dims == (1, 1, 1) ? ret : repeat_periodically(ret, dims)
end

function Base.show(io::IO, sys::ElectronicSystem) 
    print(io, "ElectronicSystem($(supercell_to_str(sys.dims, sys.crystal))")
end

function Base.show(io::IO, ::MIME"text/plain", sys::ElectronicSystem) 
    printstyled(io, "Electronic System \n"; bold=true, color=:underline)
    println(io, supercell_to_str(sys.dims, sys.crystal))
    if !isnothing(sys.origin) && cell_shape(sys) != cell_shape(sys.origin)
        shape = number_to_math_string.(cell_shape(sys))
        println(io, formatted_matrix(shape; prefix="Reshaped cell "))
    end
end

mutable struct ElectronicSystem
    const origin           :: Union{Nothing, ElectronicSystem} # System for the original chemical cell

    const crystal          :: Crystal
    const dims             :: NTuple{3, Int}            # Dimensions of lattice in unit cells


    # Interactions may be homogeneous (defined for one unit cell), or
    # inhomogeneous (defined for every cell in the system).
    interactions_union     :: Union{Vector{ElectronicInteractions}, Array{ElectronicInteractions, 4}}

    const ne                        :: Float64
    const chemical_potential        :: Float64

    # Dynamical variables and buffers (dims × natoms)
    const extfield         :: Array{Vec3, 4}            # External B field
    const mean_fields          :: Array{Vec2, 4}            # Expected dipoles

    # Global data
    const rng              :: Random.Xoshiro
end